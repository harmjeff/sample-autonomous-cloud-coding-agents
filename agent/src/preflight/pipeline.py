"""
Pre-flight pipeline — runs before any execution agent is invoked.

Stages (sequential, halt on failure):
  1a. Readiness Check  — can we run? (preconditions)
  1b. Context Hydration — load LTM memories and KB scope
  1c. Risk Assessment  — blast radius + risk tier
  1d. Admission Policy — should we run? (policy)

Returns an ExecutionBrief containing the admission decision and
hydrated context for the orchestrator to pass to execution agents.

Phase 2: admission rubric is hardcoded (simple rules).
Phase 3: replace rubric with configurable, versioned YAML policy.

Ported from AKW src/preflight/pipeline.py.
Optional dependencies (trust, graduation) are wrapped in try/except
so the pipeline remains functional without them.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from preflight.models import (
    AdmissionDecision,
    ExecutionBrief,
    HydrationResult,
    ReadinessResult,
    RiskResult,
    RiskTier,
    TaskState,
)
from registry.filesystem import FilesystemRegistryService
from registry.models import BlueprintNotFoundError, MissingToolError, ToolNotReadyError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1a — Readiness Check
# ---------------------------------------------------------------------------

class ReadinessChecker:
    """Lightweight precondition checks — runs before any expensive work."""

    def __init__(self, registry: FilesystemRegistryService) -> None:
        self._registry = registry

    def check(self, req: Any) -> ReadinessResult:
        checks: dict[str, bool] = {}
        failures: list[str] = []

        # Task state classification
        task_state = self._classify_task_state(req)
        checks["task_state"] = True

        # Blueprint + tool resolution
        try:
            self._registry.resolve_task(req.task_type)
            checks["blueprint_and_tools"] = True
        except BlueprintNotFoundError:
            checks["blueprint_and_tools"] = False
            failures.append(f"No production blueprint for task type '{req.task_type}'")
        except MissingToolError as e:
            checks["blueprint_and_tools"] = False
            failures.append(f"Missing tool: {e.tool_name}")
        except ToolNotReadyError as e:
            checks["blueprint_and_tools"] = False
            failures.append(f"Tool not ready: {e.tool_id} (status={e.current_status})")

        # Scope sanity
        checks["scope_valid"] = bool(req.scope.max_iterations > 0 and req.scope.max_tokens > 0)
        if not checks["scope_valid"]:
            failures.append("Invalid scope: max_iterations or max_tokens must be > 0")

        # Trust check
        checks["admission_present"] = bool(req.trust.admission_decision)
        if not checks["admission_present"]:
            failures.append("Missing trust.admission_decision")

        ready = not failures
        return ReadinessResult(
            ready=ready,
            task_state=task_state,
            checks=checks,
            failure_reason="; ".join(failures) if failures else None,
        )

    @staticmethod
    def _classify_task_state(req: Any) -> TaskState:
        task_id = req.task_id or ""
        if "retry" in task_id.lower():
            return TaskState.RETRY
        if "resume" in task_id.lower():
            return TaskState.RESUMPTION
        if req.context.requester_task_id:
            return TaskState.CONTINUATION
        return TaskState.NEW


# ---------------------------------------------------------------------------
# Stage 1b — Context Hydration
# ---------------------------------------------------------------------------

class ContextHydrator:
    """Loads LTM memories and determines KB scope for the task."""

    def __init__(self, ltm: Any = None) -> None:
        self._ltm = ltm  # Optional LTMInterface — None = skip hydration

    def hydrate(self, req: Any) -> HydrationResult:
        memories = []
        kb_scope = list(req.memory.ltm_scope)

        if self._ltm and req.memory.ltm_required_capabilities:
            try:
                user_id = req.context.requester or "orchestrator"
                results = self._ltm.read(req.instructions[:200], user_id=user_id)
                memories = [{"content": r.content, "score": r.score} for r in results[:10]]
            except Exception as e:
                logger.warning("Context hydration LTM read failed: %s", e)

        task_context = {
            "requester": req.context.requester,
            "session_id": req.context.session_id,
            "priority": req.context.priority,
            "deadline": req.context.deadline_iso,
        }

        return HydrationResult(
            ltm_memories=memories,
            kb_scope=kb_scope,
            task_context=task_context,
        )


# ---------------------------------------------------------------------------
# Stage 1c — Risk Assessment
# ---------------------------------------------------------------------------

class RiskAssessor:
    """Scores task risk across reversibility, scope, external effects."""

    # Phase 2: hardcoded rules. Phase 3: configurable rubric YAML.
    _HIGH_RISK_TASK_TYPES: ClassVar[set[str]] = {
        "generate_tool",
        "send_email",
        "post_message",
        "write_document",
    }
    _WRITE_RISK_MULTIPLIER: ClassVar[float] = 2.0

    def assess(self, req: Any, readiness: ReadinessResult) -> RiskResult:
        factors: dict[str, float] = {}

        # Reversibility: external writes are high risk
        if req.scope.external_write_permitted:
            factors["external_write"] = 1.0
        else:
            factors["external_write"] = 0.0

        # Scope: high iteration count = larger blast radius
        iter_ratio = req.scope.max_iterations / 50.0
        factors["iteration_scope"] = min(iter_ratio, 1.0)

        # Task type inherent risk
        factors["task_type_risk"] = 0.7 if req.task_type in self._HIGH_RISK_TASK_TYPES else 0.2

        # LTM writes propagate to persistent memory
        factors["ltm_write_risk"] = 0.3 if req.scope.write_to_ltm else 0.0

        # Retry/resumption = higher risk (previous failure context)
        factors["task_state_risk"] = 0.3 if readiness.task_state in (
            TaskState.RETRY, TaskState.RESUMPTION
        ) else 0.0

        score = sum(factors.values()) / len(factors)

        if score >= 0.7:
            tier = RiskTier.CRITICAL
        elif score >= 0.5:
            tier = RiskTier.HIGH
        elif score >= 0.3:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.LOW

        blast_radius = []
        if req.scope.external_write_permitted:
            blast_radius.append("external systems (write)")
        if req.scope.write_to_ltm:
            blast_radius.append("long-term memory")
        blast_radius.append(f"up to {req.scope.max_iterations} LLM iterations")

        return RiskResult(
            tier=tier,
            blast_radius=blast_radius,
            risk_factors=factors,
            scale_estimate={
                "max_iterations": req.scope.max_iterations,
                "max_tokens": req.scope.max_tokens,
                "estimated_cost_tier": tier,
            },
        )


# ---------------------------------------------------------------------------
# Stage 1d — Admission Policy
# ---------------------------------------------------------------------------

class AdmissionPolicy:
    """
    Applies the admissions rubric to produce a final decision.

    When an AutonomyGraduationEngine is injected, autonomy thresholds are
    looked up dynamically from the agent's earned track record.
    Falls back to hardcoded thresholds when no engine is provided.
    """

    # Fallback hardcoded thresholds (used when no graduation engine is wired in)
    _FALLBACK_THRESHOLDS: ClassVar[dict[str, RiskTier]] = {
        "supervised":  RiskTier.HIGH,      # ADMIT up to HIGH, HITL at CRITICAL
        "autonomous":  RiskTier.CRITICAL,  # ADMIT up to CRITICAL
        "restricted":  RiskTier.MEDIUM,    # ADMIT up to MEDIUM, HITL at HIGH
    }

    # Risk tier that each autonomy level can admit without HITL
    _LEVEL_THRESHOLDS: ClassVar[dict[str, RiskTier]] = {
        "restricted":  RiskTier.MEDIUM,
        "supervised":  RiskTier.HIGH,
        "autonomous":  RiskTier.CRITICAL,
    }

    def __init__(self, graduation_engine: Any = None) -> None:
        # Optional AutonomyGraduationEngine — if None, falls back to hardcoded thresholds
        self._graduation = graduation_engine

    def decide(
        self,
        req: Any,
        readiness: ReadinessResult,
        hydration: HydrationResult,
        risk: RiskResult,
    ) -> tuple[AdmissionDecision, str | None]:
        """Returns (decision, rejection_reason)."""

        # Readiness failure → REJECT or DEFER
        if not readiness.ready:
            reason = readiness.failure_reason or "Readiness check failed"
            if "not ready" in reason.lower():
                return AdmissionDecision.DEFER, reason
            return AdmissionDecision.REJECT, reason

        # Resolve autonomy level: dynamic (graduation engine) or static (request field)
        agent_id = req.context.requester or "orchestrator"
        if self._graduation is not None:
            autonomy = self._graduation.get_autonomy_level(agent_id, req.task_type)
        else:
            autonomy = req.trust.autonomy_level or "supervised"

        threshold = self._LEVEL_THRESHOLDS.get(autonomy, RiskTier.HIGH)
        tier_order = [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL]
        tier_rank = {t: i for i, t in enumerate(tier_order)}

        risk_rank = tier_rank[risk.tier]
        threshold_rank = tier_rank[threshold]

        # External writes on restricted autonomy → REJECT
        if req.scope.external_write_permitted and autonomy == "restricted":
            return (
                AdmissionDecision.REJECT,
                "External writes not permitted at restricted autonomy level",
            )

        # Above threshold → HITL
        if risk_rank > threshold_rank:
            return AdmissionDecision.ADMIT_WITH_HITL, None

        # HITL triggers from trust config
        if req.trust.hitl_triggers:
            return AdmissionDecision.ADMIT_WITH_HITL, None

        return AdmissionDecision.ADMIT, None


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class PreflightPipeline:
    """
    Runs all four pre-flight stages and returns an ExecutionBrief.
    Fails fast: each stage can halt the pipeline.
    """

    def __init__(
        self,
        registry: FilesystemRegistryService | None = None,
        ltm: Any = None,
        graduation_engine: Any = None,
    ) -> None:
        self._registry = registry or FilesystemRegistryService()
        self._readiness = ReadinessChecker(self._registry)
        self._hydrator = ContextHydrator(ltm)
        self._risk = RiskAssessor()
        self._admission = AdmissionPolicy(graduation_engine=graduation_engine)
        self._graduation = graduation_engine

    def run(self, req: Any) -> ExecutionBrief:
        logger.info("Pre-flight: task_id=%s type=%s", req.task_id, req.task_type)

        # 1a. Readiness
        readiness = self._readiness.check(req)
        logger.debug("Readiness: ready=%s state=%s", readiness.ready, readiness.task_state)

        if not readiness.ready:
            decision, reason = self._admission.decide(
                req, readiness, HydrationResult(), RiskResult(tier=RiskTier.LOW),
            )
            return ExecutionBrief(
                task_id=req.task_id,
                task_type=req.task_type,
                instructions=req.instructions,
                decision=decision,
                task_state=readiness.task_state,
                risk_tier=RiskTier.LOW,
                rejection_reason=reason or readiness.failure_reason,
            )

        # 1b. Context Hydration (only after readiness confirmed)
        hydration = self._hydrator.hydrate(req)
        logger.debug(
            "Hydration: %d memories, kb_scope=%s",
            len(hydration.ltm_memories),
            hydration.kb_scope,
        )

        # 1c. Risk Assessment
        risk = self._risk.assess(req, readiness)
        logger.debug("Risk: tier=%s factors=%s", risk.tier, risk.risk_factors)

        # 1d. Admission
        decision, reason = self._admission.decide(req, readiness, hydration, risk)
        logger.info("Admission: decision=%s reason=%s", decision, reason)

        # Emit trust event for admission decision (optional — trust module may not be present)
        try:
            from trust import get_emitter  # type: ignore[import-not-found]
            _te = get_emitter()
            if _te:
                _te.admission(
                    decision=decision.value,
                    agent_id=req.context.requester or "orchestrator",
                    task_id=req.task_id,
                    task_type=req.task_type,
                    autonomy_level=req.trust.autonomy_level,
                    risk_tier=risk.tier.value,
                    reason=reason,
                )
        except (ImportError, ModuleNotFoundError):
            pass  # trust module not available in ABCA — skip event emission

        # Evaluate graduation eligibility after every admission decision
        if self._graduation is not None:
            try:
                agent_id = req.context.requester or "orchestrator"
                grad_status = self._graduation.evaluate_and_flag(agent_id, req.task_type)
                if grad_status.eligible_for and grad_status.pending_human_approval:
                    logger.info(
                        "graduation: agent=%s task_type=%s eligible for %s — pending approval",
                        agent_id,
                        req.task_type,
                        grad_status.eligible_for,
                    )
            except Exception as e:
                logger.warning("Graduation engine evaluation failed (non-fatal): %s", e)

        hitl_triggers = (
            list(req.trust.hitl_triggers)
            if decision == AdmissionDecision.ADMIT_WITH_HITL
            else []
        )

        return ExecutionBrief(
            task_id=req.task_id,
            task_type=req.task_type,
            instructions=req.instructions,
            decision=decision,
            task_state=readiness.task_state,
            risk_tier=risk.tier,
            ltm_memories=hydration.ltm_memories,
            kb_scope=hydration.kb_scope,
            task_context=hydration.task_context,
            rejection_reason=reason,
            hitl_required_for=hitl_triggers,
        )
