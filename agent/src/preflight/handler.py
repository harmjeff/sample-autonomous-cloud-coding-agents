"""
AWS Lambda entry point for the pre-flight pipeline.

Handler: preflight.handler.handler
Runtime: Python 3.13 (ARM64)

Input event (JSON):
  task_id        — unique task identifier
  task_type      — task type (e.g. 'new_task', 'pr_iteration')
  instructions   — task instructions
  scope          — dict matching TaskScope fields
  trust          — dict matching TaskTrust fields
  memory         — dict matching TaskMemory fields
  blueprint_id   — optional blueprint override

Output (JSON):
  decision           — ADMIT | ADMIT_WITH_HITL | DEFER | REJECT
  task_state         — new | retry | continuation | resumption
  risk_tier          — low | medium | high | critical
  hitl_required_for  — list[str]
  rejection_reason   — str | null

Fail-open: if the pipeline raises an unexpected exception the handler
returns ADMIT so a handler crash never silently blocks task execution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Lazy imports — keep the module importable even if optional deps are absent
# ---------------------------------------------------------------------------

_pipeline_available = False
try:
    from preflight.pipeline import PreflightPipeline
    from registry.filesystem import FilesystemRegistryService

    _pipeline_available = True
except ImportError as _e:
    logger.warning("PreflightPipeline not available (%s) — will fail-open", _e)


# ---------------------------------------------------------------------------
# Request shims
# (AKW uses Pydantic BaseModel; we use plain dataclass shims here so the
#  Lambda handler has no Pydantic dependency.)
# ---------------------------------------------------------------------------


@dataclass
class _TaskContext:
    requester: str = "unknown"
    requester_task_id: str | None = None
    session_id: str | None = None
    priority: str = "normal"
    deadline_iso: str | None = None


@dataclass
class _TaskScope:
    allowed_tools: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    max_iterations: int = 20
    max_tokens: int = 50000
    write_to_ltm: bool = True
    external_write_permitted: bool = False


@dataclass
class _TaskTrust:
    admission_decision: str = "ADMIT"
    autonomy_level: str = "supervised"
    hitl_triggers: list[str] = field(default_factory=list)


@dataclass
class _TaskMemory:
    stm_session_id: str | None = None
    ltm_scope: list[str] = field(default_factory=list)
    ltm_required_capabilities: list[str] = field(default_factory=list)


@dataclass
class _TaskRequest:
    task_id: str
    task_type: str
    instructions: str
    context: _TaskContext = field(default_factory=_TaskContext)
    scope: _TaskScope = field(default_factory=_TaskScope)
    trust: _TaskTrust = field(default_factory=_TaskTrust)
    memory: _TaskMemory = field(default_factory=_TaskMemory)
    blueprint_id: str | None = None


def _build_request(event: dict[str, Any]) -> _TaskRequest:
    """Map the Lambda event dict to a _TaskRequest shim."""
    raw_scope = event.get("scope") or {}
    raw_trust = event.get("trust") or {}
    raw_memory = event.get("memory") or {}
    raw_context = event.get("context") or {}

    scope_fields = _TaskScope.__dataclass_fields__
    trust_fields = _TaskTrust.__dataclass_fields__
    memory_fields = _TaskMemory.__dataclass_fields__
    context_fields = _TaskContext.__dataclass_fields__

    return _TaskRequest(
        task_id=event.get("task_id", ""),
        task_type=event.get("task_type", ""),
        instructions=event.get("instructions", ""),
        context=_TaskContext(**{k: v for k, v in raw_context.items() if k in context_fields}),
        scope=_TaskScope(**{k: v for k, v in raw_scope.items() if k in scope_fields}),
        trust=_TaskTrust(**{k: v for k, v in raw_trust.items() if k in trust_fields}),
        memory=_TaskMemory(**{k: v for k, v in raw_memory.items() if k in memory_fields}),
        blueprint_id=event.get("blueprint_id"),
    )


# ---------------------------------------------------------------------------
# Module-level pipeline singleton (reused across warm Lambda invocations)
# ---------------------------------------------------------------------------

_pipeline: Any = None  # PreflightPipeline | None, typed as Any to avoid NameError


def _get_pipeline() -> Any:
    global _pipeline  # module-level singleton, intentionally reused across warm invocations
    if _pipeline is None and _pipeline_available:
        blueprints_dir = os.environ.get("BLUEPRINTS_DIR", "/var/task/blueprints")
        registry = FilesystemRegistryService(blueprints_dir=Path(blueprints_dir))
        _pipeline = PreflightPipeline(registry=registry)
    return _pipeline


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

_ADMIT_RESPONSE: dict[str, Any] = {
    "decision": "ADMIT",
    "task_state": "new",
    "risk_tier": "low",
    "hitl_required_for": [],
    "rejection_reason": None,
}


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """AWS Lambda handler for the pre-flight pipeline.

    Fail-open: any unhandled exception returns ADMIT so that a pipeline
    crash never silently blocks task execution.
    """
    _ = context  # unused: Lambda context object
    task_id = event.get("task_id", "<unknown>")

    try:
        pipeline = _get_pipeline()
        if pipeline is None:
            logger.warning(
                "preflight.handler: pipeline unavailable (import failed) — "
                "failing open for task_id=%s",
                task_id,
            )
            return _ADMIT_RESPONSE.copy()

        req = _build_request(event)
        brief = pipeline.run(req)

        return {
            "decision": brief.decision.value,
            "task_state": brief.task_state.value,
            "risk_tier": brief.risk_tier.value,
            "hitl_required_for": list(brief.hitl_required_for),
            "rejection_reason": brief.rejection_reason,
        }

    except Exception:
        logger.exception(
            "preflight.handler: unhandled exception — failing open for task_id=%s",
            task_id,
        )
        return _ADMIT_RESPONSE.copy()
