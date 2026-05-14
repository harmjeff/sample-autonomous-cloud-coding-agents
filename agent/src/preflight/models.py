"""Pre-flight pipeline domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AdmissionDecision(StrEnum):
    ADMIT = "ADMIT"
    ADMIT_WITH_HITL = "ADMIT_WITH_HITL"
    DEFER = "DEFER"
    REJECT = "REJECT"


class TaskState(StrEnum):
    NEW = "new"
    RETRY = "retry"
    CONTINUATION = "continuation"
    RESUMPTION = "resumption"


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReadinessResult:
    ready: bool
    task_state: TaskState
    checks: dict[str, bool] = field(default_factory=dict)
    failure_reason: str | None = None


@dataclass
class HydrationResult:
    ltm_memories: list[dict[str, Any]] = field(default_factory=list)
    kb_scope: list[str] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskResult:
    tier: RiskTier
    blast_radius: list[str] = field(default_factory=list)
    risk_factors: dict[str, float] = field(default_factory=dict)
    scale_estimate: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionBrief:
    """Output of Phase 1 pre-flight — handed to Phase 2 execution."""

    task_id: str
    task_type: str
    instructions: str
    decision: AdmissionDecision
    task_state: TaskState
    risk_tier: RiskTier
    ltm_memories: list[dict[str, Any]] = field(default_factory=list)
    kb_scope: list[str] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None
    hitl_required_for: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return self.decision in (AdmissionDecision.ADMIT, AdmissionDecision.ADMIT_WITH_HITL)
