"""Trust event domain models — the data foundation for Layer 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ulid import ULID


class TrustEventType(StrEnum):
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE = "tool_failure"
    SCOPE_VIOLATION = "scope_violation"
    HITL_TRIGGERED = "hitl_triggered"
    HITL_RESOLVED = "hitl_resolved"
    ADMISSION_ADMIT = "admission_admit"
    ADMISSION_REJECT = "admission_reject"
    ADMISSION_DEFER = "admission_defer"
    ADMISSION_HITL = "admission_hitl"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"


class TrustSignalStrength(StrEnum):
    POSITIVE = "positive"  # evidence of good behaviour
    NEUTRAL = "neutral"  # informational — neither good nor bad
    NEGATIVE = "negative"  # evidence of risk or failure
    CRITICAL = "critical"  # scope violation, repeated failures


# Signal strength for each event type — used by 8.2 autonomy engine
EVENT_SIGNAL_MAP: dict[TrustEventType, TrustSignalStrength] = {
    TrustEventType.TOOL_SUCCESS: TrustSignalStrength.POSITIVE,
    TrustEventType.TOOL_FAILURE: TrustSignalStrength.NEGATIVE,
    TrustEventType.SCOPE_VIOLATION: TrustSignalStrength.CRITICAL,
    TrustEventType.HITL_TRIGGERED: TrustSignalStrength.NEUTRAL,  # working as designed
    TrustEventType.HITL_RESOLVED: TrustSignalStrength.POSITIVE,
    TrustEventType.ADMISSION_ADMIT: TrustSignalStrength.NEUTRAL,
    TrustEventType.ADMISSION_REJECT: TrustSignalStrength.NEGATIVE,
    TrustEventType.ADMISSION_DEFER: TrustSignalStrength.NEUTRAL,
    TrustEventType.ADMISSION_HITL: TrustSignalStrength.NEUTRAL,
    TrustEventType.TASK_COMPLETE: TrustSignalStrength.POSITIVE,
    TrustEventType.TASK_FAILED: TrustSignalStrength.NEGATIVE,
}


@dataclass
class TrustEvent:
    event_type: TrustEventType
    agent_id: str
    task_id: str
    task_type: str
    autonomy_level: str
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{ULID()}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def signal(self) -> TrustSignalStrength:
        return EVENT_SIGNAL_MAP.get(self.event_type, TrustSignalStrength.NEUTRAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "autonomy_level": self.autonomy_level,
            "signal": self.signal.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        meta_str = ""
        if self.event_type in (TrustEventType.TOOL_SUCCESS, TrustEventType.TOOL_FAILURE):
            tool = self.metadata.get("tool_name", "")
            meta_str = f" tool={tool}"
        elif self.event_type == TrustEventType.SCOPE_VIOLATION:
            tool = self.metadata.get("attempted_tool", "")
            meta_str = f" attempted={tool}"
        elif self.event_type == TrustEventType.HITL_TRIGGERED:
            trigger = self.metadata.get("trigger", "")
            meta_str = f" trigger={trigger}"
        return (
            f"TrustEvent {self.event_type.value} agent={self.agent_id} "
            f"task_type={self.task_type} signal={self.signal.value}{meta_str}"
        )
