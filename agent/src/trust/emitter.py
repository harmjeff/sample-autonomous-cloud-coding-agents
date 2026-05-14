"""
TrustEventEmitter — thin wrapper that creates and writes TrustEvents.

Called from emission points throughout the system. Fails silently so
trust observation never blocks agent execution.

Usage:
    emitter = TrustEventEmitter(store)
    emitter.emit(
        TrustEventType.TOOL_SUCCESS,
        agent_id="generic-agent",
        task_id=task.task_id,
        task_type=task.task_type,
        autonomy_level="supervised",
        metadata={"tool_name": "web_search", "duration_ms": 420},
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from trust.models import TrustEvent, TrustEventType

if TYPE_CHECKING:
    from trust.store import TrustEventStore

logger = logging.getLogger(__name__)


class TrustEventEmitter:
    def __init__(self, store: TrustEventStore) -> None:
        self._store = store

    def emit(
        self,
        event_type: TrustEventType,
        agent_id: str,
        task_id: str,
        task_type: str,
        autonomy_level: str = "supervised",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create and persist a TrustEvent. Never raises."""
        try:
            event = TrustEvent(
                event_type=event_type,
                agent_id=agent_id,
                task_id=task_id,
                task_type=task_type,
                autonomy_level=autonomy_level,
                metadata=metadata or {},
            )
            self._store.write(event)
        except Exception as e:
            logger.warning("TrustEventEmitter.emit failed silently: %s", e)

    # ------------------------------------------------------------------
    # Convenience methods for common emission patterns
    # ------------------------------------------------------------------

    def tool_success(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        autonomy_level: str,
        tool_name: str,
        duration_ms: int,
    ) -> None:
        self.emit(
            TrustEventType.TOOL_SUCCESS,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"tool_name": tool_name, "duration_ms": duration_ms},
        )

    def tool_failure(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        autonomy_level: str,
        tool_name: str,
        error_type: str,
    ) -> None:
        self.emit(
            TrustEventType.TOOL_FAILURE,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"tool_name": tool_name, "error_type": error_type},
        )

    def scope_violation(
        self,
        agent_id: str,
        task_id: str,
        task_type: str,
        autonomy_level: str,
        attempted_tool: str,
        allowed: list[str],
    ) -> None:
        self.emit(
            TrustEventType.SCOPE_VIOLATION,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"attempted_tool": attempted_tool, "allowed_tools": allowed},
        )

    def hitl_triggered(
        self, agent_id: str, task_id: str, task_type: str, autonomy_level: str, trigger: str
    ) -> None:
        self.emit(
            TrustEventType.HITL_TRIGGERED,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"trigger": trigger},
        )

    def hitl_resolved(
        self, agent_id: str, task_id: str, task_type: str, autonomy_level: str, selected_option: str
    ) -> None:
        self.emit(
            TrustEventType.HITL_RESOLVED,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"selected_option": selected_option},
        )

    def task_complete(
        self, agent_id: str, task_id: str, task_type: str, autonomy_level: str, duration_ms: int
    ) -> None:
        self.emit(
            TrustEventType.TASK_COMPLETE,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"duration_ms": duration_ms},
        )

    def task_failed(
        self, agent_id: str, task_id: str, task_type: str, autonomy_level: str, error_code: str
    ) -> None:
        self.emit(
            TrustEventType.TASK_FAILED,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"error_code": error_code},
        )

    def admission(
        self,
        decision: str,
        agent_id: str,
        task_id: str,
        task_type: str,
        autonomy_level: str,
        risk_tier: str,
        reason: str | None = None,
    ) -> None:
        from trust.models import TrustEventType as T

        event_map = {
            "ADMIT": T.ADMISSION_ADMIT,
            "ADMIT_WITH_HITL": T.ADMISSION_HITL,
            "DEFER": T.ADMISSION_DEFER,
            "REJECT": T.ADMISSION_REJECT,
        }
        event_type = event_map.get(decision, T.ADMISSION_ADMIT)
        self.emit(
            event_type,
            agent_id,
            task_id,
            task_type,
            autonomy_level,
            {"risk_tier": risk_tier, "reason": reason},
        )
