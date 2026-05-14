"""
TrustEventStore — abstract base class for trust event persistence.

Concrete backends: DynamoTrustEventStore (ABCA), TrustEventStore (AKW/LTM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trust.models import TrustEvent, TrustSignalStrength


class TrustEventStore(ABC):
    """Abstract base for trust event stores."""

    @abstractmethod
    def write(self, event: TrustEvent) -> None:
        """Persist a trust event."""

    @abstractmethod
    def count_by_signal(
        self,
        agent_id: str,
        task_type: str | None = None,
    ) -> dict[str, int]:
        """
        Count events by signal strength for an agent.
        Returns {"positive": N, "neutral": N, "negative": N, "critical": N}
        """

    def has_critical_events(
        self,
        agent_id: str,
        task_type: str | None = None,
    ) -> bool:
        """Returns True if agent has any CRITICAL signal events."""
        counts = self.count_by_signal(agent_id, task_type=task_type)
        return counts.get(TrustSignalStrength.CRITICAL.value, 0) > 0

    def read_for_agent(
        self,
        agent_id: str,
        task_type: str | None = None,
        last_n: int = 50,
    ) -> list[dict[str, Any]]:
        """Read trust events for a specific agent (optional, override as needed)."""
        return []
