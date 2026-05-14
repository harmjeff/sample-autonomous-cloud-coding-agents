"""
AutonomyGraduationEngine — Layer 1: Trust & Quality

Reads TrustEventStore signal counts for a (agent_id, task_type) pair and
computes the earned autonomy level. Replaces hardcoded autonomy thresholds
with a dynamic, evidence-based lookup.

Rules:
  - Ratchet upward only: graduation can promote, never auto-demote.
  - Human approval required for any graduation above current level.
  - Any CRITICAL event (scope violation) blocks all graduation permanently
    for that (agent_id, task_type) pair until a human clears it.
  - Autonomy levels: restricted -> supervised -> autonomous

Scoring:
  positive signal  -> +1 point
  neutral signal   ->  0 points
  negative signal  -> -2 points
  critical event   -> blocks graduation (score irrelevant)

Graduation thresholds (net score required to reach level):
  supervised:  >= 10 net points, zero critical events
  autonomous:  >= 30 net points, zero critical events, human approval

Demotion: never automatic. Human must call demote() explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trust.store import TrustEventStore

logger = logging.getLogger(__name__)

# Net score required to reach each level
_THRESHOLDS = {
    "supervised": 10,
    "autonomous": 30,
}

# Signal point values
_SIGNAL_POINTS = {
    "positive": 1,
    "neutral": 0,
    "negative": -2,
    "critical": -10,  # also hard-blocks graduation
}

_LEVEL_ORDER = ["restricted", "supervised", "autonomous"]


@dataclass
class GraduationStatus:
    agent_id: str
    task_type: str
    current_level: str
    net_score: int
    positive: int
    neutral: int
    negative: int
    critical: int
    blocked: bool  # True if any critical events exist
    eligible_for: str | None  # next level agent could reach (None if blocked/at max)
    pending_human_approval: bool = False  # True when score qualifies but awaiting human sign-off


@dataclass
class GraduationRecord:
    """Persistent record of earned autonomy level for (agent_id, task_type)."""

    agent_id: str
    task_type: str
    current_level: str = "supervised"
    pending_approval_for: str | None = None  # level awaiting human approval


class AutonomyGraduationEngine:
    """
    Dynamic autonomy level computation from TrustEvent history.

    Injected into AdmissionPolicy to replace hardcoded autonomy thresholds.
    Falls back gracefully when TrustEventStore is unavailable.
    """

    def __init__(self, store: TrustEventStore) -> None:
        self._store = store
        # In-memory registry: (agent_id, task_type) -> GraduationRecord
        # Phase 4: persist to dedicated store
        self._records: dict[tuple[str, str], GraduationRecord] = {}

    # ------------------------------------------------------------------
    # Primary interface: AdmissionPolicy calls this
    # ------------------------------------------------------------------

    def get_autonomy_level(self, agent_id: str, task_type: str) -> str:
        """
        Return the earned autonomy level for (agent_id, task_type).
        Used by AdmissionPolicy instead of the hardcoded threshold table.
        Falls back to "supervised" on any error.
        """
        try:
            record = self._get_or_create_record(agent_id, task_type)
            # Recompute from current signal counts (status read, record not mutated)
            self.compute_status(agent_id, task_type)
            return record.current_level
        except Exception as exc:
            logger.warning(
                "graduation: get_autonomy_level failed for %s/%s: %s", agent_id, task_type, exc
            )
            return "supervised"

    # ------------------------------------------------------------------
    # Status computation
    # ------------------------------------------------------------------

    def compute_status(self, agent_id: str, task_type: str) -> GraduationStatus:
        """
        Compute the full graduation status from current TrustEvent history.
        Does NOT mutate the record — call evaluate_and_flag() for that.
        """
        counts = self._store.count_by_signal(agent_id, task_type=task_type)
        positive = counts.get("positive", 0)
        neutral = counts.get("neutral", 0)
        negative = counts.get("negative", 0)
        critical = counts.get("critical", 0)

        net_score = (
            positive * _SIGNAL_POINTS["positive"]
            + neutral * _SIGNAL_POINTS["neutral"]
            + negative * _SIGNAL_POINTS["negative"]
            + critical * _SIGNAL_POINTS["critical"]
        )
        blocked = critical > 0
        record = self._get_or_create_record(agent_id, task_type)
        current_level = record.current_level
        current_idx = _LEVEL_ORDER.index(current_level) if current_level in _LEVEL_ORDER else 1

        eligible_for: str | None = None
        if not blocked:
            # Check if agent qualifies for next level
            for level in _LEVEL_ORDER[current_idx + 1 :]:
                threshold = _THRESHOLDS.get(level)
                if threshold is not None and net_score >= threshold:
                    eligible_for = level
                    break

        pending = record.pending_approval_for is not None

        return GraduationStatus(
            agent_id=agent_id,
            task_type=task_type,
            current_level=current_level,
            net_score=net_score,
            positive=positive,
            neutral=neutral,
            negative=negative,
            critical=critical,
            blocked=blocked,
            eligible_for=eligible_for,
            pending_human_approval=pending,
        )

    # ------------------------------------------------------------------
    # Graduation lifecycle
    # ------------------------------------------------------------------

    def evaluate_and_flag(self, agent_id: str, task_type: str) -> GraduationStatus:
        """
        Compute status and, if eligible for graduation, flag for human approval.
        Called after each task completion by the trust pipeline.
        Returns the current status (human must call approve() to actually graduate).
        """
        status = self.compute_status(agent_id, task_type)
        record = self._get_or_create_record(agent_id, task_type)

        if status.eligible_for and not status.pending_human_approval:
            record.pending_approval_for = status.eligible_for
            logger.info(
                "graduation: agent=%s task_type=%s eligible for %s (score=%d) "
                "— awaiting human approval",
                agent_id,
                task_type,
                status.eligible_for,
                status.net_score,
            )
            status.pending_human_approval = True

        return status

    def approve(
        self, agent_id: str, task_type: str, approved_by: str = "human"
    ) -> GraduationRecord:
        """
        Human approves a pending graduation. Promotes the agent to the pending level.
        Ratchet: only moves forward, never backward.
        """
        record = self._get_or_create_record(agent_id, task_type)
        if not record.pending_approval_for:
            logger.warning(
                "graduation: approve() called but no pending approval for %s/%s",
                agent_id,
                task_type,
            )
            return record

        new_level = record.pending_approval_for
        old_level = record.current_level
        record.current_level = new_level
        record.pending_approval_for = None
        logger.info(
            "graduation: APPROVED %s/%s %s -> %s (approved_by=%s)",
            agent_id,
            task_type,
            old_level,
            new_level,
            approved_by,
        )
        return record

    def reject_approval(self, agent_id: str, task_type: str) -> GraduationRecord:
        """Human rejects a pending graduation. Clears the pending flag; level unchanged."""
        record = self._get_or_create_record(agent_id, task_type)
        record.pending_approval_for = None
        logger.info("graduation: REJECTED pending approval for %s/%s", agent_id, task_type)
        return record

    def demote(
        self,
        agent_id: str,
        task_type: str,
        to_level: str,
        reason: str = "",
    ) -> GraduationRecord:
        """
        Human-initiated demotion. The engine never calls this automatically.
        """
        record = self._get_or_create_record(agent_id, task_type)
        old_level = record.current_level
        record.current_level = to_level
        record.pending_approval_for = None
        logger.info(
            "graduation: DEMOTED %s/%s %s -> %s reason=%s",
            agent_id,
            task_type,
            old_level,
            to_level,
            reason,
        )
        return record

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_record(self, agent_id: str, task_type: str) -> GraduationRecord:
        key = (agent_id, task_type)
        if key not in self._records:
            self._records[key] = GraduationRecord(
                agent_id=agent_id,
                task_type=task_type,
                current_level="supervised",
            )
        return self._records[key]
