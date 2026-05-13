from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class LTMCapability(StrEnum):
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    ENTITY_LOOKUP = "ENTITY_LOOKUP"
    RELATIONSHIP_GRAPH = "RELATIONSHIP_GRAPH"
    TEMPORAL_QUERY = "TEMPORAL_QUERY"
    WRITE = "WRITE"
    IMPORTANCE_SCORING = "IMPORTANCE_SCORING"
    DECAY = "DECAY"
    CONSOLIDATION = "CONSOLIDATION"
    ATOMIC_EXTRACTION = "ATOMIC_EXTRACTION"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"


# ---------------------------------------------------------------------------
# Memory category — drives decay_rate assignment at write time
# ---------------------------------------------------------------------------


class MemoryCategory(StrEnum):
    IMMUTABLE = "immutable"  # facts that never expire (e.g. domain constants)
    STABLE = "stable"  # preferences, tool registrations — ~375 days
    ACTIVE = "active"  # research findings, task outputs — ~30 days
    TRANSIENT = "transient"  # working context, scratch notes — ~1.5 days


# decay_rate values correspond to exponential half-life:
#   importance × exp(-decay_rate × elapsed_days)
_CATEGORY_DECAY_RATE: dict[MemoryCategory, float] = {
    MemoryCategory.IMMUTABLE: 0.0,
    MemoryCategory.STABLE: 0.008,
    MemoryCategory.ACTIVE: 0.1,
    MemoryCategory.TRANSIENT: 2.0,
}


def decay_rate_for(category: MemoryCategory) -> float:
    return _CATEGORY_DECAY_RATE[category]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MemoryMetadata:
    source: str
    agent_id: str
    task_id: str
    importance_hint: float = 0.5  # 0.0–1.0, updated dynamically
    entity_type: str | None = None
    entity_id: str | None = None
    tags: list[str] = field(default_factory=list)
    # Lifecycle fields
    category: str = MemoryCategory.ACTIVE.value
    decay_rate: float = 0.1  # derived from category at write time
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed: str | None = None


@dataclass
class MemoryResult:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None
    # Convenience accessors populated from metadata when available
    importance: float | None = None
    access_count: int | None = None
    last_accessed: str | None = None
    created_at: str | None = None
    # Set by composite scoring post-processing
    composite_score: float | None = None


@dataclass
class RecallConfig:
    """
    Configurable composite scoring weights for LTM read().

    Final score = (similarity × w_sim) + (recency × w_rec) + (importance × w_imp)

    recency = exp(-elapsed_days / recency_half_life_days)
    Default half-life 30 days — a memory from 30 days ago scores ~0.5 on recency.
    """

    w_sim: float = 0.5
    w_rec: float = 0.3
    w_imp: float = 0.2
    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        total = self.w_sim + self.w_rec + self.w_imp
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RecallConfig weights must sum to 1.0, got {total}")


@dataclass
class NegotiationResult:
    supported: set[LTMCapability]
    missing_required: set[LTMCapability]
    missing_optional: set[LTMCapability]

    @property
    def ok(self) -> bool:
        return len(self.missing_required) == 0


@dataclass
class ContradictionResult:
    contradiction_found: bool
    resolution: str  # REPLACE | MERGE | KEEP_BOTH | REJECT_NEW
    conflicting_memory_id: str | None = None
    conflicting_content: str | None = None
    explanation: str = ""


@dataclass
class ConsolidationResult:
    user_id: str
    clusters_found: int
    merged_count: int
    pruned_count: int


def apply_composite_scoring(
    results: list["MemoryResult"],
    config: "RecallConfig",
) -> list["MemoryResult"]:
    """
    Re-rank results using composite score = sim×w + recency×w + importance×w.
    Mutates composite_score on each result and returns sorted list.
    """
    import math

    now = datetime.now(UTC)

    for r in results:
        sim = r.score if r.score is not None else 0.5
        imp = r.importance if r.importance is not None else 0.5

        recency = 0.5  # neutral when created_at unknown
        if r.created_at:
            try:
                created = datetime.fromisoformat(r.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                elapsed_days = (now - created).total_seconds() / 86400.0
                recency = math.exp(-elapsed_days / config.recency_half_life_days)
            except (ValueError, TypeError):
                pass

        r.composite_score = round(
            config.w_sim * sim + config.w_rec * recency + config.w_imp * imp, 4
        )

    results.sort(key=lambda r: r.composite_score or 0.0, reverse=True)
    return results


class CapabilityMismatchError(Exception):
    def __init__(self, missing: set[LTMCapability]) -> None:
        self.missing = missing
        super().__init__(
            f"Required LTM capabilities not supported: {', '.join(c.value for c in missing)}"
        )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class LTMInterface(ABC):
    @abstractmethod
    def capabilities(self) -> set[LTMCapability]:
        """Return the set of capabilities this backend supports."""
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        query: str,
        user_id: str,
        mode: LTMCapability = LTMCapability.SEMANTIC_SEARCH,
        limit: int = 10,
        recall_config: "RecallConfig | None" = None,
    ) -> list[MemoryResult]:
        """Query memory. Mode determines the retrieval strategy used.
        If recall_config provided, results are re-ranked by composite score."""
        raise NotImplementedError

    @abstractmethod
    def write(self, content: str, user_id: str, metadata: MemoryMetadata) -> str:
        """Persist a memory. Returns the assigned memory ID."""
        raise NotImplementedError

    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID. Returns True if deleted, False if not found."""
        raise NotImplementedError

    def extract_and_write(
        self,
        content: str,
        user_id: str,
        metadata: "MemoryMetadata",
    ) -> list[str]:
        """
        Decompose a content blob into atomic self-contained facts, then write
        each fact as a separate memory entry.
        Returns list of memory IDs created.
        Default: falls back to a single write() call for backends without
        atomic extraction support.
        """
        mem_id = self.write(content, user_id, metadata)
        return [mem_id] if mem_id else []

    def decay(self, user_id: str, threshold: float = 0.05) -> int:
        """
        Prune memories whose effective importance has dropped below threshold.
        Formula: importance × exp(-decay_rate × elapsed_days) × access_bonus
        Returns the number of memories pruned. Default no-op for backends
        that don't declare DECAY capability.
        """
        return 0

    def reinforce(self, memory_id: str, user_id: str) -> None:
        """
        Bump importance on successful retrieval (spaced-repetition effect).
        Formula: v_new = v + Δv × (1 - v) × exp(-n / N)
        Default no-op for backends that don't declare IMPORTANCE_SCORING.
        """

    def detect_contradictions(
        self,
        content: str,
        user_id: str,
        source_trust: float = 0.5,
        similarity_threshold: float = 0.85,
    ) -> "ContradictionResult":
        """
        Search for contradictions with existing memories before writing.
        Returns a ContradictionResult with resolution plan.
        Default no-op — returns no contradiction found.
        """
        return ContradictionResult(contradiction_found=False, resolution="KEEP_BOTH")

    def consolidate(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
    ) -> ConsolidationResult:
        """
        Merge near-duplicate memories and resolve contradictions by recency.
        Default no-op for backends that don't declare CONSOLIDATION capability.
        """
        return ConsolidationResult(
            user_id=user_id,
            clusters_found=0,
            merged_count=0,
            pruned_count=0,
        )

    def negotiate(
        self,
        required: list[LTMCapability],
        optional: list[LTMCapability] | None = None,
    ) -> NegotiationResult:
        """
        Validate that this backend meets the caller's capability requirements.
        Raises CapabilityMismatchError if any required capability is missing.
        """
        supported = self.capabilities()
        missing_required = set(required) - supported
        missing_optional = set(optional or []) - supported

        result = NegotiationResult(
            supported=supported,
            missing_required=missing_required,
            missing_optional=missing_optional,
        )

        if missing_required:
            raise CapabilityMismatchError(missing_required)

        return result
