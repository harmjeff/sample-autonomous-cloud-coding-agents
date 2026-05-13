import math
from datetime import UTC, datetime

from interfaces.ltm import (
    ConsolidationResult,
    LTMCapability,
    LTMInterface,
    MemoryCategory,
    MemoryMetadata,
    MemoryResult,
    RecallConfig,
    apply_composite_scoring,
    decay_rate_for,
)

_REINFORCE_N = 20.0  # access count at which reinforcement asymptotes


class MockFlatLTM(LTMInterface):
    """
    In-memory flat LTM stub for unit tests.
    Supports SEMANTIC_SEARCH, WRITE, DECAY, IMPORTANCE_SCORING, CONSOLIDATION.
    No entity or graph capabilities (validates capability negotiation failures).
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, MemoryMetadata]] = {}
        self._counter = 0

    def capabilities(self) -> set[LTMCapability]:
        return {
            LTMCapability.SEMANTIC_SEARCH,
            LTMCapability.WRITE,
            LTMCapability.DECAY,
            LTMCapability.IMPORTANCE_SCORING,
            LTMCapability.CONSOLIDATION,
        }

    def read(
        self,
        query: str,
        user_id: str,
        mode: LTMCapability = LTMCapability.SEMANTIC_SEARCH,
        limit: int = 10,
        recall_config: RecallConfig | None = None,
    ) -> list[MemoryResult]:
        results = []
        for mem_id, (content, meta) in self._store.items():
            if query.lower() in content.lower():
                self.reinforce(mem_id, user_id)
                results.append(
                    MemoryResult(
                        id=mem_id,
                        content=content,
                        metadata={},
                        score=meta.importance_hint,  # use importance as proxy score
                        importance=meta.importance_hint,
                        access_count=meta.access_count,
                        last_accessed=meta.last_accessed,
                        created_at=meta.created_at,
                    )
                )
        results = results[:limit]
        if recall_config is not None:
            results = apply_composite_scoring(results, recall_config)
        return results

    def write(self, content: str, user_id: str, metadata: MemoryMetadata) -> str:
        # Derive decay_rate from category
        try:
            cat = MemoryCategory(metadata.category)
        except ValueError:
            cat = MemoryCategory.ACTIVE
        if metadata.decay_rate == 0.1:  # default — override from category
            metadata.decay_rate = decay_rate_for(cat)

        self._counter += 1
        mem_id = f"mock_{self._counter}"
        self._store[mem_id] = (content, metadata)
        return mem_id

    def forget(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    def decay(self, user_id: str, threshold: float = 0.05) -> int:
        """
        Prune memories below effective importance threshold.
        effective = importance × exp(-decay_rate × elapsed_days) × access_bonus
        """
        now = datetime.now(UTC)
        to_prune = []
        for mem_id, (content, meta) in self._store.items():
            if meta.decay_rate == 0.0:
                continue  # immutable
            try:
                created = datetime.fromisoformat(meta.created_at)
            except (ValueError, TypeError):
                continue
            elapsed_days = (now - created).total_seconds() / 86400.0
            access_bonus = 1.0 + math.log1p(meta.access_count) * 0.1
            effective = (
                meta.importance_hint * math.exp(-meta.decay_rate * elapsed_days) * access_bonus
            )
            if effective < threshold:
                to_prune.append(mem_id)
        for mem_id in to_prune:
            del self._store[mem_id]
        return len(to_prune)

    def reinforce(self, memory_id: str, user_id: str) -> None:
        """
        Spaced-repetition reinforcement on retrieval.
        v_new = v + delta_v × (1 - v) × exp(-n / N)
        where delta_v = 0.1, N = 20 (asymptote after ~20 accesses).
        """
        if memory_id not in self._store:
            return
        content, meta = self._store[memory_id]
        n = meta.access_count
        delta_v = 0.1
        boost = delta_v * (1.0 - meta.importance_hint) * math.exp(-n / _REINFORCE_N)
        meta.importance_hint = min(1.0, meta.importance_hint + boost)
        meta.access_count = n + 1
        meta.last_accessed = datetime.now(UTC).isoformat()
        self._store[memory_id] = (content, meta)

    def consolidate(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
    ) -> ConsolidationResult:
        # Mock: deduplicate exact-match content within the same user_id namespace
        seen: dict[str, str] = {}
        to_prune = []
        for mem_id, (content, _meta) in self._store.items():
            key = content.strip().lower()
            if key in seen:
                to_prune.append(mem_id)
            else:
                seen[key] = mem_id
        for mem_id in to_prune:
            del self._store[mem_id]
        return ConsolidationResult(
            user_id=user_id,
            clusters_found=len(to_prune),
            merged_count=len(to_prune),
            pruned_count=0,
        )
