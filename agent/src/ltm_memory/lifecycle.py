"""
MemoryLifecycleEngine — orchestrates the post-task memory lifecycle.

Lifecycle: inject → distill → trim → consolidate
  - inject:      agents write findings during task execution (existing)
  - distill:     Mem0's own LLM-guided extraction on write (existing)
  - trim (decay): prune memories below importance threshold (new)
  - consolidate:  merge near-duplicates, resolve contradictions (new)

Call run_post_task() at task completion, before the TASK_COMPLETE TrustEvent
fires so that consolidation results can be included in event metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from interfaces.ltm import ConsolidationResult, LTMCapability, LTMInterface

logger = logging.getLogger(__name__)


@dataclass
class LifecycleResult:
    user_id: str
    pruned: int
    consolidation: ConsolidationResult | None


class MemoryLifecycleEngine:
    """
    Runs decay + consolidation after task completion.

    decay_threshold:          memories below this effective importance are pruned
    consolidation_threshold:  cosine similarity above which memories are merged
    run_consolidation:        set False to skip consolidation (e.g. for transient tasks)
    """

    def __init__(
        self,
        decay_threshold: float = 0.05,
        consolidation_threshold: float = 0.85,
        run_consolidation: bool = True,
    ) -> None:
        self._decay_threshold = decay_threshold
        self._consolidation_threshold = consolidation_threshold
        self._run_consolidation = run_consolidation

    def run_post_task(self, ltm: LTMInterface, user_id: str) -> LifecycleResult:
        """
        Run decay → consolidate for the given user_id namespace.
        Safe to call even if the backend doesn't support the capabilities —
        it degrades gracefully.
        """
        caps = ltm.capabilities()

        # Trim: prune expired memories
        pruned = 0
        if LTMCapability.DECAY in caps:
            try:
                pruned = ltm.decay(user_id=user_id, threshold=self._decay_threshold)
                if pruned:
                    logger.info("lifecycle: pruned %d expired memories for %s", pruned, user_id)
            except Exception as exc:
                logger.warning("lifecycle: decay failed for %s: %s", user_id, exc)

        # Consolidate: merge near-duplicates
        consolidation: ConsolidationResult | None = None
        if self._run_consolidation and LTMCapability.CONSOLIDATION in caps:
            try:
                consolidation = ltm.consolidate(
                    user_id=user_id,
                    similarity_threshold=self._consolidation_threshold,
                )
                if consolidation.merged_count:
                    logger.info(
                        "lifecycle: consolidated %d memories (%d clusters) for %s",
                        consolidation.merged_count,
                        consolidation.clusters_found,
                        user_id,
                    )
            except Exception as exc:
                logger.warning("lifecycle: consolidation failed for %s: %s", user_id, exc)

        return LifecycleResult(
            user_id=user_id,
            pruned=pruned,
            consolidation=consolidation,
        )
