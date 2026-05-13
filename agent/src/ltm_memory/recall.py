"""
ConfidentRecall — confidence-aware LTM retrieval with automatic broadening.

After a standard read(), evaluates the mean score of results against a
confidence_threshold. If confidence is low, broadens the search:
  1. Higher result limit (2×)
  2. Two alternative query reformulations
  3. Tracks evidence_gaps (queries that returned zero results)

Returns a ConfidenceResult instead of a raw list, so callers know whether
the answer is well-supported or speculative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from interfaces.ltm import LTMInterface, MemoryResult

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.6
_DEFAULT_LIMIT = 10
_BROAD_MULTIPLIER = 2


@dataclass
class ConfidenceResult:
    results: list[MemoryResult]
    confidence_score: float  # 0.0–1.0 mean of top-k scores
    evidence_gaps: list[str]  # queries that returned zero results
    search_strategy_used: str  # "direct" | "broadened"
    queries_tried: list[str] = field(default_factory=list)


class ConfidentRecall:
    """
    Wraps LTMInterface.read() with self-assessed confidence and automatic
    search broadening when confidence is below threshold.
    """

    def __init__(
        self,
        ltm: LTMInterface,
        confidence_threshold: float = _DEFAULT_THRESHOLD,
        default_limit: int = _DEFAULT_LIMIT,
    ) -> None:
        self._ltm = ltm
        self._threshold = confidence_threshold
        self._limit = default_limit

    def recall(
        self,
        query: str,
        user_id: str,
        limit: int | None = None,
    ) -> ConfidenceResult:
        """
        Retrieve memories with confidence evaluation.

        If initial results fall below confidence_threshold, broadens search
        using a higher limit and alternative query formulations.
        """
        effective_limit = limit or self._limit
        evidence_gaps: list[str] = []
        queries_tried: list[str] = [query]

        # --- Direct pass ---
        results = self._read(query, user_id, effective_limit)
        confidence = _score(results)

        if confidence >= self._threshold:
            return ConfidenceResult(
                results=results,
                confidence_score=confidence,
                evidence_gaps=evidence_gaps,
                search_strategy_used="direct",
                queries_tried=queries_tried,
            )

        logger.debug(
            "recall: low confidence %.2f < %.2f for '%s', broadening",
            confidence,
            self._threshold,
            query,
        )
        if not results:
            evidence_gaps.append(query)

        # --- Broadened pass ---
        broad_limit = effective_limit * _BROAD_MULTIPLIER
        all_results: list[MemoryResult] = list(results)
        seen_ids = {r.id for r in results}

        for alt_query in _alternative_queries(query):
            queries_tried.append(alt_query)
            alt_results = self._read(alt_query, user_id, broad_limit)
            if not alt_results:
                evidence_gaps.append(alt_query)
            for r in alt_results:
                if r.id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.id)

        # Re-score merged set, sort by score descending, cap at broad_limit
        all_results.sort(key=lambda r: r.score or 0.0, reverse=True)
        all_results = all_results[:broad_limit]
        final_confidence = _score(all_results)

        return ConfidenceResult(
            results=all_results,
            confidence_score=final_confidence,
            evidence_gaps=evidence_gaps,
            search_strategy_used="broadened",
            queries_tried=queries_tried,
        )

    def _read(self, query: str, user_id: str, limit: int) -> list[MemoryResult]:
        try:
            return self._ltm.read(query, user_id=user_id, limit=limit)
        except Exception as exc:
            logger.warning("recall: read failed for '%s': %s", query, exc)
            return []


def _score(results: list[MemoryResult]) -> float:
    """Mean of top-k scores. Returns 0.0 when list is empty."""
    if not results:
        return 0.0
    scores = [r.score for r in results if r.score is not None]
    if not scores:
        # No scores from backend — treat non-empty result as moderate confidence
        return 0.5
    return round(sum(scores) / len(scores), 3)


def _alternative_queries(query: str) -> list[str]:
    """Generate 2 alternative query formulations for broadened search."""
    q = query.strip().rstrip("?.")
    return [
        f"background on {q}",
        f"context about {q}",
    ]
