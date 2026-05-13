"""
PatternEvaluator — evaluates named HITL detection patterns against agent state.

Called after each tool execution in GenericAgent. Returns FiredCondition objects
for any conditions that fire. Tracks already-fired conditions to prevent duplicate
HITL for the same conflict.

Supported named patterns:
  direction_conflict          — same biomarker, contradictory directions
  low_corpus_quality          — paper count or avg score below threshold
  missing_required_entity     — target entity not found post-extraction
  cross_domain_contradiction  — opposite effects across different domains
  source_credibility_gap      — no high-credibility sources in web research

LLM-driven conditions (no pattern: field) are not evaluated here — the
LLM calls the flag_conflict tool directly when it detects a semantic conflict.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FiredCondition:
    trigger: str
    question: str
    options: list[dict[str, str]]
    context: dict[str, Any] = field(default_factory=dict)


class PatternEvaluator:
    def __init__(self) -> None:
        self._fired: set[str] = set()  # (trigger, context_hash) pairs

    def reset(self) -> None:
        """Clear fired history. Call at task start."""
        self._fired.clear()

    def evaluate(
        self,
        conditions: list[dict[str, Any]],
        state: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        current_phase: str | None = None,
        completed_phases: set[str] | None = None,
    ) -> list[FiredCondition]:
        """
        Evaluate all conditions against current state.
        Returns only newly-fired conditions.
        """
        params = parameters or {}
        completed = completed_phases or set()
        fired = []

        for cond in conditions:
            pattern = cond.get("pattern")
            if not pattern:
                continue  # LLM-driven — skip

            # Phase gating: only evaluate after specified phase is complete
            check_after = cond.get("check_after_phase")
            if check_after and check_after not in completed:
                continue

            result = self._evaluate_pattern(pattern, cond, state, params)
            if result is None:
                continue

            # Deduplication
            key = self._condition_key(cond["trigger"], result)
            if key in self._fired:
                continue
            self._fired.add(key)

            question = self._render_template(cond.get("question_template", ""), result)
            options = cond.get("options", [])
            fired.append(
                FiredCondition(
                    trigger=cond["trigger"],
                    question=question,
                    options=options,
                    context=result,
                )
            )

        return fired

    # -----------------------------------------------------------------------
    # Pattern dispatchers
    # -----------------------------------------------------------------------

    def _evaluate_pattern(
        self,
        pattern: str,
        cond: dict,
        state: dict,
        params: dict,
    ) -> dict[str, Any] | None:
        """Returns context dict if pattern fires, None otherwise."""
        handlers = {
            "direction_conflict": self._direction_conflict,
            "low_corpus_quality": self._low_corpus_quality,
            "missing_required_entity": self._missing_required_entity,
            "cross_domain_contradiction": self._cross_domain_contradiction,
            "source_credibility_gap": self._source_credibility_gap,
            "threshold_exceeded": self._threshold_exceeded,
        }
        handler = handlers.get(pattern)
        if not handler:
            return None
        return handler(cond, state, params)

    # -----------------------------------------------------------------------
    # Pattern 1: direction_conflict
    # -----------------------------------------------------------------------

    def _direction_conflict(self, cond: dict, state: dict, params: dict) -> dict | None:
        collection = state.get(cond.get("on", "biomarker_outcomes"), [])
        if not collection:
            return None

        group_by = cond.get("group_by", ["biomarker", "domain"])
        conflict_field = cond.get("conflict_field", "direction")
        conflicting_pairs = [
            tuple(p)
            for p in cond.get(
                "conflicting_pairs",
                [
                    ["increased", "decreased"],
                    ["increased", "no_effect"],
                    ["decreased", "no_effect"],
                ],
            )
        ]
        cross_domain = cond.get("cross_domain", False)

        # Group entries
        groups: dict[tuple, list[dict]] = {}
        for entry in collection:
            key = tuple(entry.get(k, "") for k in group_by)
            groups.setdefault(key, []).append(entry)

        for group_key, entries in groups.items():
            directions = [(e.get(conflict_field, ""), e) for e in entries]
            unique_dirs = {d for d, _ in directions}
            for pair in conflicting_pairs:
                if pair[0] in unique_dirs and pair[1] in unique_dirs:
                    e_a = next(e for d, e in directions if d == pair[0])
                    e_b = next(e for d, e in directions if d == pair[1])
                    ctx = dict(zip(group_by, group_key))
                    ctx.update(
                        {
                            "direction_a": pair[0],
                            "pmid_a": e_a.get("pmid", ""),
                            "direction_b": pair[1],
                            "pmid_b": e_b.get("pmid", ""),
                            "conflict_field": conflict_field,
                        }
                    )
                    return ctx

        return None

    # -----------------------------------------------------------------------
    # Pattern 2: low_corpus_quality
    # -----------------------------------------------------------------------

    def _low_corpus_quality(self, cond: dict, state: dict, params: dict) -> dict | None:
        collection = state.get(cond.get("on", "filtered_papers"), [])
        min_count = int(self._resolve(cond.get("min_count", 3), params))
        min_avg = float(self._resolve(cond.get("min_avg_score", 0.5), params))

        count = len(collection)
        avg_score = sum(p.get("score", 0) for p in collection) / count if count > 0 else 0.0

        if count < min_count or avg_score < min_avg:
            return {
                "count": count,
                "avg_score": round(avg_score, 2),
                "min_count": min_count,
                "min_avg_score": min_avg,
            }
        return None

    # -----------------------------------------------------------------------
    # Pattern 3: missing_required_entity
    # -----------------------------------------------------------------------

    def _missing_required_entity(self, cond: dict, state: dict, params: dict) -> dict | None:
        collection = state.get(cond.get("on", "extracted_biomarkers"), [])
        required = [self._resolve(e, params) for e in cond.get("required_entities", [])]
        if not required:
            return None

        found = {b.get("name", "").lower() for b in collection}
        for entity in required:
            if entity.lower() not in found:
                return {"entity": entity, "found_count": len(found)}
        return None

    # -----------------------------------------------------------------------
    # Pattern 4: cross_domain_contradiction
    # -----------------------------------------------------------------------

    def _cross_domain_contradiction(self, cond: dict, state: dict, params: dict) -> dict | None:
        collection = state.get(cond.get("on", "biomarker_outcomes"), [])
        if not collection:
            return None

        conflict_field = cond.get("conflict_field", "direction")
        conflicting_pairs = [
            tuple(p) for p in cond.get("conflicting_pairs", [["increased", "decreased"]])
        ]
        min_domains = int(cond.get("min_domains", 2))

        # Group by biomarker only
        by_biomarker: dict[str, list[dict]] = {}
        for entry in collection:
            bm = entry.get("biomarker", "")
            by_biomarker.setdefault(bm, []).append(entry)

        for biomarker, entries in by_biomarker.items():
            domains = {e.get("domain", "") for e in entries}
            if len(domains) < min_domains:
                continue
            # Check for contradictory directions across different domains
            by_domain = {}
            for e in entries:
                by_domain.setdefault(e.get("domain", ""), []).append(e)
            domain_list = list(by_domain.keys())
            for i, domain_a in enumerate(domain_list):
                for domain_b in domain_list[i + 1 :]:
                    dirs_a = {e.get(conflict_field) for e in by_domain[domain_a]}
                    dirs_b = {e.get(conflict_field) for e in by_domain[domain_b]}
                    for pair in conflicting_pairs:
                        if pair[0] in dirs_a and pair[1] in dirs_b:
                            e_a = next(
                                e for e in by_domain[domain_a] if e.get(conflict_field) == pair[0]
                            )
                            e_b = next(
                                e for e in by_domain[domain_b] if e.get(conflict_field) == pair[1]
                            )
                            return {
                                "biomarker": biomarker,
                                "domain_a": domain_a,
                                "direction_a": pair[0],
                                "pmid_a": e_a.get("pmid", ""),
                                "domain_b": domain_b,
                                "direction_b": pair[1],
                                "pmid_b": e_b.get("pmid", ""),
                            }
        return None

    # -----------------------------------------------------------------------
    # Pattern 5: source_credibility_gap
    # -----------------------------------------------------------------------

    def _source_credibility_gap(self, cond: dict, state: dict, params: dict) -> dict | None:
        sources = state.get(cond.get("on", "sources"), [])
        min_sources = int(self._resolve(cond.get("min_sources", 3), params))
        if len(sources) < min_sources:
            return None

        check = cond.get("credibility_check", {})
        prefer_patterns = check.get("prefer_domains", [])
        flag_if_none = check.get("flag_if_none_preferred", True)

        if not flag_if_none or not prefer_patterns:
            return None

        def matches(url: str) -> bool:
            for pat in prefer_patterns:
                pat_re = pat.replace(".", r"\.").replace("*", r".*")
                if re.search(pat_re, url):
                    return True
            return False

        preferred = [s for s in sources if matches(s.get("url", ""))]
        if not preferred:
            return {"count": len(sources), "preferred_count": 0}
        return None

    # -----------------------------------------------------------------------
    # Pattern 6: threshold_exceeded
    # -----------------------------------------------------------------------

    def _threshold_exceeded(self, cond: dict, state: dict, params: dict) -> dict | None:
        collection = state.get(cond.get("on", ""), [])
        filt = cond.get("filter")
        if filt:
            collection = [e for e in collection if all(e.get(k) == v for k, v in filt.items())]

        threshold = int(self._resolve(cond.get("threshold", 10), params))
        count = len(collection) if isinstance(collection, list) else int(collection)

        if count > threshold:
            return {
                "count": count,
                "threshold": threshold,
                "last_error": state.get("last_error", ""),
            }
        return None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve(self, value: Any, params: dict) -> Any:
        """Resolve {placeholder} strings against parameters."""
        if isinstance(value, str) and "{" in value:
            for k, v in params.items():
                value = value.replace(f"{{{k}}}", str(v))
        return value

    def _render_template(self, template: str, context: dict) -> str:
        try:
            return template.format(**context)
        except (KeyError, ValueError):
            return template

    @staticmethod
    def _condition_key(trigger: str, context: dict) -> str:
        ctx_str = json.dumps({k: str(v) for k, v in sorted(context.items())}, sort_keys=True)
        digest = hashlib.md5(ctx_str.encode()).hexdigest()[:8]
        return f"{trigger}:{digest}"
