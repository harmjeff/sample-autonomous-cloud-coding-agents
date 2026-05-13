"""ConsolidationReport — structured output from the Deep Consolidation Agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsolidationReport:
    user_id: str
    synthetic_memories_written: int
    memories_deprecated: int
    patterns_identified: list[str] = field(default_factory=list)
    contradictions_found: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "synthetic_memories_written": self.synthetic_memories_written,
            "memories_deprecated": self.memories_deprecated,
            "patterns_identified": self.patterns_identified,
            "contradictions_found": self.contradictions_found,
            "summary": self.summary,
        }
