"""
CapabilityIndex — semantic search over tool capability descriptions via LTM.

Wraps the LTM interface to store and retrieve tool capability descriptions.
Replaces keyword matching in FilesystemRegistryService.find_tools().

Each tool capability is stored as:
  user_id: "capability_index"
  content: "{name}: {capability_description}"
  metadata: entity_type="tool", entity_id=tool_id

Each tool usage record is stored as:
  user_id: "tool_usage"
  content: "Tool {name} {succeeded|failed} for {task_type}: {instructions_summary}"
  metadata: entity_type="tool_usage", entity_id=tool_id, tags=[name, task_type, success/failure]

Search blends semantic similarity (0.6) with context-aware success rate (0.4)
when usage records exist for the given task_type.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime

from interfaces.ltm import LTMCapability, LTMInterface, MemoryCategory, MemoryMetadata
from registry.models import ToolEntry, ToolMatch

logger = logging.getLogger(__name__)

_CI_USER_ID = "capability_index"
_USAGE_USER_ID = "tool_usage"

_SEMANTIC_WEIGHT = 0.6
_SUCCESS_WEIGHT = 0.4


class CapabilityIndex:
    def __init__(self, ltm: LTMInterface) -> None:
        self._ltm = ltm
        self._ltm.negotiate(required=[LTMCapability.SEMANTIC_SEARCH, LTMCapability.WRITE])

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolEntry) -> None:
        """Write or update a tool's capability description in the index."""
        content = f"{tool.name}: {tool.capability_description}"
        metadata = MemoryMetadata(
            source="capability_index",
            agent_id="registry",
            task_id="registration",
            importance_hint=0.9,
            category=MemoryCategory.STABLE.value,
            entity_type="tool",
            entity_id=tool.tool_id,
            tags=["tool", tool.name, tool.implementation_type],
        )
        self._ltm.write(content, user_id=_CI_USER_ID, metadata=metadata)
        logger.debug("CapabilityIndex: registered tool '%s'", tool.name)

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        tool: ToolEntry,
        task_type: str,
        instructions_summary: str,
        success: bool,
        duration_ms: int,
        error_type: str | None = None,
        output_quality: str = "complete",
    ) -> None:
        """
        Write a usage record to LTM after every production tool execution.
        Called by RegistryService.record_tool_execution().
        """
        outcome = "succeeded" if success else "failed"
        error_suffix = f" [{error_type}]" if error_type and not success else ""
        content = (
            f"Tool {tool.name} {outcome} for {task_type}: "
            f"{instructions_summary[:100]}"
            f"{error_suffix} "
            f"(quality={output_quality}, duration={duration_ms}ms)"
        )
        tags = ["tool_usage", tool.name, task_type, outcome]
        if error_type:
            tags.append(error_type)

        metadata = MemoryMetadata(
            source="tool_usage",
            agent_id="registry",
            task_id=f"usage_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            importance_hint=0.6,
            category=MemoryCategory.ACTIVE.value,
            entity_type="tool_usage",
            entity_id=tool.tool_id,
            tags=tags,
        )
        try:
            self._ltm.write(content, user_id=_USAGE_USER_ID, metadata=metadata)
            logger.debug(
                "CapabilityIndex: recorded %s for tool '%s' in '%s'", outcome, tool.name, task_type
            )
        except Exception as e:
            logger.warning("CapabilityIndex: failed to record usage: %s", e)

    def get_success_rate(
        self,
        tool_id: str,
        task_type: str | None = None,
        last_n: int = 20,
    ) -> float:
        """
        Returns success rate (0.0–1.0) for this tool, optionally filtered by task_type.
        Uses recency-weighted average: more recent records count more.
        Returns 0.5 (neutral) when no usage records exist.
        """
        query = f"tool usage {tool_id}"
        if task_type:
            query += f" {task_type}"

        try:
            memories = self._ltm.read(query, user_id=_USAGE_USER_ID, limit=last_n)
        except Exception:
            return 0.5  # neutral when LTM unavailable

        if not memories:
            return 0.5

        # Filter to records for this tool
        relevant = []
        for m in memories:
            meta = m.metadata or {}
            if meta.get("entity_id") == tool_id:
                if task_type is None or task_type in (meta.get("tags") or []):
                    relevant.append(m.content or "")

        if not relevant:
            return 0.5

        # Count success vs failure with recency weighting
        total_weight = 0.0
        success_weight = 0.0
        for i, content in enumerate(relevant):
            # More recent = higher weight (earlier in list = more recent from LTM)
            weight = 1.0 / (i + 1)
            total_weight += weight
            if "succeeded" in content:
                success_weight += weight

        return round(success_weight / total_weight, 3) if total_weight > 0 else 0.5

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        description: str,
        min_confidence: float = 0.6,
        available_tools: list[ToolEntry] | None = None,
        task_type: str | None = None,
    ) -> list[ToolMatch]:
        """
        Semantic search over tool capability descriptions.

        When task_type is provided and usage records exist, blends:
          final_score = (SEMANTIC_WEIGHT × semantic_score) + (SUCCESS_WEIGHT × success_rate)

        Returns ToolMatch list sorted by blended confidence descending.
        """
        try:
            memories = self._ltm.read(description, user_id=_CI_USER_ID)
        except Exception as e:
            logger.warning("CapabilityIndex search failed: %s", e)
            return []

        if not available_tools:
            return []

        by_id = {t.tool_id: t for t in available_tools}
        by_name = {t.name: t for t in available_tools}

        matches = []
        seen_ids = set()

        for i, memory in enumerate(memories):
            meta = memory.metadata or {}
            entity_id = meta.get("entity_id", "")
            tool = by_id.get(entity_id)

            if not tool:
                content = memory.content or ""
                for name, t in by_name.items():
                    if content.startswith(f"{name}:"):
                        tool = t
                        break

            if not tool or tool.tool_id in seen_ids:
                continue

            seen_ids.add(tool.tool_id)

            semantic_score = memory.score if memory.score is not None else max(0.9 - i * 0.1, 0.1)

            # Blend with success rate if task_type provided
            if task_type:
                success_rate = self.get_success_rate(tool.tool_id, task_type=task_type)
                blended = (_SEMANTIC_WEIGHT * semantic_score) + (_SUCCESS_WEIGHT * success_rate)
                reason = (
                    f"Semantic={semantic_score:.2f} x {_SEMANTIC_WEIGHT} + "
                    f"SuccessRate={success_rate:.2f} x {_SUCCESS_WEIGHT} "
                    f"(task_type={task_type})"
                )
            else:
                blended = semantic_score
                reason = f"Semantic match (score={semantic_score:.2f})"

            if blended >= min_confidence:
                matches.append(
                    ToolMatch(
                        tool=tool,
                        confidence=round(min(blended, 1.0), 3),
                        match_reason=reason,
                    )
                )

        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, tool_id: str) -> None:
        """Remove a tool from the index (on deprecation)."""
        with contextlib.suppress(Exception):
            self._ltm.forget(tool_id)
