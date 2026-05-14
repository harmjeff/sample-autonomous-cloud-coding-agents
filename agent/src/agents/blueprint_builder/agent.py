"""
Blueprint Builder Agent — generates blueprint YAML specs from natural language.

Ported from AKW src/agents/blueprint_builder/agent.py.

In ABCA, the BlueprintBuilderAgent is wired via the meta/blueprint-builder-v1
blueprint. The Claude Agent SDK handles the execution loop; this module
provides the concrete tool implementations invoked by the blueprint-driven
agent via local_tool_handlers (see pipeline.py Phase C2 wiring).

Key differences from AKW:
  - AKW used GenericAgent directly (same-process execution loop).
    ABCA's loop is the Claude Agent SDK; BlueprintTracker observes state.
  - AKW tool implementations were inline Tool subclasses inside _get_blueprint_tools.
    ABCA exposes them as Python callables in tools/blueprint_builder_tools.py.
    The BlueprintTracker local_tools hook intercepts tool calls and routes them
    to the appropriate callable (see pipeline.py Phase C2 wiring comment).

Network isolation: the blueprint_builder task does NOT need sandbox access.
All registry I/O is via FilesystemRegistryService (filesystem reads/writes).
"""

from __future__ import annotations

import logging

from registry.filesystem import FilesystemRegistryService

logger = logging.getLogger(__name__)


class BlueprintBuilderAgent:
    """
    Facade for the Blueprint Builder Agent in ABCA.

    In ABCA's architecture the Claude Agent SDK drives the execution loop
    via the meta/blueprint-builder-v1 blueprint system_prompt and tool specs.
    This class:
      1. Exposes a registry for use by blueprint_builder_tools.py
      2. Provides find_blueprints() used by search_blueprint_index
      3. Provides register_blueprint_entry() used by register_blueprint_action

    Instantiated once per generate_blueprint task in pipeline.py (Phase C2 wiring).
    """

    def __init__(self) -> None:
        self._registry = FilesystemRegistryService()

    # ------------------------------------------------------------------
    # Delegation helpers for blueprint_builder_tools.py
    # ------------------------------------------------------------------

    def find_blueprints(self, description: str, min_confidence: float = 0.6) -> list[dict]:
        """
        Keyword search over production blueprints by task_types / description.

        Returns a list of serialisable dicts with blueprint_id, task_types,
        and confidence so callers do not need to import registry models.
        """
        try:
            from registry.models import PromotionStatus

            blueprints = self._registry.list_blueprints(status=PromotionStatus.PRODUCTION)
            query_words = set(description.lower().split())
            matches = []
            for bp in blueprints:
                bp_words = set(" ".join(bp.task_types).lower().split())
                overlap = query_words & bp_words
                if overlap:
                    conf = len(overlap) / max(len(query_words), 1)
                    if conf >= min_confidence:
                        matches.append({
                            "blueprint_id": bp.id,
                            "task_types": bp.task_types,
                            "description": bp.system_prompt[:120] if bp.system_prompt else "",
                            "confidence": round(conf, 2),
                        })
            return sorted(matches, key=lambda m: m["confidence"], reverse=True)
        except Exception as exc:
            logger.warning("BlueprintBuilderAgent.find_blueprints failed: %s", exc)
            return []

    def register_blueprint_entry(
        self,
        blueprint_id: str,
        blueprint_yaml: str,
        promoted_by: str = "blueprint_builder_agent",
    ) -> dict:
        """
        Write blueprint YAML and register it in FilesystemRegistryService.

        Registers at DRAFT, then immediately promotes to VALIDATED so the
        blueprint is available for inspection without requiring a full
        human-approval cycle (mirrors AKW behaviour). Returns a serialisable
        result dict.
        """
        import yaml

        from registry.models import (
            Blueprint,
            PromotionStatus,
        )

        try:
            data = yaml.safe_load(blueprint_yaml)
            bp = Blueprint(
                id=blueprint_id,
                version=str(data.get("version", "1")),
                task_types=data.get("task_types", []),
                system_prompt=data.get("system_prompt", ""),
                tools=data.get("tools", []),
                phases=data.get("phases", []),
                state_schema=data.get("state_schema", {}),
                hitl_conditions=data.get("hitl_conditions", []),
                output_schema=data.get(
                    "output_schema",
                    {"type": "object", "properties": {"summary": {"type": "string"}}},
                ),
                quality_checkpoints=data.get("quality_checkpoints", []),
                parameters=data.get("parameters", {}),
                required_ltm_capabilities=data.get(
                    "required_ltm_capabilities", ["SEMANTIC_SEARCH", "WRITE"]
                ),
                max_iterations=int(data.get("max_iterations", 20)),
                max_tokens=int(data.get("max_tokens", 50000)),
                status=PromotionStatus.DRAFT,
                created_by="blueprint_builder_agent",
            )
            self._registry.register_blueprint(bp)

            # Promote to VALIDATED — mirrors AKW register_blueprint_action behaviour
            try:
                self._registry.promote_blueprint(
                    blueprint_id, PromotionStatus.VALIDATED, promoted_by
                )
                status = PromotionStatus.VALIDATED
            except Exception as promo_exc:
                logger.warning(
                    "Auto-promotion to VALIDATED failed for %s: %s", blueprint_id, promo_exc
                )
                status = PromotionStatus.DRAFT

            return {
                "registered": True,
                "blueprint_id": blueprint_id,
                "status": str(status),
            }
        except Exception as exc:
            logger.error("register_blueprint_entry failed for %s: %s", blueprint_id, exc)
            return {
                "registered": False,
                "blueprint_id": blueprint_id,
                "error": str(exc),
            }
