"""
Concrete tool implementations for the BlueprintBuilderAgent blueprint.

These callables are invoked by the BlueprintTracker local_tools hook when
the Claude Agent SDK requests a tool call that matches one of the names
declared in the meta/blueprint-builder-v1 blueprint.

Wiring (Phase C2 — see pipeline.py):
  When task_type == 'generate_blueprint', pipeline.py creates a
  BlueprintBuilderAgent instance and builds a ``local_tool_handlers`` dict
  mapping tool names to the functions in this module.  The BlueprintTracker
  intercepts tool calls and delegates to the appropriate handler, returning a
  synthetic tool result to the SDK without the SDK needing to execute the tool
  itself.

  Phase C2 note: The interception mechanism (pre-tool hook or synthetic
  ToolResult injection) is not yet wired in pipeline.py. Until it is, these
  functions are importable and unit-testable in isolation.

Tools implemented:
  - search_blueprint_index   — keyword search over PRODUCTION blueprints
  - search_capability_index  — Phase F stub (returns [] until CapabilityIndex lands)
  - generate_blueprint       — constructs Blueprint YAML from provided fields
  - validate_blueprint       — parses YAML and checks required fields
  - register_blueprint_action — writes to FilesystemRegistryService + promotes to VALIDATED
  - finish                   — signals task completion
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_BP_FIELDS = ["id", "version", "task_types", "system_prompt", "tools", "max_iterations"]


# ---------------------------------------------------------------------------
# search_blueprint_index
# ---------------------------------------------------------------------------


def search_blueprint_index(
    description: str,
    min_confidence: float = 0.6,
    _agent=None,
) -> dict[str, Any]:
    """
    Search existing PRODUCTION blueprints by capability description.

    Delegates to BlueprintBuilderAgent.find_blueprints() which does keyword
    matching against task_types.  Returns a list of matching blueprints with
    blueprint_id, task_types, description, and confidence score.

    If no agent instance is provided (unit test / dry-run), returns an empty
    match list so the blueprint loop can still reason through the process.
    """
    if _agent is not None:
        try:
            matches = _agent.find_blueprints(description, min_confidence=min_confidence)
            return {"matches": matches, "search_complete": True}
        except Exception as exc:
            logger.warning("search_blueprint_index registry lookup failed: %s", exc)

    logger.debug(
        "search_blueprint_index: no agent instance or registry unavailable — "
        "returning empty matches for description=%r",
        description,
    )
    return {"matches": [], "search_complete": True}


# ---------------------------------------------------------------------------
# search_capability_index
# ---------------------------------------------------------------------------


def search_capability_index(
    description: str,
    min_confidence: float = 0.5,
    _agent=None,
) -> dict[str, Any]:
    """
    Find existing tools relevant to this domain.

    Phase F stub: CapabilityIndex (LTM-backed semantic search) is not yet
    implemented. Returns an empty match list so the BlueprintBuilderAgent
    falls through to enumerating tools manually on every call.

    When Phase F lands, replace this stub with:
        from registry.capability_index import CapabilityIndex
        matches = CapabilityIndex(...).search(description, min_confidence=min_confidence)
    """
    # Phase F stub: no CapabilityIndex yet
    logger.debug(
        "search_capability_index: CapabilityIndex not available (Phase F pending) — "
        "returning empty matches for description=%r",
        description,
    )
    return {"matches": [], "search_complete": True}


# ---------------------------------------------------------------------------
# generate_blueprint
# ---------------------------------------------------------------------------


def generate_blueprint(
    task_type: str,
    task_description: str = "",
    system_prompt: str = "",
    phases: list | None = None,
    hitl_conditions: list | None = None,
    parameters: dict | None = None,
    required_ltm_capabilities: list | None = None,
    tools: list | None = None,
    max_iterations: int = 20,
    max_tokens: int = 50000,
    _agent=None,
    **_: Any,
) -> dict[str, Any]:
    """
    Construct a Blueprint YAML specification from provided fields.

    Assembles a complete blueprint spec dict and serialises it to YAML.
    The blueprint_id is derived from the task_type (slug form).
    Returns the blueprint_id and the YAML string so the caller can pass
    them directly to validate_blueprint and register_blueprint_action.
    """
    # Derive blueprint_id from task_type (normalise to slug)
    blueprint_id = task_type.lower().replace(" ", "-").replace("_", "-")
    # Ensure a leading namespace if not present
    if "/" not in blueprint_id:
        blueprint_id = f"meta/{blueprint_id}-v1"

    spec: dict[str, Any] = {
        "id": blueprint_id,
        "version": "1",
        "task_types": [task_type],
        "system_prompt": system_prompt or (
            f"You are an autonomous agent that handles {task_description or task_type} tasks."
        ),
        "tools": tools or [],
        "phases": phases or [],
        "hitl_conditions": hitl_conditions or [],
        "output_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        },
        "parameters": parameters or {},
        "required_ltm_capabilities": required_ltm_capabilities or ["SEMANTIC_SEARCH", "WRITE"],
        "quality_checkpoints": [],
        "state_schema": {},
        "max_iterations": max_iterations,
        "max_tokens": max_tokens,
    }
    blueprint_yaml = yaml.dump(spec, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return {
        "blueprint_id": blueprint_id,
        "blueprint_yaml": blueprint_yaml,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# validate_blueprint
# ---------------------------------------------------------------------------


def validate_blueprint(
    blueprint_id: str,
    blueprint_yaml: str,
    _agent=None,
) -> dict[str, Any]:
    """
    Parse blueprint YAML and validate required fields against the Blueprint schema.

    Checks that all fields required by FilesystemRegistryService._validate_blueprint
    are present and non-empty: id, version, task_types, system_prompt, tools,
    max_iterations.  Also verifies that at least one tool is declared.

    Returns a dict with ``valid`` (bool) and ``errors`` (list[str]).
    """
    try:
        data = yaml.safe_load(blueprint_yaml)
        if not isinstance(data, dict):
            return {
                "valid": False,
                "errors": ["Blueprint YAML must be a mapping"],
                "blueprint_id": blueprint_id,
            }
        missing = [f for f in _REQUIRED_BP_FIELDS if not data.get(f)]
        errors: list[str] = [f"Missing required field: {f}" for f in missing]

        tool_names = [t.get("name", "") for t in data.get("tools", []) if isinstance(t, dict)]
        if not tool_names:
            errors.append("Blueprint must declare at least one tool")

        if errors:
            return {"valid": False, "errors": errors, "blueprint_id": blueprint_id}

        return {"valid": True, "errors": [], "blueprint_id": blueprint_id, "tool_names": tool_names}
    except yaml.YAMLError as exc:
        return {
            "valid": False,
            "errors": [f"YAML parse error: {exc}"],
            "blueprint_id": blueprint_id,
        }


# ---------------------------------------------------------------------------
# register_blueprint_action
# ---------------------------------------------------------------------------


def register_blueprint_action(
    blueprint_id: str,
    blueprint_yaml: str,
    promoted_by: str = "blueprint_builder_agent",
    _agent=None,
) -> dict[str, Any]:
    """
    Write the blueprint YAML to FilesystemRegistryService and promote to VALIDATED.

    Delegates to BlueprintBuilderAgent.register_blueprint_entry() which:
      1. Constructs a Blueprint dataclass from the parsed YAML
      2. Registers it via FilesystemRegistryService.register_blueprint()
      3. Promotes it to VALIDATED status

    If no agent instance is provided, returns a stub success result so the
    blueprint loop can complete in unit tests / dry-run mode.
    """
    if _agent is not None:
        try:
            return _agent.register_blueprint_entry(
                blueprint_id=blueprint_id,
                blueprint_yaml=blueprint_yaml,
                promoted_by=promoted_by,
            )
        except Exception as exc:
            logger.error("register_blueprint_action failed: %s", exc)
            return {
                "registered": False,
                "blueprint_id": blueprint_id,
                "error": str(exc),
            }

    # Stub path (no agent instance — unit test or dry-run)
    logger.warning(
        "register_blueprint_action called without agent instance; returning stub result "
        "for blueprint_id=%r. Wire _agent in pipeline.py for live registration.",
        blueprint_id,
    )
    return {
        "registered": True,
        "blueprint_id": blueprint_id,
        "status": "validated",
        "stub": True,
    }


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------


def finish(
    blueprint_id: str,
    status: str = "success",
    summary: str = "",
    _agent=None,
) -> dict[str, Any]:
    """
    Signal task completion to the BlueprintTracker.

    Called by the agent when the blueprint generation task is complete.
    Returns a structured completion record that the BlueprintTracker's
    between_turns hook can pick up to advance phase state.
    """
    logger.info(
        "generate_blueprint task finished: blueprint_id=%r status=%r summary=%r",
        blueprint_id,
        status,
        summary[:200] if summary else "",
    )
    return {
        "task_complete": True,
        "blueprint_id": blueprint_id,
        "status": status,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Local tool handler registry
# ---------------------------------------------------------------------------


def build_local_tool_handlers(agent=None) -> dict[str, Any]:
    """
    Build the local_tool_handlers dict for BlueprintTracker.

    Maps tool names (as declared in meta/blueprint-builder-v1.yaml) to partial
    callables with ``_agent`` already bound.

    Usage in pipeline.py (Phase C2 wiring):

        from tools.blueprint_builder_tools import build_local_tool_handlers
        from agents.blueprint_builder.agent import BlueprintBuilderAgent

        if config.task_type == "generate_blueprint":
            bb_agent = BlueprintBuilderAgent()
            local_tools = build_local_tool_handlers(agent=bb_agent)
            if bp_tracker is not None:
                bp_tracker.local_tool_handlers = local_tools
    """
    import functools

    def _bind(fn, **kwargs):
        return functools.partial(fn, **kwargs)

    return {
        "search_blueprint_index": _bind(search_blueprint_index, _agent=agent),
        "search_capability_index": _bind(search_capability_index, _agent=agent),
        "generate_blueprint": _bind(generate_blueprint, _agent=agent),
        "validate_blueprint": _bind(validate_blueprint, _agent=agent),
        "register_blueprint_action": _bind(register_blueprint_action, _agent=agent),
        "finish": _bind(finish, _agent=agent),
    }
