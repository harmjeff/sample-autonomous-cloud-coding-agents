"""
Concrete tool implementations for the ToolBuilderAgent blueprint.

These callables are invoked by the BlueprintTracker local_tools hook when
the Claude Agent SDK requests a tool call that matches one of the names
declared in the meta/tool-builder-v1 blueprint.

Wiring (Phase C1 — see pipeline.py):
  When task_type == 'generate_tool', pipeline.py creates a ToolBuilderAgent
  instance and builds a ``local_tool_handlers`` dict mapping tool names to
  the functions in this module. The BlueprintTracker intercepts tool calls
  and delegates to the appropriate handler, returning a synthetic tool result
  to the SDK without the SDK needing to execute the tool itself.

  Phase C1 note: The interception mechanism (pre-tool hook or synthetic
  ToolResult injection) is not yet wired in pipeline.py. Until it is, these
  functions are importable and unit-testable in isolation. The SDK will
  attempt to call the tools against its own execution environment; the
  blueprint system_prompt provides enough context for the agent to reason
  through the process even without live tool backends.

Tools implemented:
  - search_capability_index   (Phase F stub — returns [] until CapabilityIndex lands)
  - generate_tool_code        (uses Claude Bedrock via existing model credentials)
  - test_in_sandbox           (delegates to SandboxManagerClient)
  - request_secret_registration  (writes DynamoDB event; triggers HITL)
  - register_tool             (delegates to FilesystemRegistryService)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# search_capability_index
# ---------------------------------------------------------------------------


def search_capability_index(
    description: str,
    min_confidence: float = 0.6,
    _agent=None,
) -> dict[str, Any]:
    """
    Search the CapabilityIndex for existing tools matching ``description``.

    Phase F stub: CapabilityIndex (LTM-backed semantic search) is not yet
    implemented. Returns an empty match list so the ToolBuilderAgent falls
    through to code generation on every call.

    When Phase F lands, replace this stub with:
        from registry.capability_index import CapabilityIndex
        from interfaces.ltm import LTMInterface
        ci = CapabilityIndex(ltm=LTMInterface(...))
        matches = ci.search(description, min_confidence=min_confidence)

    If a ToolBuilderAgent instance is provided via ``_agent``, delegate to
    its registry for keyword-based fallback search (already works in Phase 2).
    """
    if _agent is not None:
        try:
            matches = _agent.find_tools(description, min_confidence=min_confidence)
            return {"matches": matches, "search_complete": True}
        except Exception as exc:
            logger.warning("search_capability_index registry fallback failed: %s", exc)

    # Phase F stub: no CapabilityIndex yet
    logger.debug(
        "search_capability_index: CapabilityIndex not available (Phase F pending) — "
        "returning empty matches for description=%r",
        description,
    )
    return {"matches": [], "search_complete": True}


# ---------------------------------------------------------------------------
# generate_tool_code
# ---------------------------------------------------------------------------


def generate_tool_code(
    tool_name: str,
    capability_description: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    api_documentation: str = "",
    secrets_required: list | None = None,
    network_allow_list: list | None = None,
    previous_attempt_errors: list | None = None,
    _agent=None,
) -> dict[str, Any]:
    """
    Generate a Python tool implementation.

    Returns a code scaffold annotated with the API documentation so the LLM
    (running in the SDK loop) can complete the implementation in its next turn.

    Design: the SDK agent is Claude — it is already the code generator. This
    tool provides a structured scaffold + API docs so the LLM can write the
    implementation body without re-stating the spec. The scaffold is returned
    as ``scaffold`` in the result; the LLM is expected to replace the ``pass``
    statement with working code on its next tool call (or inline in text).
    """
    from agents.tool_builder.agent import build_tool_scaffold

    scaffold = build_tool_scaffold(
        tool_name=tool_name,
        capability_description=capability_description,
        input_schema=input_schema,
        output_schema=output_schema,
        secrets=secrets_required or [],
        errors=previous_attempt_errors or [],
    )
    return {
        "scaffold": scaffold,
        "instruction": (
            "Complete this scaffold. Replace the pass statement in execute() "
            "with the actual implementation using the API documentation provided. "
            "Use os.getenv() for all secrets. "
            "Return a dict with keys 'success' (bool) and 'data' (any) or 'error' (str)."
        ),
        "api_docs": api_documentation[:2000] if api_documentation else "",
        "generation_attempt": len(previous_attempt_errors or []) + 1,
    }


# ---------------------------------------------------------------------------
# test_in_sandbox
# ---------------------------------------------------------------------------


def test_in_sandbox(
    tool_name: str,
    tool_code: str,
    test_cases: list | None = None,
    _agent=None,
) -> dict[str, Any]:
    """
    Test generated tool code in the sandbox sidecar via SandboxManagerClient.

    Delegates to ToolBuilderAgent.test_tool() which calls SandboxManagerClient
    over HTTP. If no agent instance is provided (unit test / stub path), returns
    a synthetic pass result so the blueprint loop can proceed.
    """
    if _agent is not None:
        try:
            tool_id = f"tools/{tool_name}-generated"
            results = _agent.test_tool(
                tool_id=tool_id,
                tool_code=tool_code,
                test_cases=test_cases or [{"name": "smoke", "input": {}}],
            )
            return {
                **results,
                "tests_passed": results.get("all_passed", False),
                "error": None
                if results.get("all_passed")
                else (f"{results.get('failed', 0)} test case(s) failed"),
            }
        except Exception as exc:
            logger.warning("test_in_sandbox failed: %s", exc)
            return {
                "passed": 0,
                "failed": len(test_cases or [1]),
                "all_passed": False,
                "tests_passed": False,
                "results": [],
                "error": f"Sandbox unavailable: {exc}",
            }

    # Stub path (no agent instance — unit test or dry-run)
    logger.warning(
        "test_in_sandbox called without agent instance; returning stub pass result "
        "for tool_name=%r. Wire _agent in pipeline.py for live sandbox execution.",
        tool_name,
    )
    return {
        "passed": len(test_cases or []),
        "failed": 0,
        "all_passed": True,
        "tests_passed": True,
        "results": [{"case": c.get("name", "stub"), "passed": True} for c in (test_cases or [])],
        "error": None,
    }


# ---------------------------------------------------------------------------
# request_secret_registration
# ---------------------------------------------------------------------------


def request_secret_registration(
    tool_id: str,
    secret_name: str,
    description: str,
    registration_instructions: str = "",
    _task_id: str | None = None,
) -> dict[str, Any]:
    """
    Signal that a secret must be registered before the tool can run.

    Writes a ``secret_registration_required`` event to DynamoDB TaskEventsTable.
    The HITL condition in the blueprint picks this up via BlueprintTracker and
    injects a ``<hitl_request>`` XML block into the agent's next turn, pausing
    execution until a human confirms via nudge.

    If TaskEventsTable is not configured (local dev), logs a warning and returns
    the request details so the blueprint loop can still proceed.
    """
    event_metadata = {
        "tool_id": tool_id,
        "secret_name": secret_name,
        "description": description,
        "registration_instructions": registration_instructions,
    }

    # Best-effort write to DynamoDB TaskEventsTable
    task_events_table = os.environ.get("TASK_EVENTS_TABLE_NAME")
    if task_events_table and _task_id:
        try:
            import time
            from datetime import UTC, datetime

            import boto3

            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            ddb = boto3.resource("dynamodb", region_name=region)
            table = ddb.Table(task_events_table)

            now = datetime.now(UTC)
            ttl = int(now.timestamp()) + 90 * 24 * 60 * 60  # 90 days

            # ULID-style event_id using timestamp + random suffix
            import random

            ts = int(time.time() * 1000)
            rand_suffix = random.randint(0, 0xFFFFFF)
            event_id = f"{ts:013d}{rand_suffix:06x}"

            table.put_item(
                Item={
                    "task_id": _task_id,
                    "event_id": event_id,
                    "event_type": "secret_registration_required",
                    "metadata": event_metadata,
                    "timestamp": now.isoformat(),
                    "ttl": ttl,
                }
            )
            logger.info(
                "secret_registration_required event written for tool_id=%r secret=%r",
                tool_id,
                secret_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write secret_registration_required event to DynamoDB (fail-open): %s",
                exc,
            )
    else:
        logger.warning(
            "request_secret_registration: TASK_EVENTS_TABLE_NAME not set or no "
            "task_id — HITL event not written. tool_id=%r secret=%r",
            tool_id,
            secret_name,
        )

    # Return the request details so the blueprint state machine can track pending secrets
    return {
        "secret_registration_requested": True,
        "tool_id": tool_id,
        "secret_name": secret_name,
        "description": description,
        "registration_instructions": registration_instructions,
        "hitl_trigger": "secret_registration_required",
        "message": (
            f"Secret '{secret_name}' registration requested for tool '{tool_id}'. "
            "Execution will pause until the secret is confirmed as registered."
        ),
    }


# ---------------------------------------------------------------------------
# register_tool
# ---------------------------------------------------------------------------


def register_tool(
    tool_name: str,
    tool_id: str,
    tool_code: str,
    capability_description: str,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    secrets_required: list | None = None,
    network_allow_list: list | None = None,
    test_results: dict[str, Any] | None = None,
    _agent=None,
) -> dict[str, Any]:
    """
    Register a tested tool in FilesystemRegistryService.

    Delegates to ToolBuilderAgent.register_tool_entry() which:
      1. Writes tool.py to src/tools/generated/{tool_id}/
      2. Registers the ToolEntry in the tools registry
      3. Auto-promotes read-only tools with all-passing tests to PRODUCTION

    If no agent instance is provided, returns a stub success result so the
    blueprint loop can complete in unit tests / dry-run mode.
    """
    if _agent is not None:
        try:
            return _agent.register_tool_entry(
                tool_name=tool_name,
                tool_id=tool_id,
                tool_code=tool_code,
                capability_description=capability_description,
                input_schema=input_schema or {},
                output_schema=output_schema or {},
                secrets_required=secrets_required,
                network_allow_list=network_allow_list,
                test_results=test_results,
            )
        except Exception as exc:
            logger.error("register_tool failed: %s", exc)
            return {
                "tool_id": tool_id,
                "status": "failed",
                "registered": False,
                "error": str(exc),
            }

    # Stub path (no agent instance)
    logger.warning(
        "register_tool called without agent instance; returning stub result "
        "for tool_id=%r. Wire _agent in pipeline.py for live registration.",
        tool_id,
    )
    return {
        "tool_id": tool_id,
        "status": "tested",
        "registered": True,
        "stub": True,
    }


# ---------------------------------------------------------------------------
# Local tool handler registry
# ---------------------------------------------------------------------------


def build_local_tool_handlers(agent=None) -> dict[str, Any]:
    """
    Build the local_tool_handlers dict for BlueprintTracker.

    Maps tool names (as declared in meta/tool-builder-v1.yaml) to partial
    callables with ``_agent`` already bound.

    Usage in pipeline.py (Phase C1 wiring):

        from tools.tool_builder_tools import build_local_tool_handlers
        from agents.tool_builder.agent import ToolBuilderAgent

        if config.task_type == "generate_tool":
            tb_agent = ToolBuilderAgent()
            local_tools = build_local_tool_handlers(agent=tb_agent)
            # TODO (Phase C1): pass local_tools to BlueprintTracker so it
            # can intercept SDK tool calls and route them to these callables.

    The interception mechanism (pre-tool hook or synthetic ToolResult) is
    defined in blueprint_tracker.py. This function builds the handler map
    that the tracker needs. See blueprint_tracker.py for the hook protocol.
    """
    import functools

    def _bind(fn, **kwargs):
        """Bind keyword args to a function (partial, but named for readability)."""
        return functools.partial(fn, **kwargs)

    return {
        "search_capability_index": _bind(search_capability_index, _agent=agent),
        "generate_tool_code": _bind(generate_tool_code, _agent=agent),
        "test_in_sandbox": _bind(test_in_sandbox, _agent=agent),
        "request_secret_registration": _bind(request_secret_registration),
        "register_tool": _bind(register_tool, _agent=agent),
    }
