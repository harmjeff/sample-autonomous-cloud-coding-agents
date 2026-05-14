"""
Tool Builder Agent — the only agent authorised to generate, test, and register tools.

Ported from AKW src/agents/tool_builder/agent.py.

In ABCA, the ToolBuilderAgent is wired via the meta/tool-builder-v1 blueprint.
The Claude Agent SDK handles the execution loop; this module provides the concrete
tool implementations that the blueprint-driven agent invokes.

Key differences from AKW:
  - AKW used SandboxManager directly (same process).
    ABCA uses SandboxManagerClient (HTTP call to ECS sidecar) — sandbox runs as
    a separate ECS task reachable at SANDBOX_URL.
  - AKW's GenericAgent owns the loop.
    ABCA's loop is the Claude Agent SDK; BlueprintTracker observes state.
  - AKW tool implementations were inline Tool subclasses.
    ABCA exposes them as Python callables in tools/tool_builder_tools.py.
    The BlueprintTracker local_tools hook intercepts tool calls and routes them
    to the appropriate callable (see pipeline.py Phase C1 wiring comment).

Network isolation: only the tool_builder ECS task is allowed to reach the
sandbox sidecar via the toolbuilder-internal VPC subnet (Cloud Map service
discovery). All other agents cannot reach the sandbox.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from registry.filesystem import FilesystemRegistryService
from registry.models import ToolEntry, ToolStatus, SecretRequirement
from sandbox.http_client import SandboxManagerClient
from sandbox.sandbox_manager import SandboxTestCase

# AKW-specific tool imports — not present in ABCA. Wrapped in try/except so
# the module is importable in ABCA without these dependencies. The
# ToolBuilderAgent's own tools (generate_tool_code, test_in_sandbox, etc.)
# are implemented in tools/tool_builder_tools.py; they do NOT depend on
# AKW's src.tools.* hierarchy.
try:
    from src.tools.base import BaseTool, ToolResult  # type: ignore[import-not-found]
except ImportError:
    BaseTool = object  # type: ignore[assignment,misc]
    ToolResult = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools"


class ToolBuilderAgent:
    """
    Facade for the Tool Builder Agent in ABCA.

    In ABCA's architecture the Claude Agent SDK drives the execution loop
    via the meta/tool-builder-v1 blueprint system_prompt and tool specs.
    This class:
      1. Exposes a registry and sandbox client for use by tool_builder_tools.py
      2. Provides _build_tool_scaffold() used by generate_tool_code
      3. Provides register_tool_entry() used by register_tool

    Instantiated once per generate_tool task in pipeline.py (Phase C1 wiring).
    """

    def __init__(self) -> None:
        self._registry = FilesystemRegistryService()
        self._sandbox = SandboxManagerClient()

    # ------------------------------------------------------------------
    # Delegation helpers for tool_builder_tools.py
    # ------------------------------------------------------------------

    def find_tools(self, description: str, min_confidence: float = 0.6) -> list[dict]:
        """Delegate to FilesystemRegistryService.find_tools(); return serialisable list."""
        try:
            matches = self._registry.find_tools(description, min_confidence=min_confidence)
            return [
                {
                    "tool_id": m.tool.tool_id,
                    "name": m.tool.name,
                    "confidence": m.confidence,
                    "description": m.tool.capability_description,
                    "match_reason": m.match_reason,
                }
                for m in matches
            ]
        except Exception as exc:
            logger.warning("ToolBuilderAgent.find_tools failed: %s", exc)
            return []

    def test_tool(
        self,
        tool_id: str,
        tool_code: str,
        test_cases: list[dict],
    ) -> dict:
        """Run test cases via SandboxManagerClient. Returns TestResults dict."""
        cases = [
            SandboxTestCase(
                name=c.get("name", f"case_{i}"),
                input=c.get("input", {}),
                expect_error=c.get("expect_error", False),
            )
            for i, c in enumerate(test_cases or [{"name": "smoke", "input": {}}])
        ]
        # SandboxManagerClient.execute() is the single-execution API.
        # Run each test case separately and aggregate results.
        passed = 0
        failed = 0
        results = []
        for case in cases:
            try:
                result = self._sandbox.execute(
                    tool_id=tool_id,
                    tool_code=tool_code,
                    input_payload=case.input,
                )
                case_passed = result.success != case.expect_error
            except Exception as exc:
                result_error = str(exc)
                case_passed = case.expect_error  # expected error but got exception
                results.append({
                    "case": case.name,
                    "passed": case_passed,
                    "duration_ms": 0,
                    "error": result_error,
                    "schema_valid": True,
                })
                if case_passed:
                    passed += 1
                else:
                    failed += 1
                continue

            if case_passed:
                passed += 1
            else:
                failed += 1

            results.append({
                "case": case.name,
                "passed": case_passed,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "schema_valid": result.schema_valid,
            })

        all_passed = failed == 0
        return {
            "passed": passed,
            "failed": failed,
            "all_passed": all_passed,
            "results": results,
        }

    def register_tool_entry(
        self,
        tool_name: str,
        tool_id: str,
        tool_code: str,
        capability_description: str,
        input_schema: dict,
        output_schema: dict,
        secrets_required: list | None = None,
        network_allow_list: list | None = None,
        test_results: dict | None = None,
    ) -> dict:
        """Write tool code to filesystem and register in FilesystemRegistryService."""
        gen_dir = _TOOLS_DIR / "generated" / tool_id.replace("tools/", "")
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "tool.py").write_text(tool_code)

        secrets = [
            SecretRequirement(
                name=s["name"],
                description=s.get("description", ""),
                write_access=s.get("write_access", False),
            )
            for s in (secrets_required or [])
        ]

        entry = ToolEntry(
            tool_id=tool_id,
            name=tool_name,
            capability_description=capability_description,
            input_schema=input_schema,
            output_schema=output_schema,
            implementation_type="sandboxed",
            code_ref=str(gen_dir / "tool.py"),
            secrets_required=secrets,
            network_allow_list=network_allow_list or [],
            status=ToolStatus.TESTED,
            test_results=test_results or {},
            created_by="tool_builder_agent",
        )
        self._registry.register_tool(entry)

        # Auto-promote read-only tools that passed all tests
        all_passed = (test_results or {}).get("all_passed", False)
        if all_passed and not entry.has_write_access:
            try:
                self._registry.promote_tool(tool_id, ToolStatus.VALIDATED, "auto")
                self._registry.promote_tool(tool_id, ToolStatus.PRODUCTION, "auto")
                status = ToolStatus.PRODUCTION
            except Exception as exc:
                logger.warning("Auto-promotion failed for %s: %s", tool_id, exc)
                status = ToolStatus.TESTED
        else:
            status = ToolStatus.TESTED

        return {"tool_id": tool_id, "status": status, "registered": True}


def build_tool_scaffold(
    tool_name: str,
    capability_description: str,
    input_schema: dict,
    output_schema: dict,
    secrets: list,
    errors: list,
) -> str:
    """
    Generate a Python scaffold for a new tool.

    The LLM fills in the execute() body using the API documentation provided
    in the generate_tool_code call.
    """
    props = input_schema.get("properties", {})
    params = ", ".join(
        f"{k}: {v.get('type', 'str')} = None" for k, v in props.items()
    ) or "**kwargs"

    secret_lines = "\n".join(
        f'        {s["name"]} = os.getenv("{s["name"]}")'
        for s in secrets
    ) or "        pass  # no secrets required"

    error_comment = ""
    if errors:
        error_comment = "\n    # Previous attempt errors to fix:\n" + "\n".join(
            f"    # - {e}" for e in errors[-3:]
        )

    return textwrap.dedent(f"""
        import os
        import json

        class {tool_name.replace('-', '_').replace(' ', '_').title()}Tool:
            name = "{tool_name}"
            description = "{capability_description}"
            {error_comment}

            def execute(self, {params}):
                # Load secrets
        {secret_lines}

                # TODO: implement using the API documentation provided
                # Return {{"success": True, "data": {{...}}}} on success
                # Return {{"success": False, "error": "..."}} on failure
                pass
    """).strip()
