"""
SandboxManagerClient — thin HTTP client for the sandbox sidecar service.

Wraps POST /execute on the sandbox-manager ECS sidecar so ToolBuilderAgent
can call SandboxManagerClient.execute(...) with the same signature as the
original SandboxManager.execute(...) — no code changes needed in the caller.

Configuration
-------------
SANDBOX_URL  (env var, default http://sandbox-manager.toolbuilder-internal:8081)
    Base URL of the sandbox sidecar, resolved via AWS Cloud Map inside the
    toolbuilder-internal VPC subnet.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from sandbox.sandbox_manager import ExecutionResult

logger = logging.getLogger(__name__)

_DEFAULT_SANDBOX_URL = "http://sandbox-manager.toolbuilder-internal:8081"
_HTTP_TIMEOUT = 300  # seconds — generous; tool execution can take up to timeout_seconds + overhead


class SandboxManagerClient:
    """
    HTTP client that mirrors the SandboxManager.execute() interface.

    Usage::

        client = SandboxManagerClient()
        result = client.execute(
            tool_id="my-tool",
            tool_code="...",
            input_payload={"x": 1},
        )
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or os.getenv("SANDBOX_URL", _DEFAULT_SANDBOX_URL)).rstrip("/")

    def execute(
        self,
        tool_id: str,
        tool_code: str,
        input_payload: dict[str, Any],
        network_allow_list: list[str] | None = None,
        timeout_seconds: int = 30,
        output_schema: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Delegate execution to the sandbox sidecar via HTTP.

        Raises httpx.HTTPError on transport failures (caller should handle).
        """
        payload: dict[str, Any] = {
            "tool_id": tool_id,
            "tool_code": tool_code,
            "input_payload": input_payload,
            "network_allow_list": network_allow_list or [],
            "timeout_seconds": timeout_seconds,
        }
        if output_schema is not None:
            payload["output_schema"] = output_schema

        url = f"{self._base_url}/execute"
        logger.debug("SandboxManagerClient.execute → %s tool_id=%r", url, tool_id)

        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output"),
            duration_ms=data.get("duration_ms", 0),
            logs=data.get("logs", ""),
            schema_valid=data.get("schema_valid", True),
            error=data.get("error"),
            exit_code=data.get("exit_code", 0),
        )
