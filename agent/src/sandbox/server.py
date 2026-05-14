"""
SandboxManager HTTP sidecar server — port 8081.

Wraps SandboxManager as a REST API reachable only from agents inside the
toolbuilder-internal VPC subnet via AWS Cloud Map service discovery.

Endpoints
---------
POST /execute  — execute tool code; returns ExecutionResult as JSON
GET  /health   — liveness check; returns {"status": "ok"}

Fail-closed: any unhandled exception returns HTTP 500 with error detail.

Run with:
    uvicorn sandbox.server:app --host 0.0.0.0 --port 8081 --app-dir /app/src
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sandbox.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)

app = FastAPI(title="sandbox-manager-sidecar", version="1.0.0")

_sandbox = SandboxManager()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    tool_id: str
    tool_code: str
    input_payload: dict[str, Any] = {}
    network_allow_list: list[str] = []
    timeout_seconds: int = 30
    output_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — returns {"status": "ok"}."""
    return {"status": "ok"}


@app.post("/execute")
def execute(req: ExecuteRequest) -> dict[str, Any]:
    """
    Execute tool code in a sandboxed environment.

    Returns ExecutionResult fields as a JSON object.
    Any exception raises HTTP 500 (fail-closed).
    """
    try:
        result = _sandbox.execute(
            tool_id=req.tool_id,
            tool_code=req.tool_code,
            input_payload=req.input_payload,
            network_allow_list=req.network_allow_list,
            timeout_seconds=req.timeout_seconds,
            output_schema=req.output_schema,
        )
        return {
            "success": result.success,
            "output": result.output,
            "duration_ms": result.duration_ms,
            "logs": result.logs,
            "schema_valid": result.schema_valid,
            "error": result.error,
            "exit_code": result.exit_code,
        }
    except Exception as exc:
        logger.exception("Unhandled error in /execute for tool_id=%r", req.tool_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
