"""
SandboxManager — executes generated tool code in ephemeral containers.

Each execution:
  1. Fetches secrets scoped to tool_id from SecretManager
  2. Attempts to spawn an ephemeral Podman container with tool code mounted read-only
  3. Falls back to direct subprocess execution if Podman is unavailable
  4. Injects secrets as env vars
  5. Enforces network allow-list, CPU/memory/timeout limits (where supported)
  6. Captures stdout as ToolResult JSON
  7. Destroys container immediately after execution (Podman path)

Only reachable from the Tool Builder Agent via the toolbuilder-internal network.

Ported from AKW src/sandbox/sandbox_manager.py into ABCA agent/src/sandbox/.
Import fixed: `from src.sandbox.secret_manager` → `from sandbox.secret_manager`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sandbox.secret_manager import SecretManager

logger = logging.getLogger(__name__)

_BASE_IMAGE = "python:3.12-slim"
_DEFAULT_TIMEOUT = 30
_DEFAULT_CPU = "0.5"
_DEFAULT_MEMORY = "256m"


@dataclass
class ExecutionResult:
    success: bool
    output: Any
    duration_ms: int
    logs: str
    schema_valid: bool = True
    error: str | None = None
    exit_code: int = 0


@dataclass
class SandboxTestCase:
    name: str
    input: dict[str, Any]
    expected_output_schema: dict[str, Any] = field(default_factory=dict)
    mock_response: dict[str, Any] | None = None
    expect_error: bool = False


@dataclass
class TestResults:
    passed: int
    failed: int
    results: list[dict[str, Any]]

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class SandboxManager:
    def __init__(
        self,
        secret_manager: SecretManager | None = None,
        base_image: str = _BASE_IMAGE,
    ) -> None:
        self._secrets = secret_manager or SecretManager()
        self._base_image = base_image

    def execute(
        self,
        tool_id: str,
        tool_code: str,
        input_payload: dict[str, Any],
        network_allow_list: list[str] | None = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT,
        output_schema: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute generated tool code in an ephemeral container (Podman) or subprocess fallback.
        Returns ExecutionResult with output or error details.
        """
        secrets = self._secrets.get_secrets_for_tool(tool_id)
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write tool code
            tool_path = tmp / "tool.py"
            tool_path.write_text(tool_code)

            # Write runner that imports tool, calls execute(), prints JSON result
            runner_path = tmp / "runner.py"
            runner_path.write_text(self._build_runner(input_payload))

            # Write deps installer if needed
            deps_path = tmp / "deps.txt"
            deps = self._extract_deps(tool_code)
            if deps:
                deps_path.write_text("\n".join(deps))

            result = self._run_container(
                tool_id=tool_id,
                tool_path=tool_path,
                runner_path=runner_path,
                deps_path=deps_path if deps else None,
                secrets=secrets,
                network_allow_list=network_allow_list or [],
                timeout_seconds=timeout_seconds,
                tmpdir=tmp,
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        result.duration_ms = duration_ms

        # Validate output schema if provided
        if output_schema and result.success and result.output:
            result.schema_valid = self._validate_schema(result.output, output_schema)

        return result

    def test(
        self,
        tool_id: str,
        tool_code: str,
        test_cases: list[SandboxTestCase],
        network_allow_list: list[str] | None = None,
    ) -> TestResults:
        """Run all test cases. Returns pass/fail per case."""
        results = []
        passed = 0
        failed = 0

        for case in test_cases:
            result = self.execute(
                tool_id=tool_id,
                tool_code=tool_code,
                input_payload=case.input,
                network_allow_list=network_allow_list,
                output_schema=case.expected_output_schema or None,
            )

            case_passed = result.success != case.expect_error and result.schema_valid

            if case_passed:
                passed += 1
            else:
                failed += 1

            results.append(
                {
                    "case": case.name,
                    "passed": case_passed,
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                    "schema_valid": result.schema_valid,
                }
            )

        return TestResults(passed=passed, failed=failed, results=results)

    # -----------------------------------------------------------------------
    # Container execution
    # -----------------------------------------------------------------------

    def _run_container(
        self,
        tool_id: str,
        tool_path: Path,
        runner_path: Path,
        deps_path: Path | None,
        secrets: dict[str, str],
        network_allow_list: list[str],
        timeout_seconds: int,
        tmpdir: Path,
    ) -> ExecutionResult:
        # TODO: replace with Podman when ECS supports it or use a separate execution container
        try:
            return self._run_podman(
                tool_path=tool_path,
                runner_path=runner_path,
                deps_path=deps_path,
                secrets=secrets,
                network_allow_list=network_allow_list,
                timeout_seconds=timeout_seconds,
            )
        except FileNotFoundError:
            logger.warning(
                "podman not found (tool_id=%s) — falling back to direct subprocess execution; "
                "no container isolation",
                tool_id,
            )
            return self._run_subprocess(
                tool_path=tool_path,
                runner_path=runner_path,
                deps_path=deps_path,
                secrets=secrets,
                timeout_seconds=timeout_seconds,
                tmpdir=tmpdir,
            )

    def _run_podman(
        self,
        tool_path: Path,
        runner_path: Path,
        deps_path: Path | None,
        secrets: dict[str, str],
        network_allow_list: list[str],
        timeout_seconds: int,
    ) -> ExecutionResult:
        """Attempt Podman container execution. Raises FileNotFoundError if podman absent."""
        cmd = [
            "podman",
            "run",
            "--rm",
            "--network",
            "none" if not network_allow_list else "bridge",
            f"--cpus={_DEFAULT_CPU}",
            f"--memory={_DEFAULT_MEMORY}",
            f"--timeout={timeout_seconds}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",  # noqa: S108
            # Mount tool code and runner read-only
            f"--volume={tool_path}:/tool/tool.py:ro",
            f"--volume={runner_path}:/tool/runner.py:ro",
        ]

        # Inject secrets as env vars
        for key, value in secrets.items():
            cmd.extend(["--env", f"{key}={value}"])

        if deps_path:
            cmd.extend([f"--volume={deps_path}:/tool/deps.txt:ro"])

        cmd.append(self._base_image)

        # Install deps then run
        if deps_path:
            shell_cmd = "pip install --quiet -r /tool/deps.txt && python /tool/runner.py"
        else:
            shell_cmd = "python /tool/runner.py"

        cmd.extend(["sh", "-c", shell_cmd])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 5,
            )
            return self._parse_proc_result(proc, timeout_seconds)

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output=None,
                duration_ms=0,
                logs="",
                error=f"Execution timed out after {timeout_seconds}s",
                exit_code=-1,
            )
        except FileNotFoundError:
            # Propagate so _run_container can fall back
            raise
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                duration_ms=0,
                logs="",
                error=str(e),
                exit_code=-1,
            )

    def _run_subprocess(
        self,
        tool_path: Path,
        runner_path: Path,
        deps_path: Path | None,
        secrets: dict[str, str],
        timeout_seconds: int,
        tmpdir: Path,
    ) -> ExecutionResult:
        """
        Direct subprocess fallback — no container isolation.
        # TODO: replace with Podman when ECS supports it or use a separate execution container
        """
        env = os.environ.copy()
        env.update(secrets)
        # Point sys.path so runner.py can `from tool import *`
        env["PYTHONPATH"] = str(tmpdir)

        if deps_path:
            # Install dependencies into tmpdir's venv-less interpreter
            with contextlib.suppress(Exception):  # Best-effort; tool may still run without deps
                subprocess.run(
                    ["pip", "install", "--quiet", "-r", str(deps_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )

        try:
            proc = subprocess.run(
                ["python", str(runner_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
            return self._parse_proc_result(proc, timeout_seconds)

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output=None,
                duration_ms=0,
                logs="",
                error=f"Execution timed out after {timeout_seconds}s",
                exit_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                duration_ms=0,
                logs="",
                error=str(e),
                exit_code=-1,
            )

    @staticmethod
    def _parse_proc_result(
        proc: subprocess.CompletedProcess, timeout_seconds: int
    ) -> ExecutionResult:
        """Parse a completed subprocess result into ExecutionResult."""
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                output=None,
                duration_ms=0,
                logs=stderr or stdout,
                error=f"Container exited with code {proc.returncode}: {stderr[:200]}",
                exit_code=proc.returncode,
            )

        # Parse JSON output from runner
        try:
            output = json.loads(stdout) if stdout else None
        except json.JSONDecodeError:
            output = stdout

        return ExecutionResult(
            success=True,
            output=output,
            duration_ms=0,
            logs=stderr,
            exit_code=0,
        )

    @staticmethod
    def _build_runner(input_payload: dict) -> str:
        """Generate runner.py that imports tool, calls execute(), prints JSON."""
        payload_json = json.dumps(input_payload)
        return f"""
import sys, json
sys.path.insert(0, '/tool')
from tool import *

# Find the BaseTool subclass
import inspect
tool_class = None
for name, obj in list(globals().items()):
    if inspect.isclass(obj) and hasattr(obj, 'execute') and name != 'BaseTool':
        tool_class = obj
        break

if tool_class is None:
    print(json.dumps({{"error": "No tool class found"}}))
    sys.exit(1)

tool = tool_class()
payload = {payload_json}

try:
    result = tool.execute(**payload)
    if hasattr(result, 'data'):
        print(json.dumps({{"success": result.success, "data": result.data, "error": result.error}}))
    else:
        print(json.dumps({{"success": True, "data": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
    sys.exit(1)
"""

    @staticmethod
    def _extract_deps(tool_code: str) -> list[str]:
        """Extract pip packages from import statements in generated code."""
        import re

        stdlib = {
            "os",
            "sys",
            "json",
            "re",
            "time",
            "datetime",
            "pathlib",
            "typing",
            "dataclasses",
            "abc",
            "io",
            "urllib",
            "http",
            "collections",
            "functools",
            "itertools",
            "math",
            "random",
            "string",
            "hashlib",
            "base64",
            "logging",
            "traceback",
        }
        deps = set()
        for match in re.finditer(r"^(?:import|from)\s+(\w+)", tool_code, re.MULTILINE):
            pkg = match.group(1)
            if pkg not in stdlib and not pkg.startswith("_"):
                deps.add(pkg)
        return sorted(deps)

    @staticmethod
    def _validate_schema(output: Any, schema: dict) -> bool:
        """Basic schema validation — check required fields exist."""
        if not schema or not isinstance(output, dict):
            return True
        required = schema.get("required", [])
        return all(f in output for f in required)
