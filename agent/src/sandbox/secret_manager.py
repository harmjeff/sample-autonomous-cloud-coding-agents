"""
SecretManager — Phase 2 stub backed by an env file.
Same interface as the AWS Secrets Manager wrapper that replaces it in production.

Secrets are stored in .secrets (gitignored) as KEY=VALUE lines, namespaced by tool_id.
Path convention: tool_id/SECRET_NAME → [tool_id]/SECRET_NAME in the env file.

Ported from AKW src/sandbox/secret_manager.py into ABCA agent/src/sandbox/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_SECRETS_FILE = Path(__file__).parent.parent.parent / ".secrets"


@dataclass
class SecretRequirement:
    name: str
    description: str
    registered: bool = False


class SecretNotFoundError(Exception):
    def __init__(self, tool_id: str, secret_name: str) -> None:
        super().__init__(f"Secret '{secret_name}' not registered for tool '{tool_id}'")
        self.tool_id = tool_id
        self.secret_name = secret_name


class SecretManager:
    """
    Phase 2: env file backed secret store.
    Phase 3: replace with AWS Secrets Manager (same interface).

    Secret key format in .secrets file:
      [tools/pubmed-api-v1]/PUBMED_API_KEY=abc123
    """

    def __init__(self, secrets_file: Path | str | None = None) -> None:
        self._path = Path(secrets_file or os.getenv("SECRETS_FILE", str(_DEFAULT_SECRETS_FILE)))

    def get_secrets_for_tool(self, tool_id: str) -> dict[str, str]:
        """
        Return all registered secrets for tool_id.
        Only returns secrets declared under this tool's namespace.
        """
        prefix = f"[{tool_id}]/"
        secrets = {}
        if not self._path.exists():
            return secrets
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(prefix):
                rest = line[len(prefix):]
                if "=" in rest:
                    key, _, value = rest.partition("=")
                    secrets[key.strip()] = value.strip()
        return secrets

    def register_secret(self, tool_id: str, secret_name: str, value: str) -> None:
        """
        Register a secret for a tool. Called by human operator after HITL prompt.
        Appends to .secrets file — never called by agents.
        """
        key = f"[{tool_id}]/{secret_name}={value}"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._path.read_text() if self._path.exists() else ""
        # Remove existing entry for this key if present
        prefix = f"[{tool_id}]/{secret_name}="
        lines = [ln for ln in existing.splitlines() if not ln.startswith(prefix)]
        lines.append(key)
        self._path.write_text("\n".join(lines) + "\n")

    def is_registered(self, tool_id: str, secret_name: str) -> bool:
        return secret_name in self.get_secrets_for_tool(tool_id)

    def list_required_secrets(self, tool_id: str, required: list[str]) -> list[SecretRequirement]:
        registered = self.get_secrets_for_tool(tool_id)
        return [
            SecretRequirement(name=n, description="", registered=n in registered)
            for n in required
        ]

    def get_secret(self, tool_id: str, secret_name: str) -> str:
        """Get a single secret value. Raises SecretNotFoundError if not registered."""
        secrets = self.get_secrets_for_tool(tool_id)
        if secret_name not in secrets:
            raise SecretNotFoundError(tool_id, secret_name)
        return secrets[secret_name]
