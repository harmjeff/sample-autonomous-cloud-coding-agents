"""Domain models for the Blueprint and Tool registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class PromotionStatus(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class ToolStatus(StrEnum):
    GENERATED = "generated"
    TESTED = "tested"
    VALIDATED = "validated"
    PRODUCTION = "production"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class SecretRequirement:
    name: str
    description: str
    scope: str = "this_tool_only"
    write_access: bool = False


@dataclass
class Blueprint:
    id: str
    version: str
    task_types: list[str]
    system_prompt: str
    tools: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    state_schema: dict[str, Any]
    hitl_conditions: list[dict[str, Any]]
    output_schema: dict[str, Any]
    quality_checkpoints: list[dict[str, Any]]
    parameters: dict[str, Any]
    required_ltm_capabilities: list[str]
    max_iterations: int
    max_tokens: int
    status: PromotionStatus = PromotionStatus.DRAFT
    required_tool_names: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "human"
    validation_notes: str = ""
    # task_mode: 'coding' requires repo + git setup; 'knowledge' is repo-free
    task_mode: str = "coding"
    # read_only: True means the agent cannot write/edit files (e.g. pr_review)
    read_only: bool = False

    @property
    def tool_names(self) -> list[str]:
        """Tool names declared in tools: section, falling back to required_tool_names."""
        from_tools = [t.get("toolSpec", t).get("name", "") for t in self.tools if t]
        names = [n for n in from_tools if n]
        if not names:
            names = self.required_tool_names
        return names


@dataclass
class ToolEntry:
    tool_id: str
    name: str
    capability_description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    implementation_type: str  # "local" | "sandboxed"
    code_ref: str
    secrets_required: list[SecretRequirement] = field(default_factory=list)
    network_allow_list: list[str] = field(default_factory=list)
    status: ToolStatus = ToolStatus.GENERATED
    test_results: dict[str, Any] = field(default_factory=dict)
    created_by: str = "tool_builder_agent"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    production_use_count: int = 0
    error_rate: float = 0.0

    @property
    def has_write_access(self) -> bool:
        return any(s.write_access for s in self.secrets_required)

    @property
    def is_available(self) -> bool:
        return self.status in (ToolStatus.VALIDATED, ToolStatus.PRODUCTION)


@dataclass
class ToolMatch:
    tool: ToolEntry
    confidence: float
    match_reason: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BlueprintNotFoundError(Exception):
    def __init__(self, identifier: str) -> None:
        super().__init__(f"Blueprint not found: '{identifier}'")
        self.identifier = identifier


class MultipleBlueprintsError(Exception):
    def __init__(self, task_type: str, ids: list[str]) -> None:
        super().__init__(f"Multiple production blueprints for '{task_type}': {ids}")
        self.task_type = task_type
        self.blueprint_ids = ids


class BlueprintValidationError(Exception):
    def __init__(self, blueprint_id: str, errors: list[str]) -> None:
        super().__init__(f"Blueprint '{blueprint_id}' invalid: {'; '.join(errors)}")
        self.blueprint_id = blueprint_id
        self.errors = errors


class InvalidPromotionError(Exception):
    def __init__(self, from_status: str, to_status: str) -> None:
        super().__init__(f"Invalid promotion: {from_status} → {to_status}")


class ToolNotFoundError(Exception):
    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool not found: '{tool_name}'")
        self.tool_name = tool_name


class MissingToolError(Exception):
    def __init__(self, blueprint_id: str, tool_name: str) -> None:
        super().__init__(
            f"Blueprint '{blueprint_id}' requires tool '{tool_name}' which is not registered"
        )
        self.blueprint_id = blueprint_id
        self.tool_name = tool_name


class ToolNotReadyError(Exception):
    def __init__(self, tool_id: str, status: ToolStatus) -> None:
        super().__init__(f"Tool '{tool_id}' exists but is not production-ready (status={status})")
        self.tool_id = tool_id
        self.current_status = status
