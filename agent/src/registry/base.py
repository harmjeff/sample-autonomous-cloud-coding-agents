"""Abstract RegistryService interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from registry.models import (
        Blueprint,
        PromotionStatus,
        ToolEntry,
        ToolMatch,
        ToolStatus,
    )


class RegistryService(ABC):
    # -----------------------------------------------------------------------
    # Blueprint operations
    # -----------------------------------------------------------------------

    @abstractmethod
    def get_blueprint(self, blueprint_id: str) -> Blueprint:
        """Fetch blueprint by exact ID. Raises BlueprintNotFoundError."""

    @abstractmethod
    def find_blueprint_for_task(self, task_type: str) -> Blueprint | None:
        """Find production blueprint for task type. None if none exists."""

    @abstractmethod
    def list_blueprints(
        self,
        status: PromotionStatus | None = None,
        task_type: str | None = None,
    ) -> list[Blueprint]:
        """List blueprints, optionally filtered."""

    @abstractmethod
    def register_blueprint(self, blueprint: Blueprint) -> None:
        """Register or update a blueprint. Validates schema."""

    @abstractmethod
    def promote_blueprint(
        self,
        blueprint_id: str,
        to: PromotionStatus,
        promoted_by: str,
        notes: str = "",
    ) -> None:
        """
        Promote blueprint to new status.
        Valid chain: draft → validated → production → deprecated
        Promoting to production auto-deprecates previous production blueprint
        for the same task_types.
        """

    @abstractmethod
    def get_blueprint_dependencies(self, blueprint_id: str) -> list[ToolEntry]:
        """Return all ToolEntry objects required by this blueprint."""

    # -----------------------------------------------------------------------
    # Tool operations
    # -----------------------------------------------------------------------

    @abstractmethod
    def get_tool(self, tool_name: str) -> ToolEntry:
        """
        Fetch tool by name. Only returns validated/production tools.
        Raises ToolNotFoundError.
        """

    @abstractmethod
    def find_tools(
        self,
        capability_description: str,
        min_confidence: float = 0.6,
        status_filter: list[ToolStatus] | None = None,
    ) -> list[ToolMatch]:
        """Semantic search over tool capability descriptions."""

    @abstractmethod
    def register_tool(self, tool: ToolEntry) -> None:
        """Register a new tool or update an existing one."""

    @abstractmethod
    def promote_tool(
        self,
        tool_id: str,
        to: ToolStatus,
        promoted_by: str,
        notes: str = "",
    ) -> None:
        """
        Promote tool to new status.
        Auto-promotion: read-only tool with all tests passing →
        tested → validated without human review.
        """

    @abstractmethod
    def record_tool_execution(
        self,
        tool_id: str,
        success: bool,
        duration_ms: int,
        error_type: str | None = None,
    ) -> None:
        """Record a production execution. Updates use_count and error_rate."""

    @abstractmethod
    def get_tool_any_status(self, tool_id: str) -> ToolEntry:
        """Fetch tool regardless of status. For management operations."""

    @abstractmethod
    def list_tools(
        self,
        status: ToolStatus | None = None,
        has_write_access: bool | None = None,
    ) -> list[ToolEntry]:
        """List tools with optional filters."""

    # -----------------------------------------------------------------------
    # Dependency resolution — used by pre-flight
    # -----------------------------------------------------------------------

    @abstractmethod
    def resolve_task(self, task_type: str) -> tuple[Blueprint, list[ToolEntry]]:
        """
        Single pre-flight call: find production blueprint + all tool dependencies.

        Raises:
          BlueprintNotFoundError   — no production blueprint for task_type
          MissingToolError         — blueprint requires tool not in registry
          ToolNotReadyError        — required tool exists but not production
        """
