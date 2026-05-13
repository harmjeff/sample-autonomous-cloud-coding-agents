"""
FilesystemRegistryService — Phase 2 backing store.

Blueprints: YAML files in src/blueprints/**/*.yaml
            Index:     src/blueprints/registry.yaml

Tools:      Python files in src/tools/local/ and src/tools/generated/
            Metadata:  src/tools/metadata/*.yaml  (local tools)
                       src/tools/generated/{id}/metadata.yaml
            Index:     src/tools/registry.yaml

Semantic search (find_tools) uses substring matching against
capability_description for Phase 2. Replace with LTM CapabilityIndex in Phase 3.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from registry.base import RegistryService
from registry.models import (
    Blueprint,
    BlueprintNotFoundError,
    BlueprintValidationError,
    InvalidPromotionError,
    MissingToolError,
    MultipleBlueprintsError,
    PromotionStatus,
    SecretRequirement,
    ToolEntry,
    ToolMatch,
    ToolNotFoundError,
    ToolNotReadyError,
    ToolStatus,
)

_BP_REQUIRED_FIELDS = ["id", "version", "task_types", "system_prompt", "tools", "max_iterations"]
_VALID_BP_TRANSITIONS = {
    PromotionStatus.DRAFT: {PromotionStatus.VALIDATED},
    PromotionStatus.VALIDATED: {PromotionStatus.PRODUCTION, PromotionStatus.DRAFT},
    PromotionStatus.PRODUCTION: {PromotionStatus.DEPRECATED},
    PromotionStatus.DEPRECATED: set(),
}
_VALID_TOOL_TRANSITIONS = {
    ToolStatus.GENERATED: {ToolStatus.TESTED, ToolStatus.FAILED},
    ToolStatus.TESTED: {ToolStatus.VALIDATED, ToolStatus.FAILED},
    ToolStatus.VALIDATED: {ToolStatus.PRODUCTION, ToolStatus.FAILED},
    ToolStatus.PRODUCTION: {ToolStatus.DEPRECATED, ToolStatus.FAILED},
    ToolStatus.FAILED: {ToolStatus.GENERATED},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FilesystemRegistryService(RegistryService):
    def __init__(
        self,
        blueprints_dir: Path | str | None = None,
        tools_dir: Path | str | None = None,
        capability_index=None,  # Optional[CapabilityIndex]
    ) -> None:
        repo_root = Path(__file__).parent.parent
        self._bp_dir = Path(blueprints_dir or (repo_root / "blueprints"))
        self._tools_dir = Path(tools_dir or (repo_root / "tools"))
        self._bp_index_path = self._bp_dir / "registry.yaml"
        self._tool_index_path = self._tools_dir / "registry.yaml"
        self._capability_index = capability_index  # None = keyword fallback
        self._ensure_indexes()

    # -----------------------------------------------------------------------
    # Index helpers
    # -----------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        self._bp_dir.mkdir(parents=True, exist_ok=True)
        self._tools_dir.mkdir(parents=True, exist_ok=True)
        if not self._bp_index_path.exists():
            self._write_yaml(self._bp_index_path, {"blueprints": []})
        if not self._tool_index_path.exists():
            self._write_yaml(self._tool_index_path, {"tools": []})

    def _load_bp_index(self) -> list[dict]:
        return self._read_yaml(self._bp_index_path).get("blueprints", [])

    def _save_bp_index(self, entries: list[dict]) -> None:
        self._write_yaml(self._bp_index_path, {"blueprints": entries})

    def _load_tool_index(self) -> list[dict]:
        return self._read_yaml(self._tool_index_path).get("tools", [])

    def _save_tool_index(self, entries: list[dict]) -> None:
        self._write_yaml(self._tool_index_path, {"tools": entries})

    # -----------------------------------------------------------------------
    # Blueprint operations
    # -----------------------------------------------------------------------

    def get_blueprint(self, blueprint_id: str) -> Blueprint:
        path = self._bp_path(blueprint_id)
        if not path.exists():
            raise BlueprintNotFoundError(blueprint_id)
        # Pass index entry so required_tool_names can be merged in
        index_entry = next((e for e in self._load_bp_index() if e["id"] == blueprint_id), {})
        return self._load_blueprint(path, index_entry)

    def find_blueprint_for_task(self, task_type: str) -> Blueprint | None:
        matches = [
            e
            for e in self._load_bp_index()
            if task_type in e.get("task_types", [])
            and e.get("status") == PromotionStatus.PRODUCTION
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise MultipleBlueprintsError(task_type, [m["id"] for m in matches])
        return self.get_blueprint(matches[0]["id"])

    def list_blueprints(
        self,
        status: PromotionStatus | None = None,
        task_type: str | None = None,
    ) -> list[Blueprint]:
        entries = self._load_bp_index()
        if status:
            entries = [e for e in entries if e.get("status") == status]
        if task_type:
            entries = [e for e in entries if task_type in e.get("task_types", [])]
        blueprints = []
        for e in entries:
            try:
                blueprints.append(self.get_blueprint(e["id"]))
            except BlueprintNotFoundError:
                continue
        return blueprints

    def register_blueprint(self, blueprint: Blueprint) -> None:
        self._validate_blueprint(blueprint)
        path = self._bp_path(blueprint.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_yaml(path, self._blueprint_to_dict(blueprint))

        index = self._load_bp_index()
        existing = next((e for e in index if e["id"] == blueprint.id), None)
        entry = {
            "id": blueprint.id,
            "version": blueprint.version,
            "status": blueprint.status,
            "task_types": blueprint.task_types,
            "required_tool_names": blueprint.tool_names,
            "created_by": blueprint.created_by,
            "updated_at": _now(),
        }
        if existing:
            index = [entry if e["id"] == blueprint.id else e for e in index]
        else:
            entry["created_at"] = _now()
            index.append(entry)
        self._save_bp_index(index)

    def promote_blueprint(
        self,
        blueprint_id: str,
        to: PromotionStatus,
        promoted_by: str,
        notes: str = "",
    ) -> None:
        bp = self.get_blueprint(blueprint_id)
        allowed = _VALID_BP_TRANSITIONS.get(bp.status, set())
        if to not in allowed:
            raise InvalidPromotionError(bp.status, to)

        # Auto-deprecate previous production blueprint for same task types
        if to == PromotionStatus.PRODUCTION:
            for existing in self.list_blueprints(status=PromotionStatus.PRODUCTION):
                if existing.id != blueprint_id and any(
                    t in existing.task_types for t in bp.task_types
                ):
                    self._update_bp_status(existing.id, PromotionStatus.DEPRECATED)

        self._update_bp_status(blueprint_id, to)
        if notes:
            bp.validation_notes = notes
            self.register_blueprint(bp)

    def get_blueprint_dependencies(self, blueprint_id: str) -> list[ToolEntry]:
        bp = self.get_blueprint(blueprint_id)
        deps = []
        for name in bp.tool_names:
            if not name:
                continue
            try:
                deps.append(self.get_tool_any_status(name))
            except ToolNotFoundError:
                raise MissingToolError(blueprint_id, name)
        return deps

    # -----------------------------------------------------------------------
    # Tool operations
    # -----------------------------------------------------------------------

    def get_tool(self, tool_name: str) -> ToolEntry:
        tool = self.get_tool_any_status(tool_name)
        if not tool.is_available:
            raise ToolNotReadyError(tool.tool_id, tool.status)
        return tool

    def get_tool_any_status(self, tool_id: str) -> ToolEntry:
        """Fetch by name OR tool_id."""
        entry = next(
            (
                e
                for e in self._load_tool_index()
                if e.get("name") == tool_id or e.get("tool_id") == tool_id
            ),
            None,
        )
        if not entry:
            raise ToolNotFoundError(tool_id)
        return self._load_tool(entry)

    def find_tools(
        self,
        capability_description: str,
        min_confidence: float = 0.6,
        status_filter: list[ToolStatus] | None = None,
    ) -> list[ToolMatch]:
        """
        Semantic search if CapabilityIndex is configured; keyword fallback otherwise.
        """
        statuses = set(status_filter or [ToolStatus.VALIDATED, ToolStatus.PRODUCTION])
        available = [t for t in self.list_tools() if t.status in statuses]

        if self._capability_index is not None:
            return self._capability_index.search(
                capability_description,
                min_confidence=min_confidence,
                available_tools=available,
            )

        # Keyword fallback
        query_words = set(capability_description.lower().split())
        matches = []
        for tool in available:
            desc_words = set(tool.capability_description.lower().split())
            overlap = query_words & desc_words
            if not overlap:
                continue
            confidence = len(overlap) / max(len(query_words), 1)
            if confidence >= min_confidence:
                matches.append(
                    ToolMatch(
                        tool=tool,
                        confidence=min(confidence, 1.0),
                        match_reason=f"Matched keywords: {', '.join(sorted(overlap))}",
                    )
                )
        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    def register_tool(self, tool: ToolEntry) -> None:
        # Update CapabilityIndex if configured
        if self._capability_index is not None and tool.capability_description:
            try:
                self._capability_index.register(tool)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("CapabilityIndex register failed: %s", e)

        # Write metadata
        meta_path = self._tool_meta_path(tool.tool_id, tool.implementation_type)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_yaml(meta_path, self._tool_to_dict(tool))

        # Update index
        index = self._load_tool_index()
        existing = next((e for e in index if e.get("name") == tool.name), None)
        entry = {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "status": tool.status,
            "implementation_type": tool.implementation_type,
            "code_ref": tool.code_ref,
            "write_access": tool.has_write_access,
            "updated_at": _now(),
        }
        if existing:
            index = [entry if e.get("name") == tool.name else e for e in index]
        else:
            entry["created_at"] = _now()
            index.append(entry)
        self._save_tool_index(index)

    def promote_tool(
        self,
        tool_id: str,
        to: ToolStatus,
        promoted_by: str,
        notes: str = "",
    ) -> None:
        try:
            tool = self.get_tool_any_status(tool_id)
        except ToolNotFoundError:
            raise

        allowed = _VALID_TOOL_TRANSITIONS.get(tool.status, set())
        if to not in allowed:
            raise InvalidPromotionError(tool.status, to)

        # Auto-promotion: read-only + all tests passed → skip human review
        if to == ToolStatus.VALIDATED and not tool.has_write_access:
            tests = tool.test_results
            if tests and all(r.get("passed") for r in tests.get("results", [])):
                pass  # allow auto-promotion

        tool.status = to
        self.register_tool(tool)

    def record_tool_execution(
        self,
        tool_id: str,
        success: bool,
        duration_ms: int,
        error_type: str | None = None,
        task_type: str | None = None,
        instructions_summary: str = "",
        output_quality: str = "complete",
    ) -> None:
        if not self._tool_exists_by_id(tool_id):
            return
        tool = self.get_tool_any_status(tool_id)

        # Update in-memory metrics
        tool.production_use_count += 1
        alpha = 0.1
        outcome = 0.0 if success else 1.0
        tool.error_rate = alpha * outcome + (1 - alpha) * tool.error_rate
        self.register_tool(tool)

        # Write usage record to CapabilityIndex (LTM) for context-aware search
        if self._capability_index is not None and task_type:
            try:
                self._capability_index.record_usage(
                    tool=tool,
                    task_type=task_type,
                    instructions_summary=instructions_summary,
                    success=success,
                    duration_ms=duration_ms,
                    error_type=error_type,
                    output_quality=output_quality,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("CapabilityIndex.record_usage failed: %s", e)

    def list_tools(
        self,
        status: ToolStatus | None = None,
        has_write_access: bool | None = None,
    ) -> list[ToolEntry]:
        entries = self._load_tool_index()
        if status:
            entries = [e for e in entries if e.get("status") == status]
        if has_write_access is not None:
            entries = [e for e in entries if e.get("write_access") == has_write_access]
        tools = []
        for e in entries:
            try:
                tools.append(self._load_tool(e))
            except Exception:
                continue
        return tools

    # -----------------------------------------------------------------------
    # Dependency resolution
    # -----------------------------------------------------------------------

    # Tools that are GenericAgent built-ins — not registered in the ToolRegistry
    _BUILTIN_TOOLS = frozenset(
        {
            # GenericAgent built-ins
            "read_memory",
            "ltm_read",
            "ltm_write",
            "save_finding",
            "stm",
            "finish",
            "flag_conflict",
            "score_relevance",
            "extract_biomarkers",
            "record_outcome",
            # ToolBuilderAgent built-ins
            "search_capability_index",
            "generate_tool_code",
            "test_in_sandbox",
            "request_secret_registration",
            "register_tool",
            # BlueprintBuilderAgent built-ins (future)
            "search_blueprint_index",
            "generate_blueprint",
            "validate_blueprint",
            "register_blueprint_action",
        }
    )

    def resolve_task(self, task_type: str) -> tuple[Blueprint, list[ToolEntry]]:
        bp = self.find_blueprint_for_task(task_type)
        if bp is None:
            raise BlueprintNotFoundError(task_type)

        tools = []
        for name in bp.tool_names:
            if not name or name in self._BUILTIN_TOOLS:
                continue  # built-in tools are always available
            try:
                entry = self.get_tool_any_status(name)
            except ToolNotFoundError:
                raise MissingToolError(bp.id, name)
            if entry.status == ToolStatus.PRODUCTION:
                tools.append(entry)
            else:
                raise ToolNotReadyError(entry.tool_id, entry.status)

        return bp, tools

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _bp_path(self, blueprint_id: str) -> Path:
        return self._bp_dir / f"{blueprint_id}.yaml"

    def _tool_meta_path(self, tool_id: str, impl_type: str, tool_name: str = "") -> Path:
        id_suffix = tool_id.split("/")[-1] if "/" in tool_id else tool_id
        if impl_type == "local":
            # Prefer tool name (e.g. web_search.yaml) over tool_id suffix (web-search-v1.yaml)
            if tool_name:
                name_path = self._tools_dir / "metadata" / f"{tool_name}.yaml"
                if name_path.exists():
                    return name_path
            return self._tools_dir / "metadata" / f"{id_suffix}.yaml"
        return self._tools_dir / "generated" / id_suffix / "metadata.yaml"

    def _tool_exists_by_id(self, tool_id: str) -> bool:
        return any(e.get("tool_id") == tool_id for e in self._load_tool_index())

    def _update_bp_status(self, blueprint_id: str, status: PromotionStatus) -> None:
        # Update index
        index = self._load_bp_index()
        for e in index:
            if e["id"] == blueprint_id:
                e["status"] = status.value
                e["updated_at"] = _now()
        self._save_bp_index(index)
        # Also update the YAML file so get_blueprint reads the new status
        path = self._bp_path(blueprint_id)
        if path.exists():
            data = self._read_yaml(path)
            data["status"] = status.value
            self._write_yaml(path, data)

    def _validate_blueprint(self, blueprint: Blueprint) -> None:
        errors = [f for f in _BP_REQUIRED_FIELDS if not getattr(blueprint, f, None)]
        if errors:
            raise BlueprintValidationError(blueprint.id, [f"Missing: {f}" for f in errors])

    def _load_blueprint(self, path: Path, index_entry: dict | None = None) -> Blueprint:
        data = self._read_yaml(path)
        # Merge required_tool_names from index if not in YAML
        if index_entry and not data.get("required_tool_names"):
            data["required_tool_names"] = index_entry.get("required_tool_names", [])
        if index_entry and data.get("status") is None:
            data["status"] = index_entry.get("status", PromotionStatus.DRAFT)
        return Blueprint(
            id=data["id"],
            version=str(data.get("version", "1")),
            task_types=data.get("task_types", []),
            system_prompt=data.get("system_prompt", data.get("approach", "")),
            tools=data.get("tools", []),
            phases=data.get("phases", []),
            state_schema=data.get("state_schema", {}),
            hitl_conditions=data.get("hitl_conditions", []),
            output_schema=data.get("output_schema", {}),
            quality_checkpoints=data.get("quality_checkpoints", []),
            parameters=data.get("parameters", {}),
            required_ltm_capabilities=data.get("required_ltm_capabilities", []),
            max_iterations=int(data.get("max_iterations", 20)),
            max_tokens=int(data.get("max_tokens", 50000)),
            status=PromotionStatus(data.get("status", PromotionStatus.DRAFT)),
            required_tool_names=data.get("required_tool_names", []),
            created_by=data.get("created_by", "human"),
            validation_notes=data.get("validation_notes", ""),
            task_mode=data.get("task_mode", "coding"),
            read_only=bool(data.get("read_only", False)),
        )

    def _blueprint_to_dict(self, bp: Blueprint) -> dict[str, Any]:
        return {
            "id": bp.id,
            "version": bp.version,
            "task_types": bp.task_types,
            "system_prompt": bp.system_prompt,
            "tools": bp.tools,
            "phases": bp.phases,
            "state_schema": bp.state_schema,
            "hitl_conditions": bp.hitl_conditions,
            "output_schema": bp.output_schema,
            "quality_checkpoints": bp.quality_checkpoints,
            "parameters": bp.parameters,
            "required_ltm_capabilities": bp.required_ltm_capabilities,
            "max_iterations": bp.max_iterations,
            "max_tokens": bp.max_tokens,
            "status": bp.status,
            "required_tool_names": bp.required_tool_names,
            "created_by": bp.created_by,
            "validation_notes": bp.validation_notes,
        }

    def _load_tool(self, index_entry: dict) -> ToolEntry:
        tool_id = index_entry.get("tool_id", index_entry.get("name", ""))
        impl_type = index_entry.get("implementation_type", "local")
        tool_name = index_entry.get("name", "")
        meta_path = self._tool_meta_path(tool_id, impl_type, tool_name)
        data = self._read_yaml(meta_path) if meta_path.exists() else index_entry
        secrets = [
            SecretRequirement(
                name=s["name"],
                description=s.get("description", ""),
                scope=s.get("scope", "this_tool_only"),
                write_access=s.get("write_access", False),
            )
            for s in data.get("secrets_required", [])
        ]
        return ToolEntry(
            tool_id=data.get("tool_id", tool_id),
            name=data.get("name", index_entry.get("name", "")),
            capability_description=data.get("capability_description", ""),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            implementation_type=data.get("implementation_type", impl_type),
            code_ref=data.get("code_ref", index_entry.get("code_ref", "")),
            secrets_required=secrets,
            network_allow_list=data.get("network_allow_list", []),
            status=ToolStatus(data.get("status", index_entry.get("status", ToolStatus.GENERATED))),
            test_results=data.get("test_results", {}),
            created_by=data.get("created_by", "human"),
            production_use_count=data.get("production_use_count", 0),
            error_rate=data.get("error_rate", 0.0),
        )

    def _tool_to_dict(self, tool: ToolEntry) -> dict[str, Any]:
        return {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "capability_description": tool.capability_description,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "implementation_type": tool.implementation_type,
            "code_ref": tool.code_ref,
            "secrets_required": [
                {
                    "name": s.name,
                    "description": s.description,
                    "scope": s.scope,
                    "write_access": s.write_access,
                }
                for s in tool.secrets_required
            ],
            "network_allow_list": tool.network_allow_list,
            "status": tool.status,
            "test_results": tool.test_results,
            "created_by": tool.created_by,
            "production_use_count": tool.production_use_count,
            "error_rate": tool.error_rate,
        }

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open() as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        from enum import Enum

        def _coerce(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(i) for i in obj]
            return obj

        with path.open("w") as f:
            yaml.dump(
                _coerce(data), f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
