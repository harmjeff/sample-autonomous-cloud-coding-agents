"""BlueprintTracker — wires blueprint phases and PatternEvaluator into ABCA's hook system.

Architecture
------------
ABCA's hook system (PR #52) exposes two seams:
  - PostToolUse hook: fires after every tool call with the tool name + output
  - between_turns_hooks registry: fires at the Stop hook before each new turn

BlueprintTracker uses both:
  1. PostToolUse: update_from_tool() — mutate state dict, advance phase tracking
  2. between_turns_hooks: evaluate_patterns() — run PatternEvaluator, inject
     HITL messages as synthetic user messages if patterns fire

The blueprint's system_prompt replaces the hardcoded prompt at pipeline start.
This is wired from pipeline.py when a blueprint is resolved for the task type.

Design principles
-----------------
- Fail-open: any exception in tracking silently no-ops rather than crashing the task
- No LLM calls: state mutations and phase signals are deterministic
- Single instance per task: created in pipeline.py, passed to hooks via closure
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pattern_evaluator import FiredCondition, PatternEvaluator
from shell import log

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State mutation rules — extend as new blueprint tool patterns are defined
# ---------------------------------------------------------------------------


def _apply_state_mutations(
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    state: dict[str, Any],
    blueprint_mutations: dict[str, Any] | None = None,
) -> None:
    """Update state dict based on tool call. Fail-open."""
    try:
        # Blueprint-declared mutations (state_mutations field)
        if blueprint_mutations:
            for field_name, rule in blueprint_mutations.items():
                if rule.get("tool") == tool_name:
                    action = rule.get("action", "append")
                    value = rule.get("value", tool_output)
                    if action == "append":
                        lst = state.setdefault(field_name, [])
                        if isinstance(lst, list):
                            lst.append(value)
                    elif action == "set_true":
                        state[field_name] = True
                    elif action == "increment":
                        state[field_name] = state.get(field_name, 0) + 1

        # Built-in mutations for known AKW blueprint tool patterns
        if tool_name == "web_search":
            results = []
            try:
                import json as _json

                parsed = _json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                if isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = parsed.get("results", [parsed])
            except Exception:
                if tool_output:
                    results = [{"snippet": str(tool_output)[:200]}]
            state.setdefault("sources", []).extend(results)
            state["sources_count"] = len(state["sources"])

        elif tool_name in ("score_relevance", "rate_paper"):
            state.setdefault("filtered_papers", []).append({"tool": tool_name, "input": tool_input})
            state["papers_scored"] = state.get("papers_scored", 0) + 1

        elif tool_name == "extract_biomarkers":
            biomarkers = tool_input.get("biomarkers", [])
            state.setdefault("extracted_biomarkers", []).extend(biomarkers)
            state["papers_processed"] = state.get("papers_processed", 0) + 1

        elif tool_name == "record_outcome":
            state.setdefault("biomarker_outcomes", []).append(
                {
                    "biomarker": tool_input.get("biomarker", ""),
                    "outcome": tool_input.get("outcome", ""),
                    "direction": tool_input.get("direction", ""),
                    "domain": tool_input.get("domain", ""),
                    "pmid": tool_input.get("pmid", ""),
                    "confidence": tool_input.get("confidence", 0.0),
                }
            )

        elif tool_name == "flag_conflict":
            state.setdefault("flagged_conflicts", []).append(
                {
                    "biomarker": tool_input.get("biomarker", ""),
                    "conflict_description": tool_input.get("conflict_description", ""),
                    "papers": tool_input.get("papers", []),
                }
            )

        elif tool_name in ("read_memory", "read_all_memories"):
            state["memory_checked"] = True

        elif tool_name in ("fetch_github_issues",):
            state["issues_fetched"] = True
            issues = []
            try:
                import json as _json

                parsed = _json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                issues = parsed if isinstance(parsed, list) else parsed.get("issues", [])
            except Exception:
                pass
            state.setdefault("raw_issues", []).extend(issues)
            state["issues_total"] = len(state["raw_issues"])

        elif tool_name == "categorise_issues":
            state["issues_categorised"] = True

        elif tool_name == "finish":
            state["finish_called"] = True

        elif tool_name in ("Bash", "Write", "Edit", "Read"):
            # Coding tool — track consecutive build failures
            if tool_name == "Bash":
                cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
                is_build = any(
                    kw in cmd for kw in ("mise run build", "npm run build", "pytest", "cargo build")
                )
                is_error = "error" in tool_output.lower() if isinstance(tool_output, str) else False
                if is_build and is_error:
                    state["consecutive_build_failures"] = (
                        state.get("consecutive_build_failures", 0) + 1
                    )
                elif is_build:
                    state["consecutive_build_failures"] = 0

    except Exception as exc:
        log("DEBUG", f"blueprint_tracker: state mutation failed (fail-open): {exc}")


def _advance_phase(
    phases: list[dict],
    current_phase: str,
    state: dict[str, Any],
    completed_phases: set[str],
    parameters: dict[str, Any],
) -> str:
    """Check completion signal for current phase; advance if met. Returns new current phase."""
    try:
        phase_data = next((p for p in phases if p["name"] == current_phase), None)
        if not phase_data:
            return current_phase
        signal = phase_data.get("completion_signal", "")
        if not signal or signal == "true":
            # No signal = immediate completion on any tool use in this phase
            if current_phase not in completed_phases:
                completed_phases.add(current_phase)
                idx = next((i for i, p in enumerate(phases) if p["name"] == current_phase), -1)
                if idx >= 0 and idx + 1 < len(phases):
                    return phases[idx + 1]["name"]
            return current_phase

        # Evaluate the completion_signal expression against state + parameters.
        # Two forms supported:
        #   "sources_count >= {min_sources}"  — {placeholder} substituted first
        #   "memory_checked == True"          — state vars available as identifiers
        try:
            expr = signal
            # Substitute {placeholder} values from parameters
            for k, v in parameters.items():
                expr = expr.replace(f"{{{k}}}", str(v))
            if expr.strip().lower() in ("true", "1"):
                result = True
            else:
                # Eval with state variables + True/False available as names
                eval_ns = {"__builtins__": {}, "True": True, "False": False}
                eval_ns.update(state)
                result = bool(eval(expr, eval_ns))  # noqa: S307
        except Exception:
            result = False

        if result and current_phase not in completed_phases:
            completed_phases.add(current_phase)
            idx = next((i for i, p in enumerate(phases) if p["name"] == current_phase), -1)
            if idx >= 0 and idx + 1 < len(phases):
                return phases[idx + 1]["name"]
    except Exception as exc:
        log("DEBUG", f"blueprint_tracker: phase advance failed (fail-open): {exc}")
    return current_phase


class BlueprintTracker:
    """Per-task tracker: holds state dict, phase cursor, and PatternEvaluator.

    Created in pipeline.py when a blueprint is loaded. Passed via closure to
    the PostToolUse hook and the between_turns pattern hook.
    """

    def __init__(self, blueprint, task_id: str, task_type: str) -> None:
        self.blueprint = blueprint
        self.task_id = task_id
        self.task_type = task_type
        self.state: dict[str, Any] = {}
        self.current_phase: str = blueprint.phases[0]["name"] if blueprint.phases else ""
        self.completed_phases: set[str] = set()
        self._evaluator = PatternEvaluator()

    def on_tool_use(self, tool_name: str, tool_input: dict, tool_output: str) -> None:
        """Called from PostToolUse hook. Updates state and advances phase. Fail-open."""
        mutations = getattr(self.blueprint, "state_mutations", None) or {}
        _apply_state_mutations(tool_name, tool_input, tool_output, self.state, mutations)
        if self.blueprint.phases and self.current_phase:
            self.current_phase = _advance_phase(
                self.blueprint.phases,
                self.current_phase,
                self.state,
                self.completed_phases,
                self.blueprint.parameters,
            )
            log(
                "DEBUG",
                f"blueprint_tracker: phase={self.current_phase} state_keys={list(self.state.keys())}",
            )

    def evaluate_and_inject(self, ctx: dict) -> list[str]:
        """Called from between_turns_hooks. Runs PatternEvaluator; returns HITL XML messages."""
        if not self.blueprint.hitl_conditions:
            return []
        try:
            fired: list[FiredCondition] = self._evaluator.evaluate(
                self.blueprint.hitl_conditions,
                self.state,
                parameters=self.blueprint.parameters,
                completed_phases=self.completed_phases,
            )
            messages = []
            for condition in fired:
                log("HITL", f"Pattern fired: {condition.trigger} — {condition.question[:80]}")
                # Format as XML user message (same mechanism as nudges)
                options_xml = "".join(
                    f'<option id="{o["id"]}">{o["label"]}</option>' for o in condition.options
                )
                msg = (
                    f"<hitl_request trigger='{condition.trigger}'>"
                    f"<question>{condition.question}</question>"
                    f"<options>{options_xml}</options>"
                    f"</hitl_request>"
                )
                messages.append(msg)
            return messages
        except Exception as exc:
            log("DEBUG", f"blueprint_tracker: pattern evaluation failed (fail-open): {exc}")
            return []


def load_blueprint_for_task(task_type: str, blueprints_dir: str | None = None) -> Any:
    """Load blueprint for task_type from FilesystemRegistryService. Returns None if not found."""
    try:
        from registry.filesystem import FilesystemRegistryService

        dir_path = blueprints_dir or os.environ.get("BLUEPRINTS_DIR", "/blueprints")
        registry = FilesystemRegistryService(blueprints_dir=dir_path)
        return registry.find_blueprint_for_task(task_type)
    except Exception as exc:
        log("DEBUG", f"blueprint_tracker: could not load blueprint for {task_type!r}: {exc}")
        return None
