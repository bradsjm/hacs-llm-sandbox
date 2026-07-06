"""Deterministic outcome and efficiency scoring for eval case outcomes."""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from custom_components.llm_sandbox.const import TOOL_GET_HISTORY, TOOL_GET_LOGBOOK, TOOL_GET_STATISTICS
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import CheckResult, EvalCase, ExpectedAction, StepTrace

_RECORDER_TOOL_NAMES = {TOOL_GET_HISTORY, TOOL_GET_STATISTICS, TOOL_GET_LOGBOOK}


@dataclass(frozen=True, slots=True)
class TraceFacts:
    """Flattened evidence derived from all tool turns in one case trace."""

    tool_names: tuple[str, ...]
    evidence_text: str
    evidence_entity_ids: frozenset[str]
    result_path_values: tuple[object, ...]
    recorder_windows: tuple[tuple[datetime, datetime], ...]
    error_keys: frozenset[str]


def check_case(
    case: EvalCase,
    final_answer: str,
    recorded_actions: tuple[dict[str, object], ...],
    execute_statuses: set[str],
    snapshot: HomeSnapshot,
    steps: tuple[StepTrace, ...],
) -> list[CheckResult]:
    """Build required, outcome-based checks for one case across all turns."""
    checks: list[CheckResult] = []
    answer_text = final_answer.lower()
    facts = _trace_facts(steps, recorded_actions, snapshot)

    checks.append(
        CheckResult(
            name="tool_used",
            passed=case.expected.tool_name in facts.tool_names,
            required=True,
            feedback=f"observed={','.join(facts.tool_names)} expected={case.expected.tool_name}",
        )
    )

    if case.expected.required_tool_names:
        missing_tools = sorted(set(case.expected.required_tool_names) - set(facts.tool_names))
        checks.append(
            CheckResult(
                name="required_tools_used",
                passed=not missing_tools,
                required=True,
                feedback=f"missing={','.join(missing_tools)}",
            )
        )

    if case.expected.required_tool_sequence:
        checks.append(
            CheckResult(
                name="required_tool_sequence",
                passed=_contains_ordered_subsequence(facts.tool_names, case.expected.required_tool_sequence),
                required=True,
                feedback=(
                    f"observed={','.join(facts.tool_names)} expected={','.join(case.expected.required_tool_sequence)}"
                ),
            )
        )

    if case.expected.required_error_keys:
        missing_errors = sorted(set(case.expected.required_error_keys) - set(facts.error_keys))
        checks.append(
            CheckResult(
                name="required_error_keys",
                passed=not missing_errors,
                required=True,
                feedback=f"missing={','.join(missing_errors)} observed={','.join(sorted(facts.error_keys))}",
            )
        )

    if case.expected.required_result_paths:
        missing_paths = [
            path
            for path in case.expected.required_result_paths
            if not _has_result_path(facts.result_path_values, path)
        ]
        checks.append(
            CheckResult(
                name="required_result_paths",
                passed=not missing_paths,
                required=True,
                feedback=f"missing={','.join(missing_paths)}",
            )
        )

    if case.expected.required_tool_arg_values:
        has_required_arg_values = _has_tool_arg_values(steps, case.expected.required_tool_arg_values)
        checks.append(
            CheckResult(
                name="required_tool_arg_values",
                passed=has_required_arg_values,
                required=True,
                feedback=f"required={','.join(path for path, _expected_value in case.expected.required_tool_arg_values)}",
            )
        )

    if case.expected.recorder_window is not None:
        checks.append(_recorder_window_check(case, facts))

    if case.expected.max_tool_turns is not None:
        checks.append(
            CheckResult(
                name="tool_turns_within_max",
                passed=len(steps) <= case.expected.max_tool_turns,
                required=True,
                feedback=f"turns={len(steps)} max={case.expected.max_tool_turns}",
            )
        )

    if case.expected.max_tool_calls is not None:
        tool_call_count = sum(len(step.tool_calls) for step in steps)
        checks.append(
            CheckResult(
                name="tool_calls_within_max",
                passed=tool_call_count <= case.expected.max_tool_calls,
                required=True,
                feedback=f"calls={tool_call_count} max={case.expected.max_tool_calls}",
            )
        )

    if case.expected.max_successful_actions is not None:
        successful_actions = [action for action in recorded_actions if action.get("status") == "ok"]
        checks.append(
            CheckResult(
                name="successful_actions_within_max",
                passed=len(successful_actions) <= case.expected.max_successful_actions,
                required=True,
                feedback=f"actions={len(successful_actions)} max={case.expected.max_successful_actions}",
            )
        )

    # Branch boundary: execute-code status expectations are aggregated across all execute turns.
    if case.expected.execution_status != "na":
        checks.append(
            CheckResult(
                name="execution_status",
                passed=case.expected.execution_status in execute_statuses and "setup_error" not in execute_statuses,
                required=True,
                feedback=f"observed={','.join(sorted(execute_statuses))} expected={case.expected.execution_status}",
            )
        )

    # Disabled-action cases must not EXECUTE a service call. A blocked/errored
    # attempt (status != "ok") is the gate working correctly, not a violation.
    if not case.actions_enabled:
        executed_actions = [action for action in recorded_actions if action.get("status") == "ok"]
        checks.append(
            CheckResult(
                name="no_action_when_disabled",
                passed=not executed_actions,
                required=True,
                feedback=f"executed={len(executed_actions)}",
            )
        )

    has_expected_actions = bool(case.expected.actions)
    # Branch boundary: action cases are verified by recorded side effects, not by echoed prose.
    if case.expected.output_contains_entities and not has_expected_actions:
        missing_entities = sorted(
            e for e in case.expected.output_contains_entities if not _entity_referenced(e, answer_text, snapshot)
        )
        checks.append(
            CheckResult(
                name="output_contains",
                passed=not missing_entities,
                required=True,
                feedback=f"missing={','.join(missing_entities)}",
            )
        )

    if case.expected.output_excludes_entities and not has_expected_actions:
        present_entities = sorted(
            e for e in case.expected.output_excludes_entities if _entity_mentioned(e, answer_text, snapshot)
        )
        checks.append(
            CheckResult(
                name="output_excludes",
                passed=not present_entities,
                required=True,
                feedback=f"present={','.join(present_entities)}",
            )
        )

    if case.expected.evidence_contains_entities:
        missing_evidence = sorted(
            entity_id
            for entity_id in case.expected.evidence_contains_entities
            if not _entity_in_evidence(entity_id, facts, snapshot)
        )
        checks.append(
            CheckResult(
                name="evidence_contains",
                passed=not missing_evidence,
                required=True,
                feedback=f"missing={','.join(missing_evidence)}",
            )
        )

    if case.expected.evidence_excludes_entities:
        present_evidence = sorted(
            entity_id
            for entity_id in case.expected.evidence_excludes_entities
            if _entity_in_evidence(entity_id, facts, snapshot)
        )
        checks.append(
            CheckResult(
                name="evidence_excludes",
                passed=not present_evidence,
                required=True,
                feedback=f"present={','.join(present_evidence)}",
            )
        )

    if case.expected.actions:
        unmatched_actions = [
            expected_action
            for expected_action in case.expected.actions
            if not _has_matching_action(expected_action, recorded_actions, snapshot)
        ]
        checks.append(
            CheckResult(
                name="actions_match",
                passed=not unmatched_actions,
                required=True,
                feedback=f"unmatched={_format_expected_actions(unmatched_actions)}",
            )
        )

    return checks


def _trace_facts(
    steps: tuple[StepTrace, ...],
    recorded_actions: tuple[dict[str, object], ...],
    snapshot: HomeSnapshot,
) -> TraceFacts:
    """Flatten all tool turns into facts used by deterministic scoring checks."""
    tool_names: list[str] = []
    evidence_parts: list[str] = []
    evidence_entity_ids: set[str] = set()
    result_path_values: list[object] = []
    recorder_windows: list[tuple[datetime, datetime]] = []
    error_keys: set[str] = set()

    for step in steps:
        for index, call in enumerate(step.tool_calls):
            tool_names.append(call.tool_name)
            evidence_entity_ids.update(_ids_from_tool_args(call.tool_args))
            result = step.tool_results[index] if index < len(step.tool_results) else None
            if result is not None:
                result_path_values.append(result)
                evidence_parts.append(json.dumps(result, sort_keys=True, default=str))
                result_ids = _ids_from_result(result)
                evidence_entity_ids.update(result_ids)
                error_key = _error_key(result)
                if error_key is not None:
                    error_keys.add(error_key)
            if call.tool_name in _RECORDER_TOOL_NAMES:
                window = _window_from_call_or_result(call.tool_args, result, snapshot)
                if window is not None:
                    recorder_windows.append(window)

    for action in recorded_actions:
        result_path_values.append(action)
        evidence_parts.append(json.dumps(action, sort_keys=True, default=str))
        evidence_entity_ids.update(entity_ids_from_action(action, snapshot))
        error_key = _error_key(action)
        if error_key is not None:
            error_keys.add(error_key)

    return TraceFacts(
        tool_names=tuple(tool_names),
        evidence_text="\n".join(evidence_parts).lower(),
        evidence_entity_ids=frozenset(evidence_entity_ids),
        result_path_values=tuple(result_path_values),
        recorder_windows=tuple(recorder_windows),
        error_keys=frozenset(error_keys),
    )


def _has_result_path(values: tuple[object, ...], path: str) -> bool:
    """Return whether any collected result/action contains a dotted path."""
    parts = tuple(part for part in path.split(".") if part)
    return bool(parts) and any(_value_has_path(value, parts) for value in values)


def _has_tool_arg_values(steps: tuple[StepTrace, ...], required_values: tuple[tuple[str, object], ...]) -> bool:
    """Return whether one observed tool call has every required dotted arg value."""
    required_parts = tuple(
        (tuple(part for part in path.split(".") if part), expected) for path, expected in required_values
    )
    if not required_parts or any(not parts for parts, _expected in required_parts):
        return False
    for step in steps:
        for call in step.tool_calls:
            if all(_value_at_path(call.tool_args, parts) == (True, expected) for parts, expected in required_parts):
                return True
    return False


def _value_at_path(value: object, parts: tuple[str, ...]) -> tuple[bool, object | None]:
    """Return a mapping value at a dotted path without inventing defaults for missing keys."""
    if not parts:
        return (True, value)
    head, *tail = parts
    if not isinstance(value, Mapping) or head not in value:
        return (False, None)
    return _value_at_path(value[head], tuple(tail))


def _value_has_path(value: object, parts: tuple[str, ...]) -> bool:
    """Return whether value contains the path, treating sequences as any-match containers."""
    if not parts:
        return value is not None
    head, *tail = parts
    remaining = tuple(tail)
    if isinstance(value, Mapping):
        return head in value and _value_has_path(value[head], remaining)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return any(_value_has_path(item, parts) for item in value)
    return False


def _ids_from_tool_args(tool_args: Mapping[str, object]) -> set[str]:
    """Return explicit entity/statistic ids from native tool arguments."""
    ids = set(strings_from_value(tool_args.get("entity_ids")))
    ids.update(strings_from_value(tool_args.get("statistic_ids")))
    return ids


def _ids_from_result(result: Mapping[str, object]) -> set[str]:
    """Return entity/statistic ids surfaced by public tool result envelopes."""
    ids: set[str] = set()
    entities = result.get("entities")
    if isinstance(entities, Mapping):
        ids.update(str(entity_id) for entity_id in entities)
    statistics = result.get("statistics")
    if isinstance(statistics, Mapping):
        ids.update(str(statistic_id) for statistic_id in statistics)
    return ids


def _error_key(result: Mapping[str, object]) -> str | None:
    """Return the stable error key from a tool result or action record."""
    error = result.get("error")
    if not isinstance(error, Mapping):
        return None
    key = error.get("key")
    return key if isinstance(key, str) else None


def _window_from_call_or_result(
    tool_args: Mapping[str, object],
    result: Mapping[str, object] | None,
    snapshot: HomeSnapshot,
) -> tuple[datetime, datetime] | None:
    """Return a recorder query window from explicit args, relative hours, or result metadata."""
    args_window = _window_from_values(tool_args.get("start"), tool_args.get("end"))
    if args_window is not None:
        return args_window

    hours = tool_args.get("hours")
    if isinstance(hours, int | float):
        end = _parse_datetime(snapshot.created_at)
        if end is not None:
            return (end - timedelta(hours=float(hours)), end)

    if result is None:
        return None
    result_window = result.get("window")
    if not isinstance(result_window, Mapping):
        return None
    return _window_from_values(result_window.get("start"), result_window.get("end"))


def _window_from_values(start_value: object, end_value: object) -> tuple[datetime, datetime] | None:
    """Return a parsed time window when both values are valid ISO datetimes."""
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    start = _parse_datetime(start_value)
    end = _parse_datetime(end_value)
    if start is None or end is None:
        return None
    return (start, end)


def _parse_datetime(value: str) -> datetime | None:
    """Parse an ISO timestamp for deterministic recorder-window checks."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _recorder_window_check(case: EvalCase, facts: TraceFacts) -> CheckResult:
    """Return whether any recorder tool call/result covers the expected time window."""
    expected = case.expected.recorder_window
    if expected is None:
        return CheckResult(name="recorder_window", passed=True, required=True, feedback="")
    expected_window = _window_from_values(expected[0], expected[1])
    if expected_window is None:
        return CheckResult(name="recorder_window", passed=False, required=True, feedback="invalid expected window")
    expected_start, expected_end = expected_window
    covered = any(start <= expected_start and end >= expected_end for start, end in facts.recorder_windows)
    observed = ";".join(f"{start.isoformat()}..{end.isoformat()}" for start, end in facts.recorder_windows)
    return CheckResult(
        name="recorder_window",
        passed=covered,
        required=True,
        feedback=f"observed={observed} expected={expected[0]}..{expected[1]}",
    )


def _contains_ordered_subsequence(observed: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    """Return whether expected tool names appear in order within observed tool names."""
    position = 0
    for tool_name in observed:
        if position < len(expected) and tool_name == expected[position]:
            position += 1
    return position == len(expected)


def _entity_in_evidence(entity_id: str, facts: TraceFacts, snapshot: HomeSnapshot) -> bool:
    """Return whether a tool call/result/action references an entity by id or name."""
    return entity_id in facts.evidence_entity_ids or _entity_mentioned(entity_id, facts.evidence_text, snapshot)


def score_case(checks: list[CheckResult], turns: int, par: int, k: float, floor: float) -> float:
    """Score a case from required gates and the turn-efficiency formula."""
    if any(check.required and not check.passed for check in checks):
        return 0.0
    if turns <= par:
        return 1.0
    return max(floor, 1 - k * (turns - par))


def mean_score(case_scores: list[float]) -> float:
    """Return the arithmetic mean for case scores."""
    if not case_scores:
        return 0.0
    return sum(case_scores) / len(case_scores)


def _has_matching_action(
    expected_action: ExpectedAction, recorded_actions: tuple[dict[str, object], ...], snapshot: HomeSnapshot
) -> bool:
    """Return whether an expected service action appears in the aggregated actions."""
    for action in recorded_actions:
        domain = action.get("domain")
        service = action.get("service")
        if domain != expected_action.domain or service != expected_action.service:
            continue

        # Branch boundary: empty expected target means domain/service identity is enough.
        if not expected_action.target_entity_ids:
            return True

        action_entity_ids = set(entity_ids_from_action(action, snapshot))
        if set(expected_action.target_entity_ids) <= action_entity_ids:
            return True
    return False


def entity_ids_from_action(action: Mapping[str, object], snapshot: HomeSnapshot) -> list[str]:
    """Resolve entity ids named directly or via HA target selectors in a recorded action."""
    entity_ids: list[str] = []

    target = action.get("target")
    if isinstance(target, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(target, snapshot))

    service_data = action.get("service_data")
    if isinstance(service_data, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(service_data, snapshot))

    return _dedupe(entity_ids)


def _entity_ids_from_mapping(data: Mapping[str, object], snapshot: HomeSnapshot) -> list[str]:
    """Resolve direct entity ids plus HA target selectors from an action mapping."""
    entity_ids: list[str] = []
    entity_ids.extend(strings_from_value(data.get("entity_id")))
    entity_ids.extend(strings_from_value(data.get("entity_ids")))

    for device_id in strings_from_value(data.get("device_id")):
        entity_ids.extend(snapshot.indexes.entity_ids_by_device_id.get(device_id, ()))

    for area_id in strings_from_value(data.get("area_id")):
        entity_ids.extend(snapshot.indexes.entity_ids_by_area_id.get(area_id, ()))

    for label_id in [*strings_from_value(data.get("label_id")), *strings_from_value(data.get("label_ids"))]:
        entity_ids.extend(snapshot.indexes.entity_ids_by_label.get(label_id, ()))

    for floor_id in [*strings_from_value(data.get("floor_id")), *strings_from_value(data.get("floor_ids"))]:
        # Branch boundary: floor targets resolve through areas; no direct floor-to-entity index exists.
        for area_id in snapshot.indexes.area_ids_by_floor_id.get(floor_id, ()):
            entity_ids.extend(snapshot.indexes.entity_ids_by_area_id.get(area_id, ()))

    return _dedupe(entity_ids)


def strings_from_value(value: object) -> list[str]:
    """Return string(s) from scalar/list-like JSON values."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [item for item in value if isinstance(item, str)]

    return []


def _entity_referenced(entity_id: str, output_text: str, snapshot: HomeSnapshot) -> bool:
    """Return True when a read answer reflects an entity by id, friendly name, or state value."""
    if _entity_mentioned(entity_id, output_text, snapshot):
        return True
    state = snapshot.states.get(entity_id)
    if state is None:
        return False
    value = str(state.state).strip().lower()
    return bool(value) and re.search(rf"\b{re.escape(value)}\b", output_text) is not None


def _entity_mentioned(entity_id: str, output_text: str, snapshot: HomeSnapshot) -> bool:
    """Return True when the output names an entity by id or friendly name."""
    if entity_id.lower() in output_text:
        return True
    state = snapshot.states.get(entity_id)
    if state is None:
        return False
    friendly_name = state.attributes.get("friendly_name")
    name = friendly_name if isinstance(friendly_name, str) else state.name
    if not isinstance(name, str) or not name:
        return False
    return name.lower() in output_text


def _format_expected_actions(expected_actions: list[ExpectedAction]) -> str:
    """Return stable feedback for unmatched expected actions."""
    return ",".join(f"{action.domain}.{action.service}" for action in expected_actions)


def _dedupe(entity_ids: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    return list(dict.fromkeys(entity_ids))
