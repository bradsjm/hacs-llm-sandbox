"""Deterministic scoring for eval case outcomes."""

import re
from collections.abc import Iterable, Mapping

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import CheckResult, EvalCase, ExpectedAction, ToolOutcome

RECORDER_TOOLS: frozenset[str] = frozenset({TOOL_GET_HISTORY, TOOL_GET_STATISTICS, TOOL_GET_LOGBOOK})


def check_case(
    case: EvalCase,
    tool_call: dict[str, object] | None,
    outcome: ToolOutcome,
    snapshot: HomeSnapshot,
) -> list[CheckResult]:
    """Build deterministic required and optional checks for one case."""
    checks: list[CheckResult] = []
    tool_name, tool_args = _tool_call_parts(tool_call)
    valid_tool_call = tool_name is not None and tool_args is not None
    checks.append(
        CheckResult(
            name="valid_tool_call",
            passed=valid_tool_call,
            required=True,
            feedback="ok" if valid_tool_call else "missing or malformed tool call",
        )
    )

    checks.append(
        CheckResult(
            name="correct_tool_name",
            passed=tool_name == case.expected.tool_name,
            required=True,
            feedback=f"actual={tool_name!s} expected={case.expected.tool_name}",
        )
    )

    # Execute-code cases expose the nested execution.status contract.
    if case.expected.tool_name == TOOL_EXECUTE_HOME_CODE and case.expected.execution_status != "na":
        observed_status = _execution_status(outcome)
        checks.append(
            CheckResult(
                name="execution_status",
                passed=observed_status == case.expected.execution_status and observed_status != "setup_error",
                required=True,
                feedback=f"observed={observed_status!s} expected={case.expected.execution_status}",
            )
        )

    # Disabled-action cases must not EXECUTE a service call. A blocked/errored
    # attempt (status != "ok") is the gate working correctly, not a violation.
    if not case.actions_enabled:
        executed_actions = [action for action in _recorded_actions(outcome) if action.get("status") == "ok"]
        checks.append(
            CheckResult(
                name="no_action_when_disabled",
                passed=not executed_actions,
                required=True,
                feedback=f"executed={len(executed_actions)}",
            )
        )

    # Visibility-sensitive cases must not reference filtered-out entity ids.
    if case.expected.visible_only:
        referenced_entity_ids = _referenced_entity_ids(tool_call, outcome, snapshot)
        invisible_entity_ids = sorted(
            {entity_id for entity_id in referenced_entity_ids if entity_id not in snapshot.states}
        )
        checks.append(
            CheckResult(
                name="no_invisible_target",
                passed=not invisible_entity_ids,
                required=True,
                feedback=f"invisible={','.join(invisible_entity_ids)}",
            )
        )

    # Recorder tools pass this gate when the emulated tool produced a result envelope.
    if case.expected.tool_name in RECORDER_TOOLS:
        tool_ran = outcome.ok and outcome.result is not None and outcome.result.get("status") == "ok"
        checks.append(
            CheckResult(
                name="tool_ran",
                passed=tool_ran,
                required=True,
                feedback="ok" if tool_ran else _recorder_tool_feedback(outcome),
            )
        )

    output_text = _output_text(outcome)
    has_expected_actions = bool(case.expected.actions)
    is_recorder = case.expected.tool_name in RECORDER_TOOLS

    # Content checks derive the expected answer from actual data. Action cases are
    # verified through recorded_actions (below), so entity-mention checks are
    # skipped for them: a correct model performs the action without necessarily
    # echoing the entity id in prose.
    if case.expected.output_contains_entities and not has_expected_actions:
        if is_recorder:
            # Recorder correctness = the model requested the right entity/statistic id.
            requested = _recorder_requested_ids(tool_args)
            missing_entities = sorted(e for e in case.expected.output_contains_entities if e not in requested)
        else:
            # Read correctness = the output reflects the entity by id, friendly name, or state value.
            missing_entities = sorted(
                e for e in case.expected.output_contains_entities if not _entity_referenced(e, output_text, snapshot)
            )
        checks.append(
            CheckResult(
                name="output_contains",
                passed=not missing_entities,
                required=False,
                feedback=f"missing={','.join(missing_entities)}",
            )
        )

    if case.expected.output_excludes_entities and not has_expected_actions:
        present_entities = sorted(
            e for e in case.expected.output_excludes_entities if _entity_mentioned(e, output_text, snapshot)
        )
        checks.append(
            CheckResult(
                name="output_excludes",
                passed=not present_entities,
                required=False,
                feedback=f"present={','.join(present_entities)}",
            )
        )

    if case.expected.actions:
        unmatched_actions = [
            expected_action
            for expected_action in case.expected.actions
            if not _has_matching_action(expected_action, outcome, snapshot)
        ]
        checks.append(
            CheckResult(
                name="actions_match",
                passed=not unmatched_actions,
                required=False,
                feedback=f"unmatched={_format_expected_actions(unmatched_actions)}",
            )
        )

    if case.expected.recorder_window is not None and is_recorder:
        has_window = _has_time_window(tool_args)
        checks.append(
            CheckResult(
                name="recorder_window",
                passed=has_window,
                required=False,
                feedback=f"windowed={has_window}",
            )
        )

    if output_text:
        checks.append(
            CheckResult(
                name="concise_output",
                passed=len(output_text) <= 2000,
                required=False,
                feedback=f"chars={len(output_text)}",
            )
        )

    return checks


def score_case(checks: list[CheckResult]) -> float:
    """Score a case from required gates and optional-check ratio."""
    if any(check.required and not check.passed for check in checks):
        return 0.0

    passed_optional = sum(1 for check in checks if not check.required and check.passed)
    total_optional = sum(1 for check in checks if not check.required)
    if total_optional == 0:
        return 1.0
    return passed_optional / total_optional


def mean_score(case_scores: list[float]) -> float:
    """Return the arithmetic mean for case scores."""
    if not case_scores:
        return 0.0
    return sum(case_scores) / len(case_scores)


def _tool_call_parts(tool_call: dict[str, object] | None) -> tuple[str | None, dict[str, object] | None]:
    if tool_call is None:
        return None, None

    tool_name = tool_call.get("tool_name")
    tool_args = tool_call.get("tool_args")
    if isinstance(tool_name, str) and isinstance(tool_args, dict):
        return tool_name, tool_args
    return None, None


def _execution_status(outcome: ToolOutcome) -> str | None:
    result = outcome.result
    if result is None or "execution" not in result:
        return None

    execution = result["execution"]
    if not isinstance(execution, dict) or "status" not in execution:
        return None

    status = execution["status"]
    if isinstance(status, str):
        return status
    return None


def _output_text(outcome: ToolOutcome) -> str:
    result = outcome.result
    if result is None:
        return ""

    parts: list[str] = []
    output = result.get("output")
    if isinstance(output, str):
        parts.append(output)
    elif output is not None:
        parts.append(str(output))

    printed = result.get("printed")
    if isinstance(printed, list):
        parts.extend(str(item) for item in printed)
    elif printed is not None:
        parts.append(str(printed))

    entities = result.get("entities")
    if isinstance(entities, dict):
        parts.extend(str(entity_id) for entity_id in entities)
        for rows in entities.values():
            if isinstance(rows, list):
                parts.extend(str(row) for row in rows)

    statistics = result.get("statistics")
    if isinstance(statistics, dict):
        parts.extend(str(statistic_id) for statistic_id in statistics)
        for rows in statistics.values():
            if isinstance(rows, list):
                parts.extend(str(row) for row in rows)

    entries = result.get("entries")
    if isinstance(entries, list):
        parts.extend(str(item) for item in entries)

    return "\n".join(parts).lower()


def _recorder_tool_feedback(outcome: ToolOutcome) -> str:
    """Return stable feedback for a recorder required gate failure."""
    if not outcome.ok:
        return outcome.error or "recorder did not run"

    if outcome.result is None:
        return "recorder did not return a result"

    return f"status={outcome.result.get('status')!s}"


def _recorded_actions(outcome: ToolOutcome) -> list[dict[str, object]]:
    return list(outcome.recorded_actions)


def _referenced_entity_ids(
    tool_call: dict[str, object] | None,
    outcome: ToolOutcome,
    snapshot: HomeSnapshot,
) -> list[str]:
    entity_ids: list[str] = []
    tool_name, tool_args = _tool_call_parts(tool_call)

    # Recorder arguments directly identify entities or statistic ids.
    if tool_name in RECORDER_TOOLS and tool_args is not None:
        entity_ids.extend(_strings_from_value(tool_args.get("entity_ids")))
        entity_ids.extend(_strings_from_value(tool_args.get("statistic_ids")))

    # Recorded actions carry target/service_data shapes from execute_home_code.
    for action in _recorded_actions(outcome):
        entity_ids.extend(_entity_ids_from_action(action, snapshot))

    return _dedupe(entity_ids)


def _has_matching_action(expected_action: ExpectedAction, outcome: ToolOutcome, snapshot: HomeSnapshot) -> bool:
    for action in _recorded_actions(outcome):
        domain = action.get("domain")
        service = action.get("service")
        if domain != expected_action.domain or service != expected_action.service:
            continue

        # Empty expected target means domain/service identity is enough.
        if not expected_action.target_entity_ids:
            return True

        action_entity_ids = set(_entity_ids_from_action(action, snapshot))
        if set(expected_action.target_entity_ids) <= action_entity_ids:
            return True
    return False


def _entity_ids_from_action(action: Mapping[str, object], snapshot: HomeSnapshot) -> list[str]:
    entity_ids: list[str] = []

    target = action.get("target")
    if isinstance(target, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(target, snapshot))

    service_data = action.get("service_data")
    if isinstance(service_data, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(service_data, snapshot))

    return _dedupe(entity_ids)


def _entity_ids_from_mapping(data: Mapping[str, object], snapshot: HomeSnapshot) -> list[str]:
    entity_ids: list[str] = []
    entity_ids.extend(_strings_from_value(data.get("entity_id")))
    entity_ids.extend(_strings_from_value(data.get("entity_ids")))

    for device_id in _strings_from_value(data.get("device_id")):
        entity_ids.extend(snapshot.indexes.entity_ids_by_device_id.get(device_id, ()))

    for area_id in _strings_from_value(data.get("area_id")):
        entity_ids.extend(snapshot.indexes.entity_ids_by_area_id.get(area_id, ()))

    return _dedupe(entity_ids)


def _strings_from_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [item for item in value if isinstance(item, str)]

    return []


def _entity_referenced(entity_id: str, output_text: str, snapshot: HomeSnapshot) -> bool:
    """Return True when a read answer reflects an entity by id, friendly name, or state value.

    Read-case answers are natural language, so a correct response need not echo
    the raw entity id. Matching the friendly name or the actual state value
    (derived from the fixture) credits a correct answer; the state value uses a
    word boundary so short values like ``on``/``off`` do not match substrings.
    """
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


def _recorder_requested_ids(tool_args: dict[str, object] | None) -> set[str]:
    """Return the entity/statistic ids the model requested from a recorder tool."""
    if tool_args is None:
        return set()
    ids = _strings_from_value(tool_args.get("entity_ids"))
    ids.extend(_strings_from_value(tool_args.get("statistic_ids")))
    return set(ids)


def _has_time_window(tool_args: dict[str, object] | None) -> bool:
    """Return True when the model scoped the recorder query to a bounded window.

    Accepts an explicit ISO ``start`` + ``end`` or a positive relative-window
    argument (``hours``/``days``/``minutes``) so a model that expresses
    "last 24 hours" is credited even if it uses a relative convention.
    """
    if not isinstance(tool_args, dict):
        return False
    start = tool_args.get("start")
    end = tool_args.get("end")
    if isinstance(start, str) and start and isinstance(end, str) and end:
        return True
    for key in ("hours", "days", "minutes"):
        value = tool_args.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _format_expected_actions(expected_actions: list[ExpectedAction]) -> str:
    return ",".join(f"{action.domain}.{action.service}" for action in expected_actions)


def _dedupe(entity_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(entity_ids))
