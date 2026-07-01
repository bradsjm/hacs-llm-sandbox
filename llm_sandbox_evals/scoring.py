"""Deterministic outcome and efficiency scoring for eval case outcomes."""

import re
from collections.abc import Iterable, Mapping

from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import CheckResult, EvalCase, ExpectedAction


def check_case(
    case: EvalCase,
    final_answer: str,
    recorded_actions: tuple[dict[str, object], ...],
    execute_statuses: set[str],
    referenced_entity_ids: set[str],
    snapshot: HomeSnapshot,
) -> list[CheckResult]:
    """Build required, outcome-based checks for one case across all turns."""
    checks: list[CheckResult] = []
    answer_text = final_answer.lower()

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

    # Visibility-sensitive cases must not reference filtered-out entity ids.
    if case.expected.visible_only:
        invisible_entity_ids = sorted(
            entity_id for entity_id in referenced_entity_ids if entity_id not in snapshot.states
        )
        checks.append(
            CheckResult(
                name="no_invisible_target",
                passed=not invisible_entity_ids,
                required=True,
                feedback=f"invisible={','.join(invisible_entity_ids)}",
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
    """Resolve direct entity ids plus simple area/device selectors from an action mapping."""
    entity_ids: list[str] = []
    entity_ids.extend(strings_from_value(data.get("entity_id")))
    entity_ids.extend(strings_from_value(data.get("entity_ids")))

    for device_id in strings_from_value(data.get("device_id")):
        entity_ids.extend(snapshot.indexes.entity_ids_by_device_id.get(device_id, ()))

    for area_id in strings_from_value(data.get("area_id")):
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
