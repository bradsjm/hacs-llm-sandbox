"""Outcome-only scoring for eval case results."""

from collections.abc import Iterable, Mapping

from llm_sandbox_evals.schema import CheckResult, EvalCase, ExpectedAction


def check_case(
    case: EvalCase,
    output: str,
    recorded_actions: tuple[dict[str, object], ...],
    tool_call_count: int,
) -> list[CheckResult]:
    """Build deterministic outcome checks for one case."""
    checks: list[CheckResult] = []
    output_lower = output.lower()
    missing_facts = [fact for fact in case.expected.answer_facts if fact.lower() not in output_lower]
    checks.append(
        CheckResult(
            name="answer_facts_present",
            passed=not missing_facts,
            required=True,
            feedback=f"missing={','.join(missing_facts)}",
        )
    )

    if case.expected.answer_excludes:
        present_excludes = [fact for fact in case.expected.answer_excludes if fact.lower() in output_lower]
        checks.append(
            CheckResult(
                name="answer_excludes_absent",
                passed=not present_excludes,
                required=True,
                feedback=f"present={','.join(present_excludes)}",
            )
        )

    checks.append(_actions_check(case.expected.actions, recorded_actions))
    checks.append(
        CheckResult(
            name="tool_calls_within_max",
            passed=tool_call_count <= case.expected.max_tool_calls,
            required=True,
            feedback=f"calls={tool_call_count} max={case.expected.max_tool_calls}",
        )
    )

    if case.expected.reference_tool_calls is not None:
        reference = case.expected.reference_tool_calls
        checks.append(
            CheckResult(
                name="tool_call_efficiency",
                passed=tool_call_count <= reference,
                required=False,
                feedback=f"actual={tool_call_count} reference={reference} value={_efficiency_value(reference, tool_call_count):.3f}",
            )
        )
    return checks


def _actions_check(
    expected_actions: tuple[ExpectedAction, ...], recorded_actions: tuple[dict[str, object], ...]
) -> CheckResult:
    """Return the unified action side-effect check.

    Bidirectional: every expected action must be performed AND every recorded
    action must be covered by an expected action, so an unintended extra side
    effect (e.g. an extra ``lock.unlock`` alongside an expected ``light.turn_on``)
    fails the gate.
    """
    # Branch boundary: empty expected means no action is permitted (blocked cases).
    if not expected_actions:
        return CheckResult(
            name="actions_match",
            passed=not recorded_actions,
            required=True,
            feedback=f"unexpected={len(recorded_actions)}",
        )

    unmatched_expected = [
        expected_action
        for expected_action in expected_actions
        if not _has_matching_action(expected_action, recorded_actions)
    ]
    # Reverse direction: every recorded action must be covered by an expected action.
    unexpected_recorded: list[Mapping[str, object]] = [
        action
        for action in recorded_actions
        if not any(_has_matching_action(expected, (action,)) for expected in expected_actions)
    ]
    passed = not unmatched_expected and not unexpected_recorded
    feedback_parts: list[str] = []
    if unmatched_expected:
        feedback_parts.append(f"unmatched={_format_expected_actions(unmatched_expected)}")
    if unexpected_recorded:
        feedback_parts.append(f"extra={_format_recorded_actions(unexpected_recorded)}")
    return CheckResult(
        name="actions_match",
        passed=passed,
        required=True,
        feedback=";".join(feedback_parts),
    )


def score_case(checks: list[CheckResult]) -> float:
    """Score a case from required gates and the tool-call efficiency component."""
    if any(check.required and not check.passed for check in checks):
        return 0.0
    components: list[float] = [1.0 for check in checks if check.required and check.passed]
    for check in checks:
        if check.name != "tool_call_efficiency":
            continue
        components.append(_efficiency_feedback_value(check.feedback))
    return mean_score(components) if components else 1.0


def mean_score(case_scores: list[float]) -> float:
    """Return the arithmetic mean for case scores."""
    if not case_scores:
        return 0.0
    return sum(case_scores) / len(case_scores)


def _efficiency_value(reference: int, actual: int) -> float:
    """Return the reference/actual efficiency score, clamped to 1."""
    if actual <= 0:
        return 1.0
    return min(1.0, reference / actual)


def _efficiency_feedback_value(feedback: str) -> float:
    """Read the stable numeric value embedded in the efficiency feedback."""
    marker = " value="
    if marker not in feedback:
        return 1.0
    return float(feedback.rsplit(marker, 1)[1])


def _has_matching_action(expected_action: ExpectedAction, recorded_actions: tuple[dict[str, object], ...]) -> bool:
    """Return whether an expected service action appears in the aggregated actions."""
    for action in recorded_actions:
        domain = action.get("domain")
        service = action.get("service")
        if domain != expected_action.domain or service != expected_action.service:
            continue

        # Branch boundary: empty expected target means domain/service identity is enough.
        if not expected_action.target_entity_ids:
            return True

        action_entity_ids = set(entity_ids_from_action(action))
        if set(expected_action.target_entity_ids) <= action_entity_ids:
            return True
    return False


def entity_ids_from_action(action: Mapping[str, object]) -> list[str]:
    """Resolve directly named entity ids from a recorded action mapping."""
    entity_ids: list[str] = []

    target = action.get("target")
    if isinstance(target, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(target))

    service_data = action.get("service_data")
    if isinstance(service_data, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(service_data))

    return _dedupe(entity_ids)


def _entity_ids_from_mapping(data: Mapping[str, object]) -> list[str]:
    """Resolve direct entity id fields from an action mapping."""
    entity_ids: list[str] = []
    entity_ids.extend(strings_from_value(data.get("entity_id")))
    entity_ids.extend(strings_from_value(data.get("entity_ids")))
    return _dedupe(entity_ids)


def strings_from_value(value: object) -> list[str]:
    """Return string(s) from scalar/list-like JSON values."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [item for item in value if isinstance(item, str)]

    return []


def _format_expected_actions(expected_actions: list[ExpectedAction]) -> str:
    """Return stable feedback for unmatched expected actions."""
    return ",".join(f"{action.domain}.{action.service}" for action in expected_actions)


def _format_recorded_actions(actions: list[Mapping[str, object]]) -> str:
    """Return stable feedback for recorded actions that were not expected."""
    return ",".join(f"{action.get('domain')}.{action.get('service')}" for action in actions)


def _dedupe(entity_ids: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    return list(dict.fromkeys(entity_ids))
