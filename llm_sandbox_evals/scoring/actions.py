"""Exact successful action-ledger construction and scoring."""

from collections.abc import Mapping, Sequence
import json

from llm_sandbox_evals.schema import (
    ActionComparison,
    ActionLedger,
    ActionOutcomeReason,
    ActionResult,
    ObservedAction,
    RequiredAction,
)


def build_action_ledger(actions: Sequence[Mapping[str, object]]) -> ActionLedger:
    """Split copied action records into successful and rejected diagnostics."""
    copied = tuple(dict(action) for action in actions)
    return ActionLedger(
        successful=tuple(action for action in copied if action.get("status") != "error"),
        rejected=tuple(action for action in copied if action.get("status") == "error"),
    )


def score_actions(expected: tuple[RequiredAction, ...], ledger: ActionLedger) -> ActionResult:
    """Require the successful effect multiset to equal the authored effect multiset."""
    actual = tuple(_normalize_action(action) for action in ledger.successful)
    unmatched_expected = set(range(len(expected)))
    unmatched_actual = set(range(len(actual)))
    comparisons: list[ActionComparison | None] = [None] * len(expected)

    # Authored service data is matched first so an unspecified action cannot consume
    # the only effect that satisfies a more specific duplicate expectation.
    for expected_index in sorted(unmatched_expected, key=lambda index: expected[index].service_data is None):
        wanted = expected[expected_index]
        match = next(
            (index for index in sorted(unmatched_actual) if _comparison(wanted, actual[index]).matched),
            None,
        )
        if match is not None:
            comparisons[expected_index] = _comparison(wanted, actual[match])
            unmatched_expected.remove(expected_index)
            unmatched_actual.remove(match)

    partition_match = _equivalent_target_partition(expected, actual, unmatched_expected, unmatched_actual)
    if partition_match is not None:
        # Branch boundary: only the still-unmatched ledger tail is consumed as one diagnostic aggregate.
        expected_index, comparison = partition_match
        comparisons[expected_index] = comparison
        complete_comparisons = tuple(comparison for comparison in comparisons if comparison is not None)
        return ActionResult(True, "equivalent_target_partition", complete_comparisons, ())

    # Pair each remaining expectation with the still-available effect sharing the
    # most dimensions. Authored order and ledger order resolve equal-distance ties.
    for expected_index in sorted(unmatched_expected):
        wanted = expected[expected_index]
        if unmatched_actual:
            actual_index = max(
                sorted(unmatched_actual),
                key=lambda index: _match_count(_comparison(wanted, actual[index])),
            )
            comparisons[expected_index] = _comparison(wanted, actual[actual_index])
            unmatched_actual.remove(actual_index)
        else:
            comparisons[expected_index] = ActionComparison(wanted, None, False, False, False, False)

    complete_comparisons = tuple(comparison for comparison in comparisons if comparison is not None)
    unexpected = tuple(actual[index] for index in sorted(unmatched_actual))
    passed = all(comparison.matched for comparison in complete_comparisons) and not unexpected
    reason = _reason(complete_comparisons, unexpected, expected, ledger)
    return ActionResult(passed, reason, complete_comparisons, unexpected)


def _normalize_action(action: Mapping[str, object]) -> ObservedAction:
    """Normalize a successful ledger record into a stable diagnostic shape."""
    domain = str(action.get("domain", ""))
    service = str(action.get("service", ""))
    if "." in service and not domain:
        domain, service = service.split(".", 1)
    data = action.get("service_data")
    comparable_data = {
        str(key): _canonical(value)
        for key, value in (data.items() if isinstance(data, Mapping) else ())
        if key not in {"entity_id", "entity_ids"}
    }
    return ObservedAction(domain, service, tuple(sorted(_action_targets(action))), comparable_data)


def _comparison(expected: RequiredAction, actual: ObservedAction) -> ActionComparison:
    service_matches = (expected.domain, expected.service) == (actual.domain, actual.service)
    target_matches = tuple(sorted(expected.target_entity_ids)) == actual.target_entity_ids
    service_data_matches = expected.service_data is None or _data_key(expected.service_data) == _data_key(
        actual.service_data
    )
    return ActionComparison(
        expected,
        actual,
        service_matches,
        target_matches,
        service_data_matches,
        service_matches and target_matches and service_data_matches,
    )


def _match_count(comparison: ActionComparison) -> int:
    return sum((comparison.service_matches, comparison.target_matches, comparison.service_data_matches))


def _equivalent_target_partition(
    expected: tuple[RequiredAction, ...],
    actual: tuple[ObservedAction, ...],
    unmatched_expected: set[int],
    unmatched_actual: set[int],
) -> tuple[int, ActionComparison] | None:
    """Aggregate split successful calls only when they exactly partition one authored target set."""
    if len(unmatched_expected) != 1 or len(unmatched_actual) < 2:
        return None

    expected_index = next(iter(unmatched_expected))
    wanted = expected[expected_index]
    if len(wanted.target_entity_ids) < 2:
        return None

    remaining = tuple(actual[index] for index in sorted(unmatched_actual))
    aggregate = _partition_aggregate(wanted, remaining)
    if aggregate is None:
        return None
    return expected_index, _comparison(wanted, aggregate)


def _partition_aggregate(wanted: RequiredAction, actual: tuple[ObservedAction, ...]) -> ObservedAction | None:
    """Return one diagnostic action when every remaining effect is a target partition segment."""
    expected_targets = set(wanted.target_entity_ids)
    if len(expected_targets) != len(wanted.target_entity_ids):
        return None

    data_key = _data_key(actual[0].service_data)
    if wanted.service_data is not None and _data_key(wanted.service_data) != data_key:
        return None

    observed_targets: set[str] = set()
    for action in actual:
        if (action.domain, action.service) != (wanted.domain, wanted.service):
            return None
        if _data_key(action.service_data) != data_key:
            return None

        targets = action.target_entity_ids
        if not targets or any(not target for target in targets):
            return None
        unique_targets = set(targets)
        if len(unique_targets) != len(targets):
            return None
        if observed_targets & unique_targets:
            return None

        # State mutation point: successful per-call validation contributes this call's whole target set.
        observed_targets.update(unique_targets)

    if observed_targets != expected_targets:
        return None
    return ObservedAction(wanted.domain, wanted.service, tuple(sorted(observed_targets)), dict(actual[0].service_data))


def _reason(
    comparisons: tuple[ActionComparison, ...],
    unexpected: tuple[ObservedAction, ...],
    expected: tuple[RequiredAction, ...],
    ledger: ActionLedger,
) -> ActionOutcomeReason:
    """Classify a structured comparison without weakening exact correctness."""
    if not ledger.successful:
        return "action_rejected" if ledger.rejected else "no_action"
    if all(comparison.matched for comparison in comparisons):
        if not unexpected:
            return "ok"
        if all(any(_comparison(wanted, action).matched for wanted in expected) for action in unexpected):
            return "duplicate_action"
        return "unexpected_action"

    missing = tuple(comparison for comparison in comparisons if comparison.actual is None)
    paired_mismatches = tuple(
        comparison for comparison in comparisons if comparison.actual is not None and not comparison.matched
    )
    if missing:
        return "missing_action" if not paired_mismatches and not unexpected else "multiple_action_mismatches"
    if len(paired_mismatches) != 1 or unexpected:
        return "multiple_action_mismatches"

    mismatch = paired_mismatches[0]
    dimensions = (mismatch.service_matches, mismatch.target_matches, mismatch.service_data_matches)
    if dimensions == (False, True, True):
        return "wrong_service"
    if dimensions == (True, False, True):
        return "wrong_target"
    if dimensions == (True, True, False):
        return "wrong_service_data"
    if dimensions == (False, False, True):
        return "wrong_service_and_target"
    if dimensions == (False, True, False):
        return "wrong_service_and_data"
    if dimensions == (True, False, False):
        return "wrong_target_and_data"
    return "wrong_service_target_and_data"


def _data_key(data: Mapping[str, object] | None) -> str:
    comparable_data = {
        key: _canonical(value) for key, value in (data or {}).items() if key not in {"entity_id", "entity_ids"}
    }
    return json.dumps(comparable_data, sort_keys=True, separators=(",", ":"))


def _canonical(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def _action_targets(action: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    target = action.get("target")
    if isinstance(target, Mapping):
        for key in ("entity_id", "entity_ids"):
            value = target.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(item for item in value if isinstance(item, str))
    return tuple(values)
