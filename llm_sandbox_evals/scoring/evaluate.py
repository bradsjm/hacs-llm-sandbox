"""Compose semantic, grounding, and action results into binary outcomes."""

from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise, product

from llm_sandbox_evals.schema import (
    ActionResult,
    AggregateAnswer,
    AggregateExpectation,
    AnswerExpectation,
    AnswerShape,
    CaseOutcome,
    ConclusionResult,
    EntityAnswer,
    EntityCollectionAnswer,
    EntityCollectionExpectation,
    EntityExpectation,
    EntityRelationAnswer,
    EntityRelationExpectation,
    EvalCase,
    NoDataAnswer,
    NoDataExpectation,
)
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.assertions import (
    _aggregate_matches,
    _collection_matches,
    _entity_matches,
    _no_data_matches,
    _relation_matches,
    _same_scalar,
)
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence
from llm_sandbox_evals.scoring.evidence import normalize_events


def evaluate_case(
    case: EvalCase,
    answer: AnswerShape,
    tool_events: Sequence,
    recorded_actions: Sequence[dict[str, object]] = (),
) -> tuple[CaseOutcome, tuple[ConclusionResult, ...], tuple[ActionResult, ...]]:
    """Score one concrete answer shape against successful evidence and action ledgers."""
    evidence = normalize_events(tool_events)
    expectation = case.expected.expectation
    shape_result = _score_shape(expectation, answer, evidence) if expectation is not None else None
    conclusion_results = (shape_result,) if shape_result is not None else ()
    ledger = build_action_ledger(recorded_actions)
    action_result = score_actions(case.expected.actions, case.expected.blocked_outcome, ledger)
    correct = all(result.matched and result.grounded for result in conclusion_results)
    action_passed = action_result.passed if action_result is not None else not ledger.successful
    correct = correct and action_passed
    reason = "ok" if correct else _reason(conclusion_results, action_result, ledger.successful)
    return (
        CaseOutcome("correct" if correct else "incorrect", reason),
        conclusion_results,
        ((action_result,) if action_result is not None else ()),
    )


def _score_shape(expected: AnswerExpectation, answer: AnswerShape, evidence: NormalizedEvidence) -> ConclusionResult:
    """Score the selected concrete shape; mismatched runtime shapes cannot borrow another path."""
    matched = False
    grounded = False
    match expected, answer:
        case EntityExpectation() as entity, EntityAnswer() as submitted:
            matched = _entity_matches(entity, submitted)
            grounded = matched and _entity_grounded(entity, submitted, evidence)
        case EntityCollectionExpectation() as collection, EntityCollectionAnswer() as submitted:
            matched = _collection_matches(collection, submitted)
            grounded = matched and _collection_grounded(submitted, evidence)
        case AggregateExpectation() as aggregate, AggregateAnswer() as submitted:
            matched = _aggregate_matches(aggregate, submitted)
            grounded = matched and _aggregate_expectation_grounded(aggregate, evidence)
        case EntityRelationExpectation() as relation, EntityRelationAnswer() as submitted:
            matched = _relation_matches(relation, submitted)
            grounded = matched and _relation_grounded(relation, evidence)
        case NoDataExpectation() as no_data, NoDataAnswer() as submitted:
            matched = _no_data_matches(submitted)
            grounded = matched and _no_data_grounded(no_data, evidence)
    if grounded:
        return ConclusionResult(expected, matched, grounded, "ok")
    if matched:
        return ConclusionResult(expected, matched, grounded, "evidence_missing")
    return ConclusionResult(expected, matched, grounded, "answer_mismatch")


def _entity_grounded(expected: EntityExpectation, answer: EntityAnswer, evidence: NormalizedEvidence) -> bool:
    source_facts = {
        "states": ("value", "subject_id", "value", "field"),
        "history": (
            "history_attribute" if expected.input_field == "attribute" else "history_row",
            "entity_id",
            "value" if expected.input_field == "attribute" else "state",
            "attribute_name",
        ),
        "logbook": ("logbook_event", "entity_id", "message", "message"),
        "automation": (
            "automation_run" if expected.input_field == "run" else "value",
            "entity_id" if expected.input_field == "run" else "subject_id",
            "value",
            "field",
        ),
    }
    kind, id_key, value_key, field_key = source_facts[expected.source]
    return any(
        fact.get(id_key) == answer.entity_id
        and _same_scalar(fact.get(value_key), answer.value, expected.tolerance)
        and (
            expected.source == "logbook"
            or expected.input_field == "run"
            or (
                expected.source == "history"
                and (
                    expected.input_field == "state"
                    or (expected.input_field == "attribute" and fact.get("attribute_name") == expected.input_value)
                )
            )
            or (
                expected.input_field == "attribute"
                and fact.get(field_key) == "attribute"
                and fact.get("attribute_name") == expected.input_value
            )
            or fact.get(field_key) == expected.input_field
        )
        for fact in evidence.for_kind(kind)
    )


def _collection_grounded(answer: EntityCollectionAnswer, evidence: NormalizedEvidence) -> bool:
    returned_ids = {
        value
        for fact in evidence.facts
        for key in ("subject_id", "entity_id", "statistic_id")
        if isinstance((value := fact.get(key)), str)
    }
    return set(answer.entity_ids) <= returned_ids


def _relation_grounded(expected: EntityRelationExpectation, evidence: NormalizedEvidence) -> bool:
    return any(
        fact.get("subject_id") == expected.entity_id
        and fact.get("object_id") == expected.related_id
        and fact.get("relation") == expected.relation
        for fact in evidence.for_kind("relation")
    )


def _no_data_grounded(claim: NoDataExpectation, evidence: NormalizedEvidence) -> bool:
    rows = {
        "history": "history_row",
        "statistics": "statistic_value",
        "logbook": "logbook_event",
    }[claim.source]
    identity_key = "statistic_id" if claim.source == "statistics" else "entity_id"
    # A resolved empty result is indivisible: scope and rows must come from one event.
    scope_facts = evidence.for_kind("scope")
    row_facts = evidence.for_kind(rows)
    event_ids = {_event_identity(fact) for fact in scope_facts if fact.get("source") == claim.source}
    for event_id in event_ids:
        event_scope = {
            fact.get("entity_id")
            for fact in scope_facts
            if fact.get("source") == claim.source and _event_identity(fact) == event_id
        }
        if event_scope == set(claim.scope_entity_ids) and not any(
            fact.get(identity_key) in event_scope and _event_identity(fact) == event_id for fact in row_facts
        ):
            return True
    return False


def _aggregate_expectation_grounded(claim: AggregateExpectation, evidence: NormalizedEvidence) -> bool:
    if claim.source == "history" and claim.operator in {"duration_seconds", "time_in_state"}:
        return any(
            _same_scalar(value, claim.value, claim.tolerance)
            for value in _history_duration_candidates(claim, evidence)
        )
    computed = _aggregate_value(claim, evidence)
    if computed is None:
        return False
    return _same_scalar(computed, claim.value, claim.tolerance)


def _aggregate_value(claim: AggregateExpectation, evidence: NormalizedEvidence) -> object:
    subject_ids = set(claim.subject_ids)
    rows: list[tuple[str | None, object]] = []
    contributing_subjects: set[str] = set()
    explicit_duration_values: list[float] = []
    if claim.source == "states":
        # Branch boundary: state counts qualify by selected state; numeric operations qualify by source unit.
        if claim.operator in {"count", "mean", "minimum", "maximum", "sum", "convert"}:
            return _states_aggregate_value(claim, evidence, subject_ids)
        if claim.input_field == "none" and claim.operator in {"duration_seconds", "time_in_state"}:
            explicit_duration_facts = [
                (str(fact.get("subject_id")), number)
                for fact in evidence.for_kind("value")
                if fact.get("subject_id") in subject_ids
                and fact.get("field") == claim.operator
                and (number := _number(fact.get("value"))) is not None
            ]
            # Every declared state subject must contribute its own explicit duration.
            if {subject_id for subject_id, _number_value in explicit_duration_facts} != subject_ids:
                return None
            explicit_duration_values = [number for _subject_id, number in explicit_duration_facts]
            rows.extend(
                (str(fact.get("timestamp")) if fact.get("timestamp") is not None else None, fact.get("value"))
                for fact in evidence.for_kind("value")
                if fact.get("subject_id") in subject_ids and fact.get("field") in {"duration_seconds", "time_in_state"}
            )
            if not rows:
                rows.extend(
                    (str(fact.get("timestamp")) if fact.get("timestamp") is not None else None, fact.get("value"))
                    for fact in evidence.for_kind("value")
                    if fact.get("subject_id") in subject_ids and fact.get("field") == "state"
                )
        else:
            rows.extend(
                (str(fact.get("timestamp")) if fact.get("timestamp") is not None else None, fact.get("value"))
                for fact in evidence.for_kind("value")
                if fact.get("subject_id") in subject_ids and fact.get("field") == claim.input_field
            )
    elif claim.source == "statistics":
        facts = tuple(
            fact
            for fact in evidence.for_kind("statistic_value")
            if fact.get("statistic_id") in subject_ids and fact.get("field") == claim.input_field
        )
        rows.extend(
            (str(fact.get("when")) if fact.get("when") is not None else None, fact.get("value")) for fact in facts
        )
        contributing_subjects.update(str(fact.get("statistic_id")) for fact in facts)
    elif claim.source == "history":
        if claim.operator in {"duration_seconds", "time_in_state"}:
            return _history_duration(claim, evidence)
        facts = tuple(
            fact
            for fact in evidence.for_kind("history_row")
            if fact.get("entity_id") in subject_ids
            and (claim.input_value in {None, "state", "event_message"} or fact.get("state") == claim.input_value)
        )
        rows = [(str(fact.get("when")), fact.get("state")) for fact in facts]
        contributing_subjects.update(str(fact.get("entity_id")) for fact in facts)
    else:
        facts = tuple(
            fact
            for fact in evidence.for_kind("logbook_event")
            if fact.get("entity_id") in subject_ids
            and (claim.input_value in {None, "state", "event_message"} or fact.get("message") == claim.input_value)
        )
        rows = [(str(fact.get("when")), fact.get("message")) for fact in facts]
        contributing_subjects.update(str(fact.get("entity_id")) for fact in facts)
    # Recorder aggregates require qualifying source evidence from every declared subject.
    if claim.source in {"statistics", "history", "logbook"} and contributing_subjects != subject_ids:
        return None
    values = [value for _when, value in rows]
    numeric = [number for value in values if (number := _number(value)) is not None]
    if claim.operator == "count":
        return len(values)
    if claim.operator == "mean":
        return sum(numeric) / len(numeric) if numeric else None
    if claim.operator == "minimum":
        return min(numeric) if numeric else None
    if claim.operator == "maximum":
        return max(numeric) if numeric else None
    if claim.operator == "sum":
        return sum(numeric) if numeric else None
    if claim.operator in {"duration_seconds", "time_in_state"}:
        if explicit_duration_values:
            return sum(explicit_duration_values)
        return _duration(rows, claim.input_value)
    if claim.operator == "convert":
        if not numeric:
            return None
        return _convert(numeric[0], claim.input_value, claim.unit)
    return None


def _states_aggregate_value(
    claim: AggregateExpectation, evidence: NormalizedEvidence, subject_ids: set[str]
) -> float | int | None:
    state_facts = {
        subject_id: tuple(
            fact
            for fact in evidence.for_kind("value")
            if fact.get("subject_id") == subject_id and fact.get("field") == claim.input_field
        )
        for subject_id in subject_ids
    }
    # Branch boundary: every declared subject must contribute observed state evidence, even for a zero count.
    if any(not facts for facts in state_facts.values()):
        return None
    state_records = {
        subject_id: {
            "state": state_facts[subject_id][0].get("value"),
            "unit": next(
                (
                    fact.get("value")
                    for fact in evidence.for_kind("value")
                    if fact.get("subject_id") == subject_id
                    and fact.get("field") == "attribute"
                    and fact.get("attribute_name") == "unit_of_measurement"
                ),
                None,
            ),
        }
        for subject_id in subject_ids
    }
    if claim.operator == "count":
        return sum(record["state"] == claim.input_value for record in state_records.values())
    if claim.operator not in {"mean", "minimum", "maximum", "sum", "convert"}:
        return None
    if any(record["unit"] != claim.input_value for record in state_records.values()):
        return None
    numeric = [number for record in state_records.values() if (number := _number(record["state"])) is not None]
    if len(numeric) != len(subject_ids):
        return None
    if claim.operator == "convert":
        return _convert(numeric[0], claim.input_value, claim.unit)
    if claim.operator == "mean":
        return sum(numeric) / len(numeric)
    if claim.operator == "minimum":
        return min(numeric)
    if claim.operator == "maximum":
        return max(numeric)
    return sum(numeric)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration(rows: Sequence[tuple[str | None, object]], state: object) -> float | None:
    ordered = sorted(
        ((time, value) for time, value in rows if _parse_time(time) is not None),
        key=lambda item: _parse_time(item[0]) or datetime.min.replace(tzinfo=UTC),
    )
    total = 0.0
    for current, following in pairwise(ordered):
        if state is None or current[1] == state:
            total += (
                (_parse_time(following[0]) or datetime.min.replace(tzinfo=UTC))
                - (_parse_time(current[0]) or datetime.min.replace(tzinfo=UTC))
            ).total_seconds()
    return total


def _history_duration(claim: AggregateExpectation, evidence: NormalizedEvidence) -> float | None:
    candidates = _history_duration_candidates(claim, evidence)
    return candidates[0] if candidates else None


def _history_duration_candidates(claim: AggregateExpectation, evidence: NormalizedEvidence) -> tuple[float, ...]:
    window_by_event: dict[tuple[str, int, int, int], tuple[str, str]] = {}
    invalid_events: set[tuple[str, int, int, int]] = set()
    for fact in evidence.for_kind("history_window"):
        event_id = _event_identity(fact)
        start = fact.get("start")
        endpoint = fact.get("end")
        if (
            fact.get("valid") is not True
            or not isinstance(start, str)
            or not isinstance(endpoint, str)
            or _parse_time(start) is None
            or _parse_time(endpoint) is None
        ):
            invalid_events.add(event_id)
            continue
        window = (start, endpoint)
        previous = window_by_event.get(event_id)
        if previous is not None and previous != window:
            invalid_events.add(event_id)
        else:
            window_by_event[event_id] = window

    rows_by_subject_window: dict[str, dict[tuple[str, str], set[tuple[str, object]]]] = {
        subject_id: {} for subject_id in claim.subject_ids
    }
    for fact in evidence.for_kind("history_row"):
        entity_id = fact.get("entity_id")
        when = fact.get("when")
        event_id = _event_identity(fact)
        event_window = window_by_event.get(event_id)
        if (
            isinstance(entity_id, str)
            and entity_id in rows_by_subject_window
            and isinstance(when, str)
            and event_id not in invalid_events
            and event_window is not None
        ):
            rows_by_subject_window[entity_id].setdefault(event_window, set()).add((when, fact.get("state")))

    per_subject: list[tuple[float, ...]] = []
    for subject_id in claim.subject_ids:
        subject_candidates = tuple(
            duration
            for window, rows in rows_by_subject_window[subject_id].items()
            if (duration := _duration_through_window(tuple(rows), window, claim.input_value)) is not None
        )
        # Same-window pages union; different windows remain coherent alternatives.
        if not subject_candidates:
            return ()
        per_subject.append(subject_candidates)
    return tuple(sum(values) for values in product(*per_subject))


def _duration_through_window(
    rows: Sequence[tuple[str, object]], window: tuple[str, str], state: object
) -> float | None:
    start, endpoint = window
    start_time, endpoint_time = _parse_time(start), _parse_time(endpoint)
    if start_time is None or endpoint_time is None or start_time > endpoint_time:
        return None
    parsed_rows = [(when, value, _parse_time(when)) for when, value in rows]
    if any(parsed is None or parsed < start_time or parsed > endpoint_time for _when, _value, parsed in parsed_rows):
        return None
    ordered = sorted(rows, key=lambda item: _parse_time(item[0]) or datetime.min.replace(tzinfo=UTC))
    total = 0.0
    for current, following in pairwise((*ordered, (endpoint, None))):
        current_time = _parse_time(current[0])
        following_time = _parse_time(following[0])
        if current_time is None or following_time is None or following_time < current_time:
            return None
        if current[1] == state:
            total += (following_time - current_time).total_seconds()
    return total


def _event_identity(fact: EvidenceFact) -> tuple[str, int, int, int]:
    provenance = fact.provenance
    return (provenance.tool_name, provenance.call_index, provenance.turn_index, provenance.batch_index)


def _convert(value: float, source_unit: object, target_unit: str | None) -> float | None:
    if not isinstance(source_unit, str) or target_unit is None:
        return None
    if source_unit in {"°F", "F", "fahrenheit"} and target_unit in {"°C", "C", "celsius"}:
        return (value - 32) * 5 / 9
    if source_unit in {"°C", "C", "celsius"} and target_unit in {"°F", "F", "fahrenheit"}:
        return value * 9 / 5 + 32
    return value if source_unit == target_unit else None


def _reason(
    results: Sequence[ConclusionResult],
    action: ActionResult | None,
    unexpected_effects: Sequence[dict[str, object]],
) -> str:
    if any(not result.matched for result in results):
        return "answer_mismatch"
    if any(not result.grounded for result in results):
        return "evidence_missing"
    if action is None:
        return "action_mismatch" if unexpected_effects else "answer_mismatch"
    return "action_mismatch"
