"""Compose semantic, grounding, and action results into binary outcomes."""

from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise, product

from llm_sandbox_evals.schema import (
    ActionAnswer,
    ActionResult,
    AggregateClaim,
    AnswerShape,
    CaseOutcome,
    CollectionClaim,
    ConclusionResult,
    EvalCase,
    EventClaim,
    ExpectedConclusion,
    Finding,
    ListAnswer,
    NoDataClaim,
    ReadAnswer,
    RelationClaim,
    ValueClaim,
)
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.assertions import _list_items_match, _read_finding_matches
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence
from llm_sandbox_evals.scoring.evidence import normalize_events


def evaluate_case(
    case: EvalCase,
    answer: AnswerShape,
    tool_events: Sequence,
    recorded_actions: Sequence[dict[str, object]] = (),
) -> tuple[CaseOutcome, tuple[ConclusionResult, ...], tuple[ActionResult, ...]]:
    """Score a case-selected flat answer against oracle claims and action ledgers."""
    evidence = normalize_events(tool_events)
    # Branch boundary: each model-facing shape has one finite scoring path.
    match answer:
        case ReadAnswer(findings=findings):
            conclusion_results = tuple(
                _score_read_conclusion(expected, findings, evidence) for expected in case.expected.conclusions
            )
            selected_findings = {
                id(result.matched_finding): result.expected
                for result in conclusion_results
                if result.matched_finding is not None
            }
            all_answer_content_grounded = all(
                _finding_grounded(finding, evidence, selected_findings.get(id(finding))) for finding in findings
            )
        case ListAnswer(items=items):
            conclusion_results = tuple(
                _score_list_conclusion(expected, items, evidence) for expected in case.expected.conclusions
            )
            grounded_items = set().union(
                *(
                    _collection_items(conclusion.claim, evidence)
                    for conclusion in case.expected.conclusions
                    if isinstance(conclusion.claim, CollectionClaim)
                )
            )
            all_answer_content_grounded = all(item in grounded_items for item in items)
        case ActionAnswer():
            conclusion_results = tuple(
                ConclusionResult(expected, None, "mismatched", "ungrounded", "missing_finding")
                for expected in case.expected.conclusions
            )
            all_answer_content_grounded = True
    ledger = build_action_ledger(recorded_actions)
    action_result = score_actions(case.expected.actions, case.expected.blocked_outcome, ledger)
    correct = all(
        result.semantic_status == "matched" and result.grounding_status == "grounded" for result in conclusion_results
    )
    action_passed = action_result.passed if action_result is not None else not ledger.successful
    correct = correct and all_answer_content_grounded and action_passed
    reason = (
        "ok" if correct else _reason(conclusion_results, action_result, all_answer_content_grounded, ledger.successful)
    )
    return (
        CaseOutcome("correct" if correct else "incorrect", reason),
        conclusion_results,
        ((action_result,) if action_result is not None else ()),
    )


def _score_read_conclusion(
    expected: ExpectedConclusion, findings: Sequence[Finding], evidence: NormalizedEvidence
) -> ConclusionResult:
    if isinstance(expected.claim, NoDataClaim):
        # No-data is represented by an empty answer and proved by the oracle's exact resolved scope.
        if findings:
            return ConclusionResult(expected, None, "mismatched", "ungrounded", "expected_empty_findings")
        grounded = _no_data_grounded(expected.claim, evidence)
        return ConclusionResult(
            expected,
            None,
            "matched",
            "grounded" if grounded else "ungrounded",
            "ok" if grounded else "scope_mismatch",
        )
    matching = next(
        (finding for finding in findings if _read_finding_matches(expected.claim, finding, expected.tolerance)),
        None,
    )
    # Some registry joins are evidenced as relations but have no useful scalar model value.
    if (
        matching is None
        and isinstance(expected.claim, RelationClaim)
        and expected.claim.relation
        in {
            "entity_area",
            "device_area",
            "area_floor",
        }
    ):
        matching = next((finding for finding in findings if finding.subject == expected.claim.subject_id), None)
    if matching is None:
        return ConclusionResult(expected, None, "mismatched", "ungrounded", "missing_finding")
    expected_grounded = _oracle_claim_grounded(expected, evidence)
    if not expected_grounded:
        return ConclusionResult(expected, matching, "matched", "ungrounded", "oracle_not_grounded")
    actual_grounded = _finding_grounded(matching, evidence, expected)
    return ConclusionResult(
        expected,
        matching,
        "matched",
        "grounded" if actual_grounded else "ungrounded",
        "ok" if actual_grounded else "finding_not_grounded",
    )


def _score_list_conclusion(
    expected: ExpectedConclusion, items: list[str], evidence: NormalizedEvidence
) -> ConclusionResult:
    if not isinstance(expected.claim, CollectionClaim) or not _list_items_match(
        expected.claim, items, expected.assertion
    ):
        return ConclusionResult(expected, None, "mismatched", "ungrounded", "collection_mismatch")
    grounded = set(expected.claim.items) <= _collection_items(expected.claim, evidence)
    return ConclusionResult(
        expected,
        None,
        "matched",
        "grounded" if grounded else "ungrounded",
        "ok" if grounded else "collection_not_grounded",
    )


def _oracle_claim_grounded(expected: ExpectedConclusion, evidence: NormalizedEvidence) -> bool:
    claim = expected.claim
    if isinstance(claim, AggregateClaim):
        return _aggregate_claim_grounded(claim, evidence, expected.tolerance)
    if isinstance(claim, RelationClaim):
        return any(
            all(
                fact.get(key) == getattr(claim, key)
                for key in ("subject_kind", "subject_id", "relation", "object_kind", "object_id")
            )
            for fact in evidence.for_kind("relation")
        )
    if isinstance(claim, EventClaim):
        matching = Finding(subject=claim.entity_id, value=claim.value, when=claim.when)
    elif isinstance(claim, ValueClaim):
        matching = Finding(subject=claim.subject_id, value=claim.value)
    else:
        return False
    return _finding_grounded(matching, evidence, expected)


def _finding_grounded(
    finding: Finding,
    evidence: NormalizedEvidence,
    expected: ExpectedConclusion | None,
) -> bool:
    if expected is not None and isinstance(expected.claim, AggregateClaim):
        return _aggregate_claim_grounded(expected.claim, evidence, expected.tolerance)
    if expected is not None and isinstance(expected.claim, RelationClaim):
        return _oracle_claim_grounded(expected, evidence)
    tolerance = expected.tolerance if expected is not None else None
    scalar_facts = (
        *((fact, "subject_id", "value", "timestamp") for fact in evidence.for_kind("value")),
        *((fact, "entity_id", "value", "when") for fact in evidence.for_kind("history_attribute")),
        *((fact, "entity_id", "state", "when") for fact in evidence.for_kind("history_row")),
        *((fact, "statistic_id", "value", "when") for fact in evidence.for_kind("statistic_value")),
        *((fact, "entity_id", "message", "when") for fact in evidence.for_kind("logbook_event")),
        *((fact, "entity_id", "value", "when") for fact in evidence.for_kind("automation_run")),
    )
    return any(
        fact.get(subject_key) == finding.subject
        and _same_scalar(fact.get(value_key), finding.value, tolerance)
        and (finding.when is None or fact.get(when_key) == finding.when)
        for fact, subject_key, value_key, when_key in scalar_facts
    ) or any(
        fact.get("subject_id") == finding.subject and fact.get("object_id") == finding.value
        for fact in evidence.for_kind("relation")
    )


def _no_data_grounded(claim: NoDataClaim, evidence: NormalizedEvidence) -> bool:
    rows = {
        "history": "history_row",
        "statistics": "statistic_value",
        "logbook": "logbook_event",
        "automation": "automation_run",
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


def _collection_items(claim: CollectionClaim, evidence: NormalizedEvidence) -> set[str]:
    collection = claim.collection
    candidates: dict[str, list[EvidenceFact]] = {}
    for fact in evidence.for_kind("value"):
        subject_id = fact.get("subject_id")
        if isinstance(subject_id, str):
            candidates.setdefault(subject_id, []).append(fact)
    if collection == "entity_ids":
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(fact.get("subject_kind") == "entity" for fact in facts)
        }
    if collection == "automation_ids":
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(fact.get("subject_kind") == "automation" for fact in facts)
        }
    if claim.filter_kind == "domain":
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(fact.get("field") == "domain" and fact.get("value") == claim.filter_value for fact in facts)
            or subject_id.split(".", 1)[0] == claim.filter_value
        }
    if claim.filter_kind == "state":
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(fact.get("field") == "state" and fact.get("value") == claim.filter_value for fact in facts)
        }
    if claim.filter_kind in {"area", "device", "floor"}:
        if claim.filter_kind == "floor":
            area_ids = {
                fact.get("subject_id")
                for fact in evidence.for_kind("relation")
                if fact.get("relation") == "area_floor" and fact.get("object_id") == claim.filter_value
            }
            candidates = {
                subject_id: facts
                for subject_id, facts in candidates.items()
                if any(
                    fact.get("relation") == "entity_area" and fact.get("object_id") in area_ids
                    for fact in evidence.for_kind("relation")
                    if fact.get("subject_id") == subject_id
                )
            }
            return set(candidates)
        relation = {"area": "entity_area", "device": "entity_device"}[claim.filter_kind]
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(
                fact.get("relation") == relation and fact.get("object_id") == claim.filter_value
                for fact in evidence.for_kind("relation")
                if fact.get("subject_id") == subject_id
            )
        }
    if claim.filter_kind == "label":
        candidates = {
            subject_id: facts
            for subject_id, facts in candidates.items()
            if any(
                fact.get("entity_id") == subject_id and fact.get("value") == claim.filter_value
                for fact in evidence.for_kind("association")
            )
        }
    return set(candidates)


def _aggregate_claim_grounded(claim: AggregateClaim, evidence: NormalizedEvidence, tolerance: float | None) -> bool:
    if claim.source == "history" and claim.operator in {"duration_seconds", "time_in_state"}:
        return any(
            _same_scalar(value, claim.value, tolerance) for value in _history_duration_candidates(claim, evidence)
        )
    computed = _aggregate_value(claim, evidence)
    if computed is None:
        return False
    return _same_scalar(computed, claim.value, tolerance)


def _aggregate_value(claim: AggregateClaim, evidence: NormalizedEvidence) -> object:
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
    if claim.operator in {"first_seen", "last_seen"}:
        timestamps = sorted(
            (when for when, _value in rows if _parse_time(when) is not None),
            key=lambda value: _parse_time(value) or datetime.min.replace(tzinfo=UTC),
        )
        return (timestamps[0] if claim.operator == "first_seen" else timestamps[-1]) if timestamps else None
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
    claim: AggregateClaim, evidence: NormalizedEvidence, subject_ids: set[str]
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


def _history_duration(claim: AggregateClaim, evidence: NormalizedEvidence) -> float | None:
    candidates = _history_duration_candidates(claim, evidence)
    return candidates[0] if candidates else None


def _history_duration_candidates(claim: AggregateClaim, evidence: NormalizedEvidence) -> tuple[float, ...]:
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


def _same_scalar(actual: object, expected: object, tolerance: float | None) -> bool:
    actual_number, expected_number = _number(actual), _number(expected)
    if actual_number is not None and expected_number is not None:
        return abs(actual_number - expected_number) <= (tolerance or 0.0)
    return actual == expected


def _reason(
    results: Sequence[ConclusionResult],
    action: ActionResult | None,
    all_grounded: bool,
    unexpected_effects: Sequence[dict[str, object]],
) -> str:
    if not all_grounded:
        return "extra_ungrounded_finding"
    if any(result.semantic_status == "mismatched" for result in results):
        return next((result.reason for result in results if result.semantic_status == "mismatched"), "missing_finding")
    if any(result.grounding_status == "ungrounded" for result in results):
        return next(
            (result.reason for result in results if result.grounding_status == "ungrounded"),
            "finding_not_grounded",
        )
    if action is None:
        return "unexpected_effect" if unexpected_effects else "incorrect"
    return action.mismatches[0] if action.mismatches else "incorrect"
