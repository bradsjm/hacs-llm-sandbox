"""Compose semantic, grounding, and action results into binary outcomes."""

from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise

from llm_sandbox_evals.schema import (
    ActionResult,
    AggregateClaim,
    AnswerClaim,
    CaseOutcome,
    CollectionClaim,
    ConclusionResult,
    EvalAnswer,
    EvalCase,
    ExpectedConclusion,
)
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.assertions import assertion_matches
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence
from llm_sandbox_evals.scoring.evidence import normalize_events


def evaluate_case(
    case: EvalCase,
    answer: EvalAnswer,
    tool_events: Sequence,
    recorded_actions: Sequence[dict[str, object]] = (),
) -> tuple[CaseOutcome, tuple[ConclusionResult, ...], tuple[ActionResult, ...]]:
    """Score all claims against unioned successful facts and separate action ledgers."""
    evidence = normalize_events(tool_events)
    conclusion_results = tuple(
        _score_conclusion(expected, answer.claims, evidence) for expected in case.expected.conclusions
    )
    ledger = build_action_ledger(recorded_actions)
    action_result = score_actions(case.expected.actions, case.expected.blocked_outcome, ledger)
    all_answer_claims_grounded = all(_claim_grounded(claim, evidence) for claim in answer.claims)
    correct = all(
        result.semantic_status == "matched" and result.grounding_status == "grounded" for result in conclusion_results
    )
    correct = correct and all_answer_claims_grounded and action_result.passed
    reason = "ok" if correct else _reason(conclusion_results, action_result, all_answer_claims_grounded)
    return CaseOutcome("correct" if correct else "incorrect", reason), conclusion_results, (action_result,)


def is_incomplete(outcome: CaseOutcome | str) -> bool:
    """Return whether an outcome is provider/infra incomplete rather than incorrect."""
    return outcome.state == "incomplete" if isinstance(outcome, CaseOutcome) else outcome == "incomplete"


def score_case(outcome: CaseOutcome) -> float:
    """Expose the native binary score used by downstream evaluators."""
    return outcome.score


def _score_conclusion(
    expected: ExpectedConclusion, claims: Sequence[AnswerClaim], evidence: NormalizedEvidence
) -> ConclusionResult:
    matching = next((claim for claim in claims if assertion_matches(expected, claim)), None)
    if matching is None:
        return ConclusionResult(expected, None, "mismatched", "ungrounded", "missing_matching_claim")
    expected_grounded = (
        _aggregate_claim_grounded(expected.claim, evidence, expected.tolerance)
        if isinstance(expected.claim, AggregateClaim)
        else _claim_grounded(expected.claim, evidence)
    )
    actual_grounded = (
        _aggregate_claim_grounded(matching, evidence, expected.tolerance)
        if isinstance(matching, AggregateClaim)
        else _claim_grounded(matching, evidence)
    )
    if not expected_grounded:
        return ConclusionResult(expected, matching, "matched", "ungrounded", "expected_claim_not_grounded")
    return ConclusionResult(
        expected,
        matching,
        "matched",
        "grounded" if actual_grounded else "ungrounded",
        "ok" if actual_grounded else "claim_not_grounded",
    )


def _claim_grounded(claim: AnswerClaim, evidence: NormalizedEvidence) -> bool:
    if claim.kind == "value":
        direct = any(
            all(fact.get(key) == getattr(claim, key) for key in ("subject_kind", "subject_id", "field"))
            and (claim.field != "attribute" or fact.get("attribute_name") == claim.attribute_name)
            and fact.get("value") == claim.value
            for fact in evidence.for_kind("value")
        )
        if direct:
            return True
        return claim.field == "attribute" and any(
            fact.get("entity_id") == claim.subject_id
            and fact.get("attribute_name") == claim.attribute_name
            and fact.get("value") == claim.value
            for fact in evidence.for_kind("history_attribute")
        )
    if claim.kind == "relation":
        return any(
            all(
                fact.get(key) == getattr(claim, key)
                for key in ("subject_kind", "subject_id", "relation", "object_kind", "object_id")
            )
            for fact in evidence.for_kind("relation")
        )
    if claim.kind == "event":
        if claim.event_kind == "logbook_message":
            facts = evidence.for_kind("logbook_event")
            return any(
                fact.get("entity_id") == claim.entity_id
                and fact.get("message") == claim.value
                and (claim.when is None or fact.get("when") == claim.when)
                for fact in facts
            )
        if claim.event_kind == "state_transition":
            facts = evidence.for_kind("history_row")
            return any(
                fact.get("entity_id") == claim.entity_id
                and fact.get("state") == claim.value
                and (claim.when is None or fact.get("when") == claim.when)
                for fact in facts
            )
        facts = evidence.for_kind("automation_run")
        return any(
            fact.get("entity_id") == claim.entity_id
            and fact.get("value") == claim.value
            and (claim.when is None or fact.get("when") == claim.when)
            for fact in facts
        )
    if claim.kind == "no_data":
        source_scope = {
            fact.get("entity_id") for fact in evidence.for_kind("scope") if fact.get("source") == claim.source
        }
        rows = {
            "history": "history_row",
            "statistics": "statistic_value",
            "logbook": "logbook_event",
            "automation": "value",
        }[claim.source]
        if claim.source == "automation":
            rows = "automation_run"
        identity_key = "statistic_id" if claim.source == "statistics" else "entity_id"
        return source_scope == set(claim.scope_entity_ids) and not any(
            fact.get(identity_key) in source_scope for fact in evidence.for_kind(rows)
        )
    if claim.kind == "collection":
        grounded = _collection_items(claim, evidence)
        return set(claim.items) <= grounded
    return isinstance(claim, AggregateClaim) and _aggregate_claim_grounded(claim, evidence, None)


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
        relation = {"area": "entity_area", "device": "entity_device", "floor": "entity_floor"}[claim.filter_kind]
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
    computed = _aggregate_value(claim, evidence)
    if computed is None:
        return False
    return _same_scalar(computed, claim.value, tolerance)


def _aggregate_value(claim: AggregateClaim, evidence: NormalizedEvidence) -> object:
    subject_ids = set(claim.subject_ids)
    rows: list[tuple[str | None, object]] = []
    explicit_duration_values: list[float] = []
    if claim.source == "states":
        if claim.input_field == "none" and claim.operator in {"duration_seconds", "time_in_state"}:
            explicit_duration_values = [
                number
                for fact in evidence.for_kind("value")
                if fact.get("subject_id") in subject_ids
                and fact.get("field") == claim.operator
                and (number := _number(fact.get("value"))) is not None
            ]
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
        rows.extend(
            (str(fact.get("when")) if fact.get("when") is not None else None, fact.get("value"))
            for fact in evidence.for_kind("statistic_value")
            if fact.get("statistic_id") in subject_ids and fact.get("field") == claim.input_field
        )
    elif claim.source == "history":
        rows = [
            (str(fact.get("when")), fact.get("state"))
            for fact in evidence.for_kind("history_row")
            if fact.get("entity_id") in subject_ids
        ]
    else:
        rows = [
            (str(fact.get("when")), fact.get("message"))
            for fact in evidence.for_kind("logbook_event")
            if fact.get("entity_id") in subject_ids
        ]
    if (
        claim.source in {"history", "logbook"}
        and claim.operator not in {"duration_seconds", "time_in_state"}
        and claim.input_value not in {None, "state", "event_message"}
    ):
        rows = [(when, value) for when, value in rows if value == claim.input_value]
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


def _reason(results: Sequence[ConclusionResult], action: ActionResult, all_grounded: bool) -> str:
    if not all_grounded:
        return "extra_claim_not_grounded"
    if any(result.semantic_status == "mismatched" for result in results):
        return "conclusion_mismatch"
    if any(result.grounding_status == "ungrounded" for result in results):
        return "conclusion_not_grounded"
    return action.mismatches[0] if action.mismatches else "incorrect"
