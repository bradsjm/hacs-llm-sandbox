"""Finite semantic assertion operations for flat model answers."""

from math import isclose

from llm_sandbox_evals.schema import (
    AggregateClaim,
    AnswerClaim,
    CollectionClaim,
    EventClaim,
    Finding,
    RelationClaim,
    ValueClaim,
)


def _read_finding_matches(claim: AnswerClaim, finding: Finding, tolerance: float | None) -> bool:
    """Match one flat finding to an authored read claim without parsing prose."""
    if isinstance(claim, ValueClaim):
        return finding.subject == claim.subject_id and _same_scalar(finding.value, claim.value, tolerance)
    if isinstance(claim, EventClaim):
        return (
            finding.subject == claim.entity_id
            and finding.value == claim.value
            and (claim.when is None or finding.when == claim.when)
        )
    if isinstance(claim, AggregateClaim):
        return (
            finding.subject in claim.subject_ids
            and _same_scalar(finding.value, claim.value, tolerance)
            and (claim.unit is None or finding.unit == claim.unit)
        )
    if isinstance(claim, RelationClaim):
        return finding.subject == claim.subject_id and finding.value == claim.object_id
    return False


def _list_items_match(claim: CollectionClaim, items: list[str], assertion: str) -> bool:
    """Apply an authored finite set assertion to submitted collection items."""
    expected_items = set(claim.items)
    actual_items = set(items)
    return actual_items == expected_items if assertion == "exact_items" else expected_items <= actual_items


def _same_scalar(actual: object, expected: object, tolerance: float | None) -> bool:
    actual_number, expected_number = _number(actual), _number(expected)
    if actual_number is not None and expected_number is not None:
        return isclose(actual_number, expected_number, abs_tol=tolerance or 0.0, rel_tol=0.0)
    return actual == expected


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
