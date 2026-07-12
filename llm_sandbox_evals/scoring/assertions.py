"""Finite semantic assertion operations."""

from math import isclose

from llm_sandbox_evals.schema import AggregateClaim, AnswerClaim, ExpectedConclusion, ValueClaim


def assertion_matches(expected: ExpectedConclusion, actual: AnswerClaim) -> bool:
    """Compare only authored typed fields; no prose or token search is involved."""
    if expected.claim.kind != actual.kind:
        return False
    if expected.assertion == "equals":
        return expected.claim == actual
    if expected.assertion == "approximate":
        if isinstance(expected.claim, ValueClaim) and isinstance(actual, ValueClaim):
            return _same_value_identity(expected.claim, actual) and _close(
                actual.value, expected.claim.value, expected.tolerance or 0.0
            )
        if isinstance(expected.claim, AggregateClaim) and isinstance(actual, AggregateClaim):
            return _same_aggregate_identity(expected.claim, actual) and _close(
                actual.value, expected.claim.value, expected.tolerance or 0.0
            )
        return False
    if expected.assertion == "empty":
        return expected.claim == actual
    expected_items = set(expected.claim.items)  # type: ignore[union-attr]
    actual_items = set(actual.items)  # type: ignore[union-attr]
    return actual_items == expected_items if expected.assertion == "exact_items" else expected_items <= actual_items


def _same_value_identity(expected: object, actual: object) -> bool:
    fields = ("subject_kind", "subject_id", "field", "attribute_name")
    return all(getattr(expected, field) == getattr(actual, field) for field in fields)


def _same_aggregate_identity(expected: object, actual: object) -> bool:
    fields = ("source", "operator", "subject_ids", "input_field", "input_value", "unit")
    return all(getattr(expected, field) == getattr(actual, field) for field in fields)


def _close(actual: object, expected: object, tolerance: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0)
    )
