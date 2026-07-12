"""Finite semantic comparisons for concrete model answers."""

from math import isclose

from llm_sandbox_evals.schema import (
    AggregateAnswer,
    AggregateExpectation,
    EntityAnswer,
    EntityCollectionAnswer,
    EntityCollectionExpectation,
    EntityExpectation,
    EntityRelationAnswer,
    EntityRelationExpectation,
    NoDataAnswer,
)


def _entity_matches(expected: EntityExpectation, answer: EntityAnswer) -> bool:
    """Match an entity identifier and scalar value."""
    return answer.entity_id == expected.entity_id and _same_scalar(answer.value, expected.value, expected.tolerance)


def _collection_matches(expected: EntityCollectionExpectation, answer: EntityCollectionAnswer) -> bool:
    """Match a collection as an exact set without accepting duplicates."""
    return len(answer.entity_ids) == len(set(answer.entity_ids)) and set(answer.entity_ids) == set(expected.entity_ids)


def _aggregate_matches(expected: AggregateExpectation, answer: AggregateAnswer) -> bool:
    """Match an aggregate scalar using its authored tolerance when present."""
    return _same_scalar(answer.value, expected.value, expected.tolerance)


def _relation_matches(expected: EntityRelationExpectation, answer: EntityRelationAnswer) -> bool:
    """Match both identifiers in an authored relation."""
    return answer.entity_id == expected.entity_id and answer.related_id == expected.related_id


def _no_data_matches(answer: NoDataAnswer) -> bool:
    """Accept only an explicit true no-data result."""
    return answer.no_data is True


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
