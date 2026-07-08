"""Tests for Monty builtin-forgiveness normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.normalization.builtin_normalization import (
    TYPE_NAME_RESOLVED,
    WRAPPED_NEXT_ITER,
    normalize_builtins,
)


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = type(floor_registry).__name__",
            "result = 'SafeFloorRegistry'",
            {TYPE_NAME_RESOLVED},
            id="type-name-resolved",
        ),
        pytest.param(
            "result = type(date).__name__",
            "result = 'SafeDateFacade'",
            {TYPE_NAME_RESOLVED},
            id="type-name-date",
        ),
        pytest.param(
            "result = type(datetime).__name__",
            "result = 'SafeDateTimeFacade'",
            {TYPE_NAME_RESOLVED},
            id="type-name-datetime",
        ),
        pytest.param(
            "result = type(floor_registry)",
            "result = type(floor_registry)",
            set(),
            id="bare-type-left-alone",
        ),
        pytest.param(
            "x = floor_registry\nresult = hasattr(x, 'async_list_floors')",
            "x = floor_registry\nresult = hasattr(x, 'async_list_floors')",
            set(),
            id="loop-variable-not-rewritten",
        ),
        pytest.param(
            "result = hasattr(floor_registry.async_list_floors(), 'name')",
            "result = hasattr(floor_registry.async_list_floors(), 'name')",
            set(),
            id="chain-root-not-rewritten",
        ),
        pytest.param(
            "result = hasattr(floor_registry, name)",
            "result = hasattr(floor_registry, name)",
            set(),
            id="dynamic-name-not-rewritten",
        ),
        pytest.param(
            "result = getattr(area_registry, 'missing')",
            "result = getattr(area_registry, 'missing')",
            set(),
            id="getattr-miss-no-default-left-alone",
        ),
        pytest.param(
            "result = hasattr(floor_registry,",
            "result = hasattr(floor_registry,",
            set(),
            id="syntax-error-fail-open",
        ),
    ],
)
def test_normalize_builtins(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = normalize_builtins(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = next(xs)",
            "result = next(iter(xs))",
            {WRAPPED_NEXT_ITER},
            id="next-one-arg",
        ),
        pytest.param(
            "result = next(xs, default)",
            "result = next(iter(xs), default)",
            {WRAPPED_NEXT_ITER},
            id="next-default",
        ),
        pytest.param(
            "result = next(iter(xs))",
            "result = next(iter(xs))",
            set(),
            id="explicit-iter-unchanged",
        ),
    ],
)
def test_normalize_builtins_wraps_next(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = normalize_builtins(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels
