"""Tests for Monty builtin-forgiveness normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.builtin_normalization import (
    GETATTR_RESOLVED,
    HASATTR_RESOLVED,
    TYPE_NAME_RESOLVED,
    normalize_builtins,
)


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = hasattr(floor_registry, 'async_list_floors')",
            "result = True",
            {HASATTR_RESOLVED},
            id="hasattr-true",
        ),
        pytest.param(
            "result = hasattr(floor_registry, 'nope')",
            "result = False",
            {HASATTR_RESOLVED},
            id="hasattr-false",
        ),
        pytest.param(
            "result = getattr(area_registry, 'async_list_areas')",
            "result = area_registry.async_list_areas",
            {GETATTR_RESOLVED},
            id="getattr-resolves-to-attribute",
        ),
        pytest.param(
            "result = getattr(area_registry, 'nope', 'fallback')",
            "result = 'fallback'",
            {GETATTR_RESOLVED},
            id="getattr-returns-default",
        ),
        pytest.param(
            "result = type(floor_registry).__name__",
            "result = 'SafeFloorRegistry'",
            {TYPE_NAME_RESOLVED},
            id="type-name-resolved",
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
