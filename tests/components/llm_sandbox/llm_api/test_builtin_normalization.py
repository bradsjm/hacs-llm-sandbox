"""Tests for Monty builtin-forgiveness normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.builtin_normalization import (
    GETATTR_RESOLVED,
    HASATTR_RESOLVED,
    REWROTE_MAP_FILTER,
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
            "result = hasattr(date, 'today')",
            "result = True",
            {HASATTR_RESOLVED},
            id="hasattr-date-today",
        ),
        pytest.param(
            "result = hasattr(datetime, 'now')",
            "result = True",
            {HASATTR_RESOLVED},
            id="hasattr-datetime-now",
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
            "result = getattr(date, 'fromisoformat')",
            "result = date.fromisoformat",
            {GETATTR_RESOLVED},
            id="getattr-date-fromisoformat",
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
    ("code", "dropped_token"),
    [
        pytest.param("result = list(map(str, items))", "map(", id="map-name"),
        pytest.param("result = list(map(lambda x: x * 2, items))", "map(", id="map-lambda"),
        pytest.param("result = list(map(f, a, b))", "map(", id="map-multi-iterable"),
        pytest.param("result = list(filter(pred, items))", "filter(", id="filter-name"),
        pytest.param("result = list(filter(lambda x: x > 1, items))", "filter(", id="filter-lambda"),
        pytest.param("result = list(filter(None, items))", "filter(", id="filter-none"),
    ],
)
def test_normalize_builtins_rewrites_map_filter(code: str, dropped_token: str) -> None:
    """map/filter rewrite to list comprehensions with fresh loop targets."""
    normalized, labels = normalize_builtins(code)

    assert REWROTE_MAP_FILTER in labels
    assert dropped_token not in normalized
    assert "_lsbx_" in normalized
    # Multi-iterable map is driven through zip to preserve pairing.
    if code == "result = list(map(f, a, b))":
        assert "zip(" in normalized


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("result = map(obj.method, items)", id="map-attribute-func"),
        pytest.param("result = filter(obj.method, items)", id="filter-attribute-func"),
        pytest.param("result = map(f, items, extra=1)", id="map-keyword-arg"),
    ],
)
def test_normalize_builtins_leaves_unsafe_map_filter_unchanged(code: str) -> None:
    """Unsafe map/filter shapes are left untouched so Monty surfaces the natural error."""
    normalized, labels = normalize_builtins(code)

    assert normalized == code
    assert REWROTE_MAP_FILTER not in labels
