"""Behavioral tests for history analytics helpers."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from custom_components.llm_sandbox.llm_api.data.history import analytics_spec_from_data, run_analytics
from custom_components.llm_sandbox.llm_api.errors import RecoverableToolError
from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeConfig,
    SafeContext,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)
import pytest


def test_run_analytics_groups_buckets_and_counts_non_numeric_skips() -> None:
    """Numeric analytics group by snapshot-derived floor keys and report skipped rows."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:05:00+00:00", "state": "20", "value": 20.0},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:35:00+00:00", "state": "bad", "value": None},
        {"entity_id": "sensor.temp", "when": "2026-01-01T01:05:00+00:00", "state": "22", "value": 22.0},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"value_operations": ["mean"], "group_by": ["floor_id"], "bucket": "1h"}),
        (start, datetime(2026, 1, 1, 2, tzinfo=UTC)),
        snapshot,
    )

    assert result == [
        {
            "bucket": "2026-01-01T00:00:00+00:00",
            "floor_id": "floor-main",
            "value_mean": 20.0,
            "value_skipped_non_numeric": 1,
        },
        {"bucket": "2026-01-01T01:00:00+00:00", "floor_id": "floor-main", "value_mean": 22.0},
    ]


def test_bucketed_duration_aggregates_use_bucket_bounds() -> None:
    """Legacy duration modes inside buckets are clipped to each bucket window."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "on"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T01:00:00+00:00", "state": "on"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"aggregate": "on_duration", "bucket": "1h"}),
        (start, datetime(2026, 1, 1, 2, tzinfo=UTC)),
        snapshot,
    )

    assert result == [
        {"bucket": "2026-01-01T00:00:00+00:00", "on_duration": 3600.0, "unit": "seconds"},
        {"bucket": "2026-01-01T01:00:00+00:00", "on_duration": 3600.0, "unit": "seconds"},
    ]


def test_group_by_without_aggregate_counts_rows() -> None:
    """Implicit analytics aggregate counts rows per requested group."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:05:00+00:00", "state": "20"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:10:00+00:00", "state": "21"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"group_by": ["domain"]}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"domain": "sensor", "count": 2}]


def test_malformed_where_missing_field_is_invalid_input() -> None:
    """Malformed where entries fail validation instead of silently matching every row."""
    with pytest.raises(RecoverableToolError) as err:
        analytics_spec_from_data({"where": [{}]})

    assert err.value.key == "invalid_tool_input"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"aggregate": {"value": ["mean"]}}, id="aggregate-object"),
        pytest.param(
            {"aggregate": "on_duration", "value_operations": ["mean"]},
            id="aggregate-and-value-operations",
        ),
    ],
)
def test_aggregate_input_rejects_object_and_combined_numeric_mode(data: dict[str, object]) -> None:
    """Aggregate modes and numeric value operations are separate input forms."""
    with pytest.raises(RecoverableToolError) as err:
        analytics_spec_from_data(data)

    assert err.value.key == "invalid_tool_input"


def test_group_by_missing_and_string_keys_sort_deterministically() -> None:
    """Grouping by optional location keys does not crash when some values are missing."""
    base = _snapshot()
    temp_state = base.states["sensor.temp"]
    snapshot = replace(
        base,
        states=base.states
        | {"sensor.other": replace(temp_state, entity_id="sensor.other", object_id="other", area_id=None)},
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.other", "when": "2026-01-01T00:00:00+00:00", "state": "10"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "20"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"group_by": ["area_id"]}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"area_id": "area-main", "count": 1}, {"area_id": None, "count": 1}]


def test_empty_bucketed_duration_with_group_returns_no_rows() -> None:
    """Grouped duration buckets with no history rows keep the standard empty analytics result."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)

    result = run_analytics(
        [],
        analytics_spec_from_data({"aggregate": "on_duration", "bucket": "1h", "group_by": ["domain"]}),
        (start, datetime(2026, 1, 1, 2, tzinfo=UTC)),
        snapshot,
    )

    assert result == []


def test_grouped_on_duration_partitions_entity_streams() -> None:
    """Domain-grouped duration analytics sum per-entity streams without interleaving."""
    base = _snapshot()
    temp_state = base.states["sensor.temp"]
    snapshot = replace(
        base,
        states=base.states
        | {"sensor.other": replace(temp_state, entity_id="sensor.other", object_id="other", name="Other")},
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "on"},
        {"entity_id": "sensor.other", "when": "2026-01-01T00:05:00+00:00", "state": "on"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:10:00+00:00", "state": "off"},
        {"entity_id": "sensor.other", "when": "2026-01-01T00:15:00+00:00", "state": "off"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"aggregate": "on_duration", "group_by": ["domain"]}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"domain": "sensor", "on_duration": 1200.0, "unit": "seconds"}]


def test_declarative_count_transitions_uses_from_to_filters() -> None:
    """Sequence-dependent declarative aggregates pass filters into per-entity streams."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "off"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:10:00+00:00", "state": "on"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:20:00+00:00", "state": "off"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:30:00+00:00", "state": "on"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data(
            {"aggregate": "count_transitions", "group_by": ["domain"], "from_state": "off", "to_state": "on"}
        ),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"domain": "sensor", "transitions": 2}]


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("first_seen", id="first-seen"),
        pytest.param("last_seen", id="last-seen"),
        pytest.param("time_in_state", id="time-in-state"),
        pytest.param("state_counts", id="state-counts"),
        pytest.param("on_duration", id="on-duration"),
    ],
)
def test_from_state_rejected_outside_count_transitions(mode: str) -> None:
    """from_state has no prior-state semantics outside count_transitions."""
    with pytest.raises(RecoverableToolError) as err:
        analytics_spec_from_data({"aggregate": mode, "from_state": "off"})

    assert err.value.key == "invalid_tool_input"


def test_bucketed_count_transitions_carries_previous_state() -> None:
    """Transition buckets include the previous row needed for boundary transitions."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:50:00+00:00", "state": "off"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T01:10:00+00:00", "state": "on"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data(
            {"aggregate": "count_transitions", "bucket": "1h", "from_state": "off", "to_state": "on"}
        ),
        (start, datetime(2026, 1, 1, 2, tzinfo=UTC)),
        snapshot,
    )

    assert result == [
        {"bucket": "2026-01-01T00:00:00+00:00", "transitions": 0},
        {"bucket": "2026-01-01T01:00:00+00:00", "transitions": 1},
    ]


def test_analytics_applies_default_limit_deterministically() -> None:
    """Analytics without a user limit still returns a bounded deterministic prefix."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": f"sensor.temp_{index:03}", "when": "2026-01-01T00:00:00+00:00", "state": str(index)}
        for index in range(501)
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"group_by": ["entity_id"]}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert len(result) == 500
    assert result[0] == {"entity_id": "sensor.temp_000", "count": 1}
    assert result[-1] == {"entity_id": "sensor.temp_499", "count": 1}


def test_descending_string_order_by_reverses_lexical_order() -> None:
    """Descending order_by works for string fields such as entity_id."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.other", "when": "2026-01-01T00:00:00+00:00", "state": "10"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "20"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"group_by": ["entity_id"], "order_by": "-entity_id"}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [
        {"entity_id": "sensor.temp", "count": 1},
        {"entity_id": "sensor.other", "count": 1},
    ]


def test_order_by_preserves_numeric_order_before_limit() -> None:
    """Numeric aggregate fields sort numerically rather than lexicographically."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "2", "value": 2.0},
        {"entity_id": "sensor.other", "when": "2026-01-01T00:00:00+00:00", "state": "10", "value": 10.0},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data(
            {"value_operations": ["sum"], "group_by": ["entity_id"], "order_by": "-value_sum", "limit": 1}
        ),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"entity_id": "sensor.other", "value_sum": 10.0}]


def test_descending_order_by_keeps_missing_values_last() -> None:
    """Descending numeric sorts do not lift missing aggregate values above real numbers."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:00:00+00:00", "state": "2", "value": 2.0},
        {"entity_id": "sensor.other", "when": "2026-01-01T00:00:00+00:00", "state": "bad", "value": None},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"value_operations": ["mean"], "group_by": ["entity_id"], "order_by": "-value_mean"}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [
        {"entity_id": "sensor.temp", "value_mean": 2.0},
        {"entity_id": "sensor.other", "value_mean": None, "value_skipped_non_numeric": 1},
    ]


def test_grouped_first_seen_uses_timestamp_order_across_entities() -> None:
    """Grouped seen aggregates choose the actual earliest target-state row."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:01:00+00:00", "state": "ignored"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:10:00+00:00", "state": "target"},
        {"entity_id": "sensor.other", "when": "2026-01-01T00:05:00+00:00", "state": "target"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"aggregate": "first_seen", "group_by": ["domain"], "to_state": "target"}),
        (start, datetime(2026, 1, 1, 1, tzinfo=UTC)),
        snapshot,
    )

    assert result == [{"domain": "sensor", "first_seen": {"state": "target", "at": "2026-01-01T00:05:00+00:00"}}]


def test_bucketed_time_in_state_attributes_seconds_once_per_bucket() -> None:
    """Bucketed time-in-state assigns only the overlapping seconds to each bucket."""
    snapshot = _snapshot()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = [
        {"entity_id": "sensor.temp", "when": "2026-01-01T00:30:00+00:00", "state": "on"},
        {"entity_id": "sensor.temp", "when": "2026-01-01T01:15:00+00:00", "state": "off"},
    ]

    result = run_analytics(
        rows,
        analytics_spec_from_data({"aggregate": "time_in_state", "bucket": "1h"}),
        (start, datetime(2026, 1, 1, 2, tzinfo=UTC)),
        snapshot,
    )

    first_bucket_states = cast(dict[str, float], result[0]["time_in_state"])
    second_bucket_states = cast(dict[str, float], result[1]["time_in_state"])
    assert len(result) == 2
    assert result[0]["bucket"] == "2026-01-01T00:00:00+00:00"
    assert first_bucket_states["on"] == pytest.approx(1800.0)
    assert result[1]["bucket"] == "2026-01-01T01:00:00+00:00"
    assert second_bucket_states["on"] == pytest.approx(900.0)
    assert second_bucket_states["off"] == pytest.approx(2700.0)


def _snapshot() -> HomeSnapshot:
    return HomeSnapshot(
        created_at="2026-01-01T00:00:00+00:00",
        states={
            "sensor.temp": SafeState(
                entity_id="sensor.temp",
                domain="sensor",
                object_id="temp",
                name="Temp",
                state="20",
                attributes={},
                last_changed="2026-01-01T00:00:00+00:00",
                last_changed_timestamp=0,
                last_reported=None,
                last_reported_timestamp=None,
                last_updated="2026-01-01T00:00:00+00:00",
                last_updated_timestamp=0,
                context=SafeContext(id=None, parent_id=None, user_id=None),
                area_id="area-main",
                floor_id="floor-main",
            )
        },
        entities={},
        devices={},
        areas={},
        floors={},
        config=SafeConfig(
            location_name="Home",
            latitude=0,
            longitude=0,
            elevation=0,
            time_zone="UTC",
            language="en",
            country=None,
            currency="USD",
            internal_url=None,
            external_url=None,
            units=SafeUnitSystem(
                temperature_unit="°C",
                length_unit="m",
                mass_unit="kg",
                pressure_unit="Pa",
                volume_unit="L",
                area_unit="m²",
                wind_speed_unit="m/s",
                accumulated_precipitation_unit="mm",
            ),
        ),
        services={},
        services_supports_response={},
        indexes=SnapshotIndexes({}, {}, {}, {}, {}, {}, {}),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
    )
