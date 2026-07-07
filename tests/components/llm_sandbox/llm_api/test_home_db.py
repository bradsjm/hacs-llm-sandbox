"""Behavioral tests for the bounded home SQLite database."""

import pytest
from custom_components.llm_sandbox.llm_api.errors import HelperExecutionError
from custom_components.llm_sandbox.llm_api.home_db import HomeDatabase, referenced_tables

from tests.components.llm_sandbox.llm_api.tools.test_analytics import _snapshot


def test_home_db_queries_visible_states_and_json_attributes() -> None:
    """Read-only SQL can query the snapshot states table and JSON attributes."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        result = db.execute(
            "select entity_id, floor_id, json_extract(attributes, '$.missing') as missing from states", 9999999999
        )
    finally:
        db.close()

    assert result.rows == [{"entity_id": "sensor.temp", "floor_id": "floor-main", "missing": None}]
    assert result.truncated is False


def test_home_db_blocks_writes() -> None:
    """User SQL cannot write to the per-run SQLite database."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        with pytest.raises(HelperExecutionError) as err:
            db.execute("delete from states", 9999999999)
    finally:
        db.close()

    assert err.value.key == "sql_read_only"


def test_home_db_can_load_history_after_prior_user_query() -> None:
    """A prior read-only user query does not prevent later trusted lazy loads."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        assert db.execute("select count(*) as count from states", 9999999999).rows == [{"count": 1}]

        db.load_history(
            [
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.0,
                }
            ]
        )
        result = db.execute("select entity_id, value from history", 9999999999)
    finally:
        db.close()

    assert result.rows == [{"entity_id": "sensor.temp", "value": 20.0}]


def test_home_db_comma_join_queries_loaded_history() -> None:
    """SQLite implicit comma joins over history return loaded recorder rows."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        db.load_history(
            [
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.0,
                }
            ]
        )
        result = db.execute(
            "select history.entity_id, history.value from states, history where states.entity_id = history.entity_id",
            9999999999,
        )
    finally:
        db.close()

    assert result.rows == [{"entity_id": "sensor.temp", "value": 20.0}]


def test_home_db_allows_table_list_pragma() -> None:
    """Approved schema discovery includes SQLite PRAGMA table_list."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        result = db.execute("pragma table_list", 9999999999)
    finally:
        db.close()

    assert any(row["name"] == "states" for row in result.rows)


def test_referenced_tables_detects_quoted_and_schema_qualified_names() -> None:
    """Lazy-loading table detection handles valid SQLite quoted and qualified references."""
    assert referenced_tables('select * from "history"') == {"history"}
    assert referenced_tables("select * from main.history join `statistics` on 1 = 1") == {
        "history",
        "statistics",
    }
    assert referenced_tables("select * from states s, history h where s.entity_id = h.entity_id") == {
        "states",
        "history",
    }


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "select row_number() over (partition by entity_id order by when_ts) as rn from history order by rn",
            id="window-row-number",
        ),
        pytest.param("select substr(entity_id, 1, 6) as prefix from states", id="substr"),
        pytest.param("select group_concat(state) as states_seen from history", id="group-concat"),
        pytest.param("select nullif(state, 'missing') as value from states", id="nullif"),
        pytest.param("select iif(value is not null, 'numeric', 'other') as kind from states", id="iif"),
        pytest.param("select json_extract(attributes, '$.missing') as missing from states", id="json-extract"),
    ],
)
def test_home_db_runs_window_scalar_and_json_functions(sql: str) -> None:
    """SQLite read functions execute while the database remains read-only."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        db.load_history(
            [
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.0,
                },
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:05:00+00:00",
                    "state": "21",
                    "value": 21.0,
                },
            ]
        )
        result = db.execute(sql, 9999999999)
    finally:
        db.close()

    assert len(result.rows) >= 1
    assert result.truncated is False


def test_home_db_unknown_column_error_carries_fix() -> None:
    """Unknown SQL columns expose a stable key and concrete column candidates."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        with pytest.raises(HelperExecutionError) as err:
            db.execute("select no_such_column from states", 9999999999)
    finally:
        db.close()

    assert err.value.key == "sql_unknown_column"
    assert err.value.fix is not None
    assert len(err.value.fix) > 0
    assert "entity_id" in err.value.fix


def test_home_db_unknown_table_error_carries_fix() -> None:
    """Unknown SQL tables expose a stable key and concrete table candidates."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        with pytest.raises(HelperExecutionError) as err:
            db.execute("select * from no_such_table", 9999999999)
    finally:
        db.close()

    assert err.value.key == "sql_unknown_table"
    assert err.value.fix is not None
    assert len(err.value.fix) > 0
    assert "states" in err.value.fix


@pytest.mark.parametrize(
    "view_name",
    [
        pytest.param("states_meta", id="states-meta"),
        pytest.param("statistics_meta", id="statistics-meta"),
        pytest.param("statistics_short_term", id="statistics-short-term"),
        pytest.param("state_history", id="state-history"),
        pytest.param("long_term_statistics", id="long-term-statistics"),
    ],
)
def test_home_db_compat_views_queryable(view_name: str) -> None:
    """Recorder-schema compatibility views are queryable names."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        db.load_history(
            [
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.0,
                }
            ]
        )
        db.load_statistics(
            [
                {
                    "statistic_id": "sensor.temp",
                    "entity_id": "sensor.temp",
                    "when": "2026-01-01T00:00:00+00:00",
                    "mean": 20.0,
                    "min": 20.0,
                    "max": 20.0,
                }
            ]
        )
        result = db.execute(f"select * from {view_name}", 9999999999)
    finally:
        db.close()

    assert isinstance(result.rows, list)
    assert result.truncated is False


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("-- line comment\nselect 1 as one", id="line-comment"),
        pytest.param("/* block */ select 1 as one", id="block-comment"),
        pytest.param("/* multi\nline */ select 1 as one", id="multi-line-block-comment"),
    ],
)
def test_home_db_accepts_leading_comments(sql: str) -> None:
    """Leading SQL comments do not prevent read queries from executing."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        result = db.execute(sql, 9999999999)
    finally:
        db.close()

    assert result.rows == [{"one": 1}]
    assert result.truncated is False


def test_home_db_preserves_distinct_history_samples() -> None:
    """History deduplication keeps samples that differ by loaded value."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        db.load_history(
            [
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.0,
                },
                {
                    "entity_id": "sensor.temp",
                    "domain": "sensor",
                    "area_id": "area-main",
                    "floor_id": "floor-main",
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "20",
                    "value": 20.5,
                },
            ]
        )
        result = db.execute("select count(*) as c from history", 9999999999)
    finally:
        db.close()

    assert result.rows == [{"c": 2}]


def test_home_db_dedups_identical_history_rows_with_null_fields() -> None:
    """Byte-identical history rows (including NULL fields) collapse on re-load."""
    db = HomeDatabase(_snapshot())
    try:
        db.initialize()
        row = {
            "entity_id": "binary_sensor.motion",
            "domain": "binary_sensor",
            "area_id": None,
            "floor_id": None,
            "device_id": None,
            "when": "2026-01-01T00:00:00+00:00",
            "state": "on",
            "value": None,
        }
        # Loading the same row twice (e.g. a capped re-fetch) must not duplicate it,
        # even though SQLite unique indexes treat NULLs as distinct.
        db.load_history([row, row])
        result = db.execute("select count(*) as c from history", 9999999999)
    finally:
        db.close()

    assert result.rows == [{"c": 1}]
