"""Behavioral tests for the bounded home SQLite database."""

import pytest
from custom_components.llm_sandbox.llm_api.errors import HelperExecutionError
from custom_components.llm_sandbox.llm_api.home_db import HomeDatabase, referenced_tables

from tests.components.llm_sandbox.llm_api.tools.test_analytics import _snapshot


def test_home_db_queries_visible_states_and_json_attributes() -> None:
    """Read-only SQL can query the snapshot states table and JSON attributes."""
    db = HomeDatabase(_snapshot())
    try:
        db.load_states()
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
        db.load_states()
        with pytest.raises(HelperExecutionError) as err:
            db.execute("delete from states", 9999999999)
    finally:
        db.close()

    assert err.value.key == "sql_read_only"


def test_home_db_can_load_history_after_prior_user_query() -> None:
    """A prior read-only user query does not prevent later trusted lazy loads."""
    db = HomeDatabase(_snapshot())
    try:
        db.load_states()
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
        db.load_states()
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
