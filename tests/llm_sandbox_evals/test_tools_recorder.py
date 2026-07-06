from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import cast

import pytest
from custom_components.llm_sandbox.const import (
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_STATISTICS_ROWS,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected, ToolOutcome

from llm_sandbox_evals import tools as eval_tools

_CREATED_AT = datetime(2026, 6, 29, 12, tzinfo=UTC)
_LIVING_TEMP = "sensor.living_temp"
_LIVING_LIGHT = "light.living"


@pytest.mark.parametrize(
    ("aggregate", "expected_summary"),
    [
        pytest.param("count_transitions", {"transitions": 2}, id="count-transitions"),
        pytest.param("state_counts", {"state_counts": {"24.4": 1, "24.9": 1, "25.2": 1}}, id="state-counts"),
    ],
)
def test_history_aggregate_envelopes_match_production_shape(
    aggregate: str, expected_summary: dict[str, object]
) -> None:
    outcome = eval_tools._run_history(
        {"entity_ids": [_LIVING_TEMP], "hours": 24, "aggregate": aggregate},
        _case(TOOL_GET_HISTORY),
        _snapshot(),
    )

    assert _result(outcome) == {
        "window": {
            "start": "2026-06-28T12:00:00+00:00",
            "end": "2026-06-29T12:00:00+00:00",
        },
        "mode": aggregate,
        "summary": {_LIVING_TEMP: expected_summary},
    }


def test_history_raw_cursor_round_trip_returns_next_older_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = _ascending_timestamps(MAX_HISTORY_STATES + 205)
    _stub_recorder(
        monkeypatch,
        {"history": {_LIVING_TEMP: _history_rows(timestamps)}, "statistics": {}, "logbook": {}},
    )
    first_result = _result(
        eval_tools._run_history(
            {"entity_ids": [_LIVING_TEMP], "start": timestamps[0], "end": timestamps[-1]},
            _case(TOOL_GET_HISTORY),
            _snapshot(),
        )
    )

    first_rows = _history_result_rows(first_result, _LIVING_TEMP)
    assert len(first_rows) == MAX_HISTORY_STATES
    assert first_rows[0] == [timestamps[205], "205"]
    assert first_rows[-1] == [timestamps[-1], "1204"]
    cursor = cast(str, first_result["next_cursor"])

    second_result = _result(
        eval_tools._run_history(
            {"entity_ids": [_LIVING_TEMP], "cursor": cursor},
            _case(TOOL_GET_HISTORY),
            _snapshot(),
        )
    )
    second_rows = _history_result_rows(second_result, _LIVING_TEMP)
    assert len(second_rows) == 205
    assert second_rows[0] == [timestamps[0], "0"]
    assert second_rows[-1] == [timestamps[204], "204"]
    assert second_rows[-1][0] < first_rows[0][0]
    assert "next_cursor" not in second_result


def test_history_attributes_are_opt_in() -> None:
    with_attributes = _result(
        eval_tools._run_history(
            {"entity_ids": [_LIVING_TEMP], "attributes": ["unit_of_measurement"]},
            _case(TOOL_GET_HISTORY),
            _snapshot(),
        )
    )
    rows_with_attributes = _history_result_rows(with_attributes, _LIVING_TEMP)
    assert rows_with_attributes == [["2026-06-29T12:00:00+00:00", "25.2", {"unit_of_measurement": "°C"}]]

    without_attributes = _result(
        eval_tools._run_history({"entity_ids": [_LIVING_TEMP]}, _case(TOOL_GET_HISTORY), _snapshot())
    )
    rows_without_attributes = _history_result_rows(without_attributes, _LIVING_TEMP)
    assert rows_without_attributes == [["2026-06-29T12:00:00+00:00", "25.2"]]


def test_history_relative_hours_window_uses_snapshot_created_at() -> None:
    result = _result(
        eval_tools._run_history({"entity_ids": [_LIVING_TEMP], "hours": 12}, _case(TOOL_GET_HISTORY), _snapshot())
    )

    assert result["window"] == {
        "start": "2026-06-29T00:00:00+00:00",
        "end": "2026-06-29T12:00:00+00:00",
    }


@pytest.mark.parametrize(
    ("tool_args", "expected_key"),
    [
        pytest.param({"entity_ids": [_LIVING_TEMP], "hours": 999}, "time_window_too_large", id="too-large"),
        pytest.param(
            {
                "entity_ids": [_LIVING_TEMP],
                "start": "2026-06-29T12:00:01+00:00",
                "end": "2026-06-29T12:00:00+00:00",
            },
            "invalid_tool_input",
            id="start-after-end",
        ),
    ],
)
def test_history_window_errors_return_recoverable_envelopes(tool_args: dict[str, object], expected_key: str) -> None:
    result = _result(eval_tools._run_history(tool_args, _case(TOOL_GET_HISTORY), _snapshot()))
    error = cast(dict[str, object], result["error"])

    assert result["status"] == "error"
    assert error["key"] == expected_key


def test_statistics_cursor_round_trip_preserves_period_and_returns_older_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = _ascending_timestamps(MAX_STATISTICS_ROWS + 205)
    _stub_recorder(
        monkeypatch,
        {"history": {}, "statistics": {_LIVING_TEMP: _statistics_rows(timestamps)}, "logbook": {}},
    )
    first_result = _result(
        eval_tools._run_statistics(
            {
                "statistic_ids": [_LIVING_TEMP],
                "start": timestamps[0],
                "end": timestamps[-1],
                "period": "5minute",
                "types": ["sum"],
            },
            _case(TOOL_GET_STATISTICS),
            _snapshot(),
        )
    )

    first_rows = _statistics_result_rows(first_result, _LIVING_TEMP)
    assert first_result["period"] == "5minute"
    assert len(first_rows) == MAX_STATISTICS_ROWS
    assert first_rows[0] == [timestamps[205], {"sum": 205.0}]
    assert first_rows[-1] == [timestamps[-1], {"sum": 1204.0}]
    cursor = cast(str, first_result["next_cursor"])

    second_result = _result(
        eval_tools._run_statistics(
            {"statistic_ids": [_LIVING_TEMP], "cursor": cursor},
            _case(TOOL_GET_STATISTICS),
            _snapshot(),
        )
    )
    second_rows = _statistics_result_rows(second_result, _LIVING_TEMP)
    assert second_result["period"] == "5minute"
    assert len(second_rows) == 205
    assert second_rows[0] == [timestamps[0], {"sum": 0.0}]
    assert second_rows[-1] == [timestamps[204], {"sum": 204.0}]
    assert second_rows[-1][0] < first_rows[0][0]
    assert "next_cursor" not in second_result


def test_logbook_cursor_round_trip_returns_next_older_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = _ascending_timestamps(MAX_LOGBOOK_ENTRIES + 35)
    _stub_recorder(
        monkeypatch,
        {"history": {}, "statistics": {}, "logbook": {_LIVING_LIGHT: _logbook_rows(timestamps)}},
    )
    first_result = _result(
        eval_tools._run_logbook(
            {"entity_ids": [_LIVING_LIGHT], "start": timestamps[0], "end": timestamps[-1]},
            _case(TOOL_GET_LOGBOOK),
            _snapshot(),
        )
    )

    first_entries = _logbook_entries(first_result)
    assert len(first_entries) == MAX_LOGBOOK_ENTRIES
    assert first_entries[0]["when"] == timestamps[35]
    assert first_entries[-1]["when"] == timestamps[-1]
    assert {entry["entity_id"] for entry in first_entries} == {_LIVING_LIGHT}
    cursor = cast(str, first_result["next_cursor"])

    second_result = _result(
        eval_tools._run_logbook(
            {"entity_ids": [_LIVING_LIGHT], "cursor": cursor},
            _case(TOOL_GET_LOGBOOK),
            _snapshot(),
        )
    )
    second_entries = _logbook_entries(second_result)
    assert len(second_entries) == 35
    assert second_entries[0]["when"] == timestamps[0]
    assert second_entries[-1]["when"] == timestamps[34]
    assert second_entries[-1]["when"] < first_entries[0]["when"]
    assert {entry["entity_id"] for entry in second_entries} == {_LIVING_LIGHT}
    assert "next_cursor" not in second_result


def _case(tool_name: str) -> EvalCase:
    return EvalCase(
        id=f"{tool_name}-unit",
        category="unit",
        home="home_default",
        user_request="exercise recorder emulator",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(tool_name=tool_name),
        par_turns=1,
    )


def _snapshot() -> HomeSnapshot:
    # ``get_home`` returns a ``ModuleType`` whose ``snapshot()`` is dynamically typed; cast for mypy.
    return cast(HomeSnapshot, get_home("home_default").snapshot())


def _result(outcome: ToolOutcome) -> dict[str, object]:
    assert outcome.ok is True
    assert outcome.error is None
    assert outcome.result is not None
    return outcome.result


def _ascending_timestamps(count: int) -> list[str]:
    start = _CREATED_AT - timedelta(minutes=count - 1)
    return [(start + timedelta(minutes=index)).isoformat() for index in range(count)]


def _history_rows(timestamps: list[str]) -> list[dict[str, object]]:
    return [
        {
            "state": str(index),
            "attributes": {"unit_of_measurement": "°C"},
            "last_changed": timestamp,
            "last_updated": timestamp,
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _statistics_rows(timestamps: list[str]) -> list[dict[str, object]]:
    return [
        {
            "start": timestamp,
            "end": (datetime.fromisoformat(timestamp) + timedelta(minutes=1)).isoformat(),
            "state": float(index),
            "sum": float(index),
            "min": float(index),
            "max": float(index),
            "mean": float(index),
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _logbook_rows(timestamps: list[str]) -> list[dict[str, object]]:
    return [
        {"when": timestamp, "name": "Living Room Light", "message": f"changed state {index}"}
        for index, timestamp in enumerate(timestamps)
    ]


def _stub_recorder(monkeypatch: pytest.MonkeyPatch, recorder_data: dict[str, object]) -> None:
    # ModuleType permits arbitrary attributes at runtime; populate via ``__dict__`` so mypy sees a typed mapping.
    module = ModuleType("oversize_home")

    def recorder() -> dict[str, object]:
        return recorder_data

    module.__dict__["recorder"] = recorder

    def get_home_stub(name: str) -> ModuleType:
        return module

    monkeypatch.setattr(eval_tools, "get_home", get_home_stub)


def _history_result_rows(result: dict[str, object], entity_id: str) -> list[list[object]]:
    entities = cast(dict[str, dict[str, object]], result["entities"])
    return cast(list[list[object]], entities[entity_id]["rows"])


def _statistics_result_rows(result: dict[str, object], statistic_id: str) -> list[list[object]]:
    statistics = cast(dict[str, dict[str, object]], result["statistics"])
    return cast(list[list[object]], statistics[statistic_id]["rows"])


def _logbook_entries(result: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], result["entries"])
