from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import cast

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE, MAX_HISTORY_STATES, MAX_LOGBOOK_ENTRIES
from custom_components.llm_sandbox.llm_api.prompts.profiles import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, GetLogbookTool
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from homeassistant.helpers import llm
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.runtime import build_eval_runtime, build_fixture_recorder_source
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope

_CREATED_AT = datetime(2026, 6, 29, 12, tzinfo=UTC)
_LIVING_TEMP = "sensor.living_temp"
_LIVING_LIGHT = "light.living"


async def test_history_run_query_uses_production_aggregate_shape() -> None:
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, get_home("home_default"))
    tool = GetHistoryTool("eval")
    data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_TEMP], "hours": 24, "aggregate": "state_counts"}),
    )

    result = await tool.run_query(snapshot, data, source)

    assert result == {
        "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
        "mode": "state_counts",
        "summary": {_LIVING_TEMP: {"state_counts": {"24.4": 1, "24.9": 1, "25.2": 1}}},
    }


async def test_history_run_query_cursor_round_trip_uses_production_pagination() -> None:
    timestamps = _ascending_timestamps(MAX_HISTORY_STATES + 5)
    fixture = _fixture({"history": {_LIVING_TEMP: _history_rows(timestamps)}, "statistics": {}, "logbook": {}})
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetHistoryTool("eval")
    first_data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_TEMP], "start": timestamps[0], "end": timestamps[-1]}),
    )

    first_result = await tool.run_query(snapshot, first_data, source)

    first_rows = _history_result_rows(first_result, _LIVING_TEMP)
    assert len(first_rows) == MAX_HISTORY_STATES
    assert first_rows[0] == [timestamps[5], "5"]
    cursor = cast(str, first_result["next_cursor"])
    second_data = cast(dict[str, object], tool.parameters({"entity_ids": [_LIVING_TEMP], "cursor": cursor}))
    second_result = await tool.run_query(snapshot, second_data, source)
    assert _history_result_rows(second_result, _LIVING_TEMP) == [
        [timestamp, str(index)] for index, timestamp in enumerate(timestamps[:5])
    ]
    assert "next_cursor" not in second_result


async def test_logbook_source_injects_entity_id_and_production_paginates() -> None:
    timestamps = _ascending_timestamps(MAX_LOGBOOK_ENTRIES + 3)
    fixture = _fixture({"history": {}, "statistics": {}, "logbook": {_LIVING_LIGHT: _logbook_rows(timestamps)}})
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetLogbookTool("eval")
    data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_LIGHT], "start": timestamps[0], "end": timestamps[-1]}),
    )

    result = await tool.run_query(snapshot, data, source)

    entries = cast(list[dict[str, object]], result["entries"])
    assert len(entries) == MAX_LOGBOOK_ENTRIES
    assert {entry["entity_id"] for entry in entries} == {_LIVING_LIGHT}
    assert entries[0]["when"] == timestamps[3]


async def test_execute_home_code_runs_with_eval_runtime_context() -> None:
    case = _case()
    fixture = get_home("home_default")
    snapshot = apply_scope(_snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
    runtime = build_eval_runtime(
        case, baseline_candidate(), resolve_profile(DEFAULT_PROMPT_PROFILE), snapshot, fixture
    )
    data = cast(
        dict[str, object],
        runtime.code_tool.parameters(
            {
                "code": "result = await hass.query(\"select entity_id, state from states where entity_id = 'light.living'\")"
            }
        ),
    )

    result = await runtime.code_tool.run_execute(
        snapshot,
        data,
        llm.LLMContext("test", None, "en", None, None),
        runtime.runtime_context_factory(),
    )

    assert result["execution"] == {"status": "ok"}
    assert result["output"] == [{"entity_id": "light.living", "state": "on"}]


def _case() -> EvalCase:
    return EvalCase(
        id="production-core-unit",
        category="unit",
        home="home_default",
        user_request="exercise production core",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(),
    )


def _snapshot() -> HomeSnapshot:
    return cast(HomeSnapshot, get_home("home_default").snapshot())


def _fixture(recorder_data: dict[str, object]) -> ModuleType:
    module = ModuleType("fixture_home")

    def recorder() -> dict[str, object]:
        return recorder_data

    module.__dict__["recorder"] = recorder
    return module


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


def _logbook_rows(timestamps: list[str]) -> list[dict[str, object]]:
    return [
        {"when": timestamp, "name": "Living Room Light", "message": f"changed state {index}"}
        for index, timestamp in enumerate(timestamps)
    ]


def _history_result_rows(result: dict[str, object], entity_id: str) -> list[list[object]]:
    entities = cast(dict[str, dict[str, object]], result["entities"])
    return cast(list[list[object]], entities[entity_id]["rows"])
