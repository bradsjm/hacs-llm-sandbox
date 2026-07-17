from typing import cast

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts.profiles import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.automation import GetAutomationTool
from custom_components.llm_sandbox.llm_api.tools.energy import GetEnergyTool
from custom_components.llm_sandbox.llm_api.tools.recorder import (
    GetHistoryTool,
    GetLogbookTool,
    GetStatisticsTool,
    RecorderSource,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from homeassistant.helpers import llm
from llm_sandbox_evals.agent_runner import _validate_recorder_tool
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.runtime import build_eval_runtime, build_fixture_recorder_source
from llm_sandbox_evals.schema import EvalCase, RequestVariant, RequiredAction
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
from voluptuous_openapi import convert


def _provider_schema(tool: llm.Tool) -> dict[str, object]:
    return cast(
        dict[str, object],
        convert(tool.parameters, custom_serializer=llm.selector_serializer),
    )


def test_provider_schemas_expose_canonical_tool_inputs() -> None:
    automation = cast(dict[str, dict[str, object]], _provider_schema(GetAutomationTool("eval"))["properties"])
    history = cast(dict[str, dict[str, object]], _provider_schema(GetHistoryTool("eval"))["properties"])
    energy = cast(dict[str, dict[str, object]], _provider_schema(GetEnergyTool("eval"))["properties"])
    statistics = cast(dict[str, dict[str, object]], _provider_schema(GetStatisticsTool("eval"))["properties"])
    logbook = cast(dict[str, dict[str, object]], _provider_schema(GetLogbookTool("eval"))["properties"])

    assert automation["query"]["type"] == "string"
    for properties in (automation, history, logbook):
        assert properties["entity_ids"]["type"] == "array"
        assert cast(dict[str, object], properties["entity_ids"]["items"])["type"] == "string"
    assert automation["include"]["type"] == "array"
    assert cast(dict[str, object], automation["include"]["items"])["enum"] == ["content", "runs"]
    assert statistics["statistic_ids"]["type"] == "array"
    assert cast(dict[str, object], statistics["statistic_ids"]["items"])["type"] == "string"
    assert energy["hours"]["type"] == "number"
    assert energy["hours"]["minimum"] == 0
    assert tuple(cast(tuple[str, ...], energy["period"]["enum"])) == (
        "auto",
        "5minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    )
    assert tuple(cast(tuple[str, ...], cast(dict[str, object], energy["source_types"]["items"])["enum"])) == (
        "grid",
        "solar",
        "battery",
        "gas",
        "water",
        "device",
        "device_water",
    )
    assert tuple(cast(tuple[str, ...], cast(dict[str, object], energy["include"]["items"])["enum"])) == (
        "summary",
        "series",
        "current",
        "forecast",
        "carbon",
        "validation",
    )
    assert tuple(cast(tuple[str, ...], energy["compare"]["enum"])) == ("previous", "year_over_year")
    assert "device_statistic_ids" in energy
    for field in ("start", "end"):
        assert energy[field]["type"] == "string"
        assert energy[field]["format"] == "date-time"

    for properties in (automation, history, statistics, logbook):
        assert properties["start"]["type"] == "string"
        assert properties["start"]["format"] == "date-time"
        assert properties["end"]["type"] == "string"
        assert properties["end"]["format"] == "date-time"

    bounded_arrays = (
        automation["entity_ids"],
        automation["include"],
        history["entity_ids"],
        history["attributes"],
        history["value_operations"],
        history["group_by"],
        statistics["statistic_ids"],
        statistics["types"],
        logbook["entity_ids"],
    )
    for field_schema in bounded_arrays:
        assert "minLength" not in field_schema
        assert "maxLength" not in field_schema


async def test_recorder_selector_no_match_returns_error_with_guidance() -> None:
    snapshot = _scoped_snapshot()
    tool = GetHistoryTool("eval")
    data = cast(dict[str, object], tool.parameters({"area_id": "area_missing", "hours": 24}))

    result = await tool.run_query(snapshot, data, _recorder_source(snapshot))

    assert result["status"] == "error"
    error = cast(dict[str, object], result["error"])
    assert error["key"] == "selector_no_match"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]
    assert "guidance" in error


def test_eval_recorder_validation_returns_invalid_tool_input_for_bad_iso() -> None:
    validation = _validate_recorder_tool(
        GetHistoryTool("eval"),
        {"entity_ids": ["light.bedroom"], "start": "not-a-date"},
    )

    assert validation.data == {}
    assert validation.error is not None
    error = cast(dict[str, object], validation.error["error"])
    assert validation.error["status"] == "error"
    assert error["key"] == "invalid_tool_input"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]


def test_eval_recorder_validation_omits_empty_cursor_before_cursor_handling() -> None:
    """Empty cursor placeholders follow production normalization semantics."""
    statistics = _validate_recorder_tool(
        GetStatisticsTool("eval"),
        {
            "statistic_ids": ["sensor.balcony_power"],
            "cursor": "",
        },
    )
    logbook = _validate_recorder_tool(
        GetLogbookTool("eval"),
        {
            "entity_ids": ["light.living_room_accent"],
            "cursor": "",
        },
    )

    assert statistics.error is None
    assert statistics.data["statistic_ids"] == ["sensor.balcony_power"]
    assert "cursor" not in statistics.data
    assert logbook.error is None
    assert logbook.data["entity_ids"] == ["light.living_room_accent"]
    assert "cursor" not in logbook.data


async def test_recorder_window_too_large_returns_stable_error_key() -> None:
    snapshot = _scoped_snapshot()
    tool = GetHistoryTool("eval")
    data = cast(dict[str, object], tool.parameters({"entity_ids": ["light.bedroom"], "hours": 10_000}))

    result = await tool.run_query(snapshot, data, _recorder_source(snapshot))

    assert result["status"] == "error"
    error = cast(dict[str, object], result["error"])
    assert error["key"] == "time_window_too_large"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]


async def test_eval_sql_query_can_filter_visible_entities_by_domain() -> None:
    result, _invoker_calls = await _run_execute(
        _case(),
        "result = await hass.query(\"select entity_id from states where domain = 'light' order by entity_id\")",
    )

    assert result["execution"] == {"status": "ok"}
    entity_ids = [row["entity_id"] for row in cast(list[dict[str, object]], result["output"])]
    assert entity_ids == ["light.bedroom", "light.living"]


async def _run_execute(case: EvalCase, code: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    fixture = get_home(case.home)
    snapshot = _scoped_snapshot()
    runtime = build_eval_runtime(
        case,
        baseline_candidate(),
        resolve_profile(DEFAULT_PROMPT_PROFILE),
        snapshot,
        fixture,
    )
    data = cast(dict[str, object], runtime.code_tool.parameters({"code": code}))

    result = cast(
        dict[str, object],
        await runtime.code_tool.run_execute(
            snapshot,
            data,
            llm.LLMContext("test", None, "en", None, None),
            runtime.runtime_context_factory(),
        ),
    )
    return result, runtime.invoker.calls


def _case() -> EvalCase:
    return EvalCase(
        id="tool-contract-unit",
        home="home_minimal",
        category="test",
        requests=(RequestVariant("canonical", "exercise production tool contract"),),
        required_actions=(RequiredAction("light", "turn_on", ("light.living",)),),
    )


def _scoped_snapshot(*, anchor_device_id: str | None = None) -> HomeSnapshot:
    snapshot = cast(HomeSnapshot, get_home("home_minimal").snapshot())
    return apply_scope(snapshot, EVAL_SCOPE, anchor_device_id=anchor_device_id)


def _recorder_source(snapshot: HomeSnapshot) -> RecorderSource:
    return build_fixture_recorder_source(snapshot, get_home("home_minimal"))
