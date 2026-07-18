from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.data.energy import EnergyPeriod, SafeEnergyCurrentPrice
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.energy import run_energy_query, validate_energy_args
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from homeassistant.helpers import llm
from llm_sandbox_evals.agent_runner import build_agent_tools
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import run_case
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.runtime import EvalRuntime, build_eval_runtime, build_fixture_energy_source
from llm_sandbox_evals.schema import EvalCase, ExpectedToolCall, RequestVariant
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
import pytest

from llm_sandbox_evals import agent_runner

_WINDOW = {
    "start": "2026-06-22T00:00:00Z",
    "end": "2026-06-29T00:00:00Z",
    "period": "day",
}
_DIRECT_ARGS = {
    **_WINDOW,
    "source_types": ["device"],
    "device_statistic_ids": ["sensor.storage_room_power"],
    "include": ["summary", "series"],
}



async def test_home_full_direct_energy_uses_production_core_and_redacts_private_config() -> None:
    runtime = _runtime("home_full")
    assert runtime.energy_source is not None

    result = await runtime.energy_tool.run_query(
        validate_energy_args({**_WINDOW, "include": ["summary", "series", "current", "forecast"]}),
        runtime.energy_source,
    )

    summary = cast(dict[str, object], result["summary"])
    electricity = cast(dict[str, dict[str, object]], summary["electricity"])
    assert electricity["home_consumption"] == {
        "value": 117.0,
        "unit": "kWh",
        "series": [
            [f"2026-06-{day:02d}T00:00:00+00:00", value]
            for day, value in zip(range(22, 29), (16.0, 18.0, 14.0, 20.0, 17.0, 14.0, 18.0), strict=True)
        ],
    }

    sources = cast(list[dict[str, object]], result["sources"])
    grid = next(source for source in sources if source["source_type"] == "grid")
    assert grid["current_rate"] == {"value": 0.042, "unit": "kW"}
    assert cast(list[dict[str, object]], grid["measures"])[0]["current_price"] == {
        "value": 0.28,
        "unit": "USD/kWh",
        "source": "fixed",
    }
    rate_series = cast(dict[str, object], grid["rate_series"])
    assert rate_series["unit"] == "kW"
    assert len(cast(list[list[object]], rate_series["points"])) == 7

    devices = cast(list[dict[str, object]], result["devices"])
    parent = next(device for device in devices if device["name"] == "Workshop circuit")
    child = next(device for device in devices if device["name"] == "Workshop tools")
    assert (parent["inclusive_value"], parent["exclusive_value"], parent["unit"]) == (30.0, 22.0, "kWh")
    assert (child["inclusive_value"], child["exclusive_value"], child["unit"]) == (8.0, 8.0, "kWh")
    cost = cast(list[dict[str, object]], summary["cost"])
    compensation = cast(list[dict[str, object]], summary["compensation"])
    assert (cost[0]["value"], cost[0]["unit"]) == (14.0, "USD")
    assert len(cast(list[list[object]], cost[0]["series"])) == 7
    assert (compensation[0]["value"], compensation[0]["unit"]) == (3.7, "USD")
    assert len(cast(list[dict[str, object]], result["forecast"])) == 1
    assert "forecast-config-private" not in json.dumps(result)

@pytest.mark.parametrize(
    ("period", "expected_start"),
    (("week", "2026-06-22T00:00:00+00:00"), ("month", "2026-06-01T00:00:00+00:00")),
)
async def test_fixture_energy_statistics_aggregate_calendar_periods(
    period: str, expected_start: str
) -> None:
    runtime = _runtime("home_full")
    assert runtime.energy_source is not None

    rows = await runtime.energy_source.fetch_statistics(
        {"sensor.storage_room_power", "sensor.server_room_power"},
        datetime.fromisoformat("2026-06-22T00:00:00+00:00"),
        datetime.fromisoformat("2026-06-29T00:00:00+00:00"),
        cast(EnergyPeriod, period),
        None,
        {"change", "mean"},
    )

    assert rows["sensor.storage_room_power"] == [
        {"start": expected_start, "change": 30.0}
    ]
    assert rows["sensor.server_room_power"] == [
        {"start": expected_start, "mean": pytest.approx(805.7142857142857)}
    ]


async def test_run_case_routes_canonical_energy_call_through_registered_agent_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The agent-facing tool validates and records the shared-core Energy response."""
    candidate = baseline_candidate()
    case = EvalCase(
        id="direct-energy-agent",
        home="home_full",
        category="test",
        requests=(RequestVariant("canonical", "Retrieve the workshop Energy trend."),),
        required_actions=(),
        oracle="tool_calls",
        expected_tool_calls=(ExpectedToolCall("get_energy", dict(_DIRECT_ARGS)),),
    )

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=_direct_energy_stream, model_name="energy-agent")

    monkeypatch.setattr(agent_runner, "make_model", make_model)
    trace = await run_case(
        candidate,
        "energy-agent",
        case,
        case.requests[0],
        EvalConfig(
            models=["energy-agent"],
            candidates=[candidate.id],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )

    assert trace.provider_error is None
    assert trace.outcome.state == "correct"
    assert trace.answer == "Energy complete."
    assert len(trace.tool_events) == 1
    event = trace.tool_events[0]
    assert event.tool_name == "get_energy"
    assert event.args == _DIRECT_ARGS
    assert event.output["summary"] == {}
    assert event.output["sources"] == []
    devices = cast(list[dict[str, object]], event.output["devices"])
    assert len(devices) == 1
    assert {
        "source_type": devices[0]["source_type"],
        "name": devices[0]["name"],
        "statistic_id": devices[0]["statistic_id"],
        "inclusive_value": devices[0]["inclusive_value"],
        "exclusive_value": devices[0]["exclusive_value"],
        "unit": devices[0]["unit"],
        "series": devices[0]["series"],
    } == {
        "source_type": "device",
        "name": "Workshop circuit",
        "statistic_id": "sensor.storage_room_power",
        "inclusive_value": 30.0,
        "exclusive_value": 22.0,
        "unit": "kWh",
        "series": [
            {
                "start": f"2026-06-{day:02d}T00:00:00+00:00",
                "inclusive_value": inclusive,
                "exclusive_value": exclusive,
            }
            for day, inclusive, exclusive in zip(
                range(22, 29),
                (4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 4.0),
                (3.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0),
                strict=True,
            )
        ],
    }
    serialized = json.dumps(event.output, sort_keys=True)
    for private_identifier in (
        "forecast-config-private",
        "sensor.external_raw_source",
        "sensor.raw_cost_private",
        "sensor.laundry_room_power",
    ):
        assert private_identifier not in serialized




@pytest.mark.parametrize(
    ("price_field", "measure_role", "price_state", "stat_energy_to", "expected_price"),
    [
        pytest.param(
            "entity_energy_price",
            "grid_import",
            "0.31",
            None,
            SafeEnergyCurrentPrice(0.31, "USD/kWh", "entity"),
            id="import",
        ),
        pytest.param(
            "entity_energy_price_export",
            "grid_export",
            "0.08",
            "sensor.balcony_power",
            SafeEnergyCurrentPrice(0.08, "USD/kWh", "entity"),
            id="export",
        ),
        pytest.param(
            "entity_energy_price",
            "grid_import",
            "unavailable",
            None,
            None,
            id="visible-nonnumeric",
        ),
    ],
)
async def test_fixture_energy_validation_price_sidecar_is_safe(
    price_field: str,
    measure_role: str,
    price_state: str,
    stat_energy_to: str | None,
    expected_price: SafeEnergyCurrentPrice | None,
) -> None:
    """Fixture validation retains only a source locator for visible price entities."""
    fixture = get_home("home_full")
    snapshot = apply_scope(cast(HomeSnapshot, fixture.snapshot()), EVAL_SCOPE)
    template = snapshot.states["sensor.balcony_power"]
    price_id = "sensor.eval_price_private"
    price = replace(
        template,
        entity_id=price_id,
        domain="sensor",
        object_id="eval_price_private",
        state=price_state,
        attributes={"unit_of_measurement": "USD/kWh"},
    )
    local_snapshot = replace(snapshot, states={**snapshot.states, price_id: price})
    raw = fixture.energy()
    preferences = cast(dict[str, object], raw["preferences"])
    sources = cast(list[dict[str, object]], preferences["energy_sources"])
    grid = {
        **sources[0],
        price_field: price_id,
        "stat_energy_to": stat_energy_to,
    }
    validation = {
        "energy_sources": [
            [{"type": "invalid_price", "affected_entities": [(price_id, None)]}],
            [],
        ],
        "device_consumption": [[], []],
        "device_consumption_water": [],
    }

    def energy() -> dict[str, object]:
        return {
            **raw,
            "preferences": {**preferences, "energy_sources": [grid, *sources[1:]]},
            "validation": validation,
        }

    local_fixture = cast(ModuleType, SimpleNamespace(energy=energy, recorder=fixture.recorder))
    source = build_fixture_energy_source(local_snapshot, local_fixture)
    assert source is not None

    current_price = next(
        measure.current_price
        for measure in source.catalog.sources[0].measures
        if measure.role == measure_role
    )
    mapped_validation = await source.fetch_validation()
    serialized = json.dumps({"catalog": source.catalog, "validation": mapped_validation}, default=str)

    assert current_price == expected_price
    assert mapped_validation == (
        {"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "grid:0"}]},
    )
    assert price_id not in serialized


async def test_fixture_price_validation_sidecars_are_group_scoped_and_safe() -> None:
    """Map each visible price ID only to its source's safe locator."""
    fixture = get_home("home_full")
    snapshot = apply_scope(cast(HomeSnapshot, fixture.snapshot()), EVAL_SCOPE)
    template = snapshot.states["sensor.balcony_power"]
    duplicate_price_id = "sensor.eval_duplicate_price"
    shared_price_id = "sensor.eval_shared_price"
    hidden_price_id = "sensor.eval_hidden_price"
    shared_price = replace(
        template,
        entity_id=shared_price_id,
        domain="sensor",
        object_id="eval_shared_price",
        state="0.31",
        attributes={"unit_of_measurement": "USD/kWh"},
    )
    duplicate_price = replace(
        shared_price,
        entity_id=duplicate_price_id,
        object_id="eval_duplicate_price",
        state="0.08",
    )
    local_snapshot = replace(
        snapshot,
        states={**snapshot.states, shared_price_id: shared_price, duplicate_price_id: duplicate_price},
    )
    raw = fixture.energy()
    preferences = cast(dict[str, object], raw["preferences"])
    sources = cast(list[dict[str, object]], preferences["energy_sources"])
    grid = {
        **sources[0],
        "entity_energy_price": shared_price_id,
        "entity_energy_price_export": duplicate_price_id,
    }
    gas = {
        "type": "gas",
        "name": "Metered gas",
        "stat_energy_from": "sensor.utility_room_power",
        "entity_energy_price": shared_price_id,
    }
    validation = {
        "energy_sources": [
            [
                {
                    "type": "invalid_price",
                    "affected_entities": [
                        (shared_price_id, None),
                        (duplicate_price_id, None),
                        (hidden_price_id, None),
                    ],
                }
            ],
            [],
            [
                {
                    "type": "invalid_price",
                    "affected_entities": [(shared_price_id, None), (hidden_price_id, None)],
                }
            ],
        ],
        "device_consumption": [[], []],
        "device_consumption_water": [],
    }

    def energy() -> dict[str, object]:
        return {
            **raw,
            "preferences": {**preferences, "energy_sources": [grid, sources[1], gas]},
            "validation": validation,
        }

    local_fixture = cast(ModuleType, SimpleNamespace(energy=energy, recorder=fixture.recorder))
    source = build_fixture_energy_source(local_snapshot, local_fixture)

    result = await run_energy_query(
        validate_energy_args({**_WINDOW, "include": ["validation"]}),
        source,
    )
    mapped_validation = cast(list[dict[str, object]], result["validation"])
    serialized = json.dumps(result)
    assert mapped_validation == [
        {"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "grid:0"}]},
        {"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "gas:0"}]},
    ]
    assert shared_price_id not in serialized
    assert duplicate_price_id not in serialized
    assert hidden_price_id not in serialized

async def test_energy_tool_registration_depends_on_fixture_configuration() -> None:
    full_tools = build_agent_tools(_runtime("home_full"))
    minimal_tools = build_agent_tools(_runtime("home_minimal"))

    assert "get_energy" in {tool.name for tool in full_tools}
    assert "get_energy" not in {tool.name for tool in minimal_tools}


async def test_execute_home_code_composes_energy_with_safe_state() -> None:
    runtime = _runtime("home_full")
    snapshot = runtime.snapshot
    assert runtime.energy_source is not None
    direct = await runtime.energy_tool.run_query(
        validate_energy_args({**_WINDOW, "include": ["summary", "series"]}),
        runtime.energy_source,
    )
    data = cast(
        dict[str, object],
        runtime.code_tool.parameters(
            {
                "code": (
                    "energy = await hass.energy(start='2026-06-22T00:00:00Z', "
                    "end='2026-06-29T00:00:00Z', period='day', include=['summary', 'series'])\n"
                    "state = states.get('sensor.balcony_power')\n"
                    "result = {'energy': energy, 'state': {'entity_id': state.entity_id, 'state': state.state}}"
                )
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
    output = cast(dict[str, object], result["output"])
    energy = cast(dict[str, object], output["energy"])
    electricity = cast(dict[str, dict[str, object]], cast(dict[str, object], energy["summary"])["electricity"])
    assert electricity["home_consumption"]["value"] == 117.0
    assert {"window", "period", "scope", "summary", "sources", "devices", "omissions"} <= set(energy)
    assert json.dumps(direct, sort_keys=True, separators=(",", ":")) == json.dumps(
        energy, sort_keys=True, separators=(",", ":")
    )
    assert output["state"] == {"entity_id": "sensor.balcony_power", "state": "42"}
    serialized = json.dumps(output, sort_keys=True)
    for private_identifier in (
        "forecast-config-private",
        "sensor.external_raw_source",
        "sensor.raw_cost_private",
        "sensor.laundry_room_power",
    ):
        assert private_identifier not in serialized


async def _direct_energy_stream(
    messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """Call get_energy once, then return a deterministic final response."""
    if any(
        isinstance(part, ToolCallPart)
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    ):
        yield "Energy complete."
        return
    tool_call = ToolCallPart(tool_name="get_energy", args=_DIRECT_ARGS, tool_call_id="energy-1")
    yield {
        0: DeltaToolCall(
            name="get_energy",
            json_args=tool_call.args_as_json_str(),
            tool_call_id=tool_call.tool_call_id,
        )
    }


def _runtime(home: str) -> EvalRuntime:
    fixture = get_home(home)
    snapshot = apply_scope(cast(HomeSnapshot, fixture.snapshot()), EVAL_SCOPE)
    case = EvalCase(
        id=f"energy-{home}",
        home=home,
        category="test",
        requests=(RequestVariant("canonical", "exercise Energy"),),
        required_actions=(),
    )
    return build_eval_runtime(
        case,
        baseline_candidate(),
        resolve_profile(DEFAULT_PROMPT_PROFILE),
        snapshot,
        fixture,
    )
