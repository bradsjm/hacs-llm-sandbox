"""End-to-end tests for the Monty executor with HA-native facades."""

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import pytest
from custom_components.llm_sandbox.const import CONF_ACTIONS_ENABLED, DEFAULT_PROMPT_PROFILE, TOOL_EXECUTE_HOME_CODE
from custom_components.llm_sandbox.llm_api import executor
from custom_components.llm_sandbox.llm_api.data.home_db import MAX_HISTORY_LOAD_ROWS
from custom_components.llm_sandbox.llm_api.errors import HelperExecutionError
from custom_components.llm_sandbox.llm_api.executor_support import ExecutionState
from custom_components.llm_sandbox.llm_api.facades import (
    SafeHass as SandboxHass,
)
from custom_components.llm_sandbox.llm_api.facades import build_facades
from custom_components.llm_sandbox.llm_api.prompts.profiles import resolve_profile
from custom_components.llm_sandbox.llm_api.sandbox_context import RuntimeContext, activate_runtime, clear_runtime
from custom_components.llm_sandbox.llm_api.tools.code import ExecuteHomeCodeTool
from custom_components.llm_sandbox.llm_api.tools.recorder import MAX_HISTORY_STATES, MAX_RECORDER_ENTITY_IDS
from custom_components.llm_sandbox.runtime import SandboxSettings
from custom_components.llm_sandbox.snapshot.models import DEFAULT_SCOPE, HomeSnapshot, SafeAreaEntry, SnapshotIndexes
from custom_components.llm_sandbox.types import ProposedAction
from homeassistant.core import Context, HomeAssistant, SupportsResponse
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .test_analytics import _snapshot


async def _run_code(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    code: str,
) -> dict[str, object]:
    """Run Monty code through the execute_home_code tool path."""
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    tool = ExecuteHomeCodeTool(entry.entry_id)
    tool_input = llm.ToolInput(id="", tool_name=TOOL_EXECUTE_HOME_CODE, tool_args={"code": code})
    return cast(dict[str, object], await tool.async_call(hass, tool_input, llm_context))


def _history_facade(
    snapshot: HomeSnapshot,
    fetch_history: Callable[[Sequence[str], datetime, datetime], Awaitable[list[dict[str, object]]]],
    fetch_statistics: Callable[[Sequence[str], datetime, datetime], Awaitable[list[dict[str, object]]]] | None = None,
    fetch_short_term_statistics: Callable[[Sequence[str], datetime, datetime], Awaitable[list[dict[str, object]]]]
    | None = None,
) -> tuple[SandboxHass, RuntimeContext]:
    """Activate a SafeHass facade with test runtime seams."""

    async def _invoke(_action: ProposedAction) -> object:
        return None

    async def _fetch_statistics(
        _entity_ids: Sequence[str], _start: datetime, _end: datetime
    ) -> list[dict[str, object]]:
        return []

    async def _run_blocking(fn: Callable[[], object]) -> object:
        return fn()

    runtime = RuntimeContext(
        state=ExecutionState(),
        settings=SandboxSettings(
            execution_timeout_seconds=10,
            helper_call_budget=20,
            scope=DEFAULT_SCOPE,
            actions_enabled=False,
            action_domains=frozenset(),
            prompt_profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
        ),
        invoke=_invoke,
        fetch_history=fetch_history,
        fetch_statistics=fetch_statistics or _fetch_statistics,
        fetch_short_term_statistics=fetch_short_term_statistics or _fetch_statistics,
        run_blocking=_run_blocking,
    )
    clear_runtime()
    activate_runtime(runtime, snapshot)
    return cast(SandboxHass, build_facades(snapshot)["hass"]), runtime


async def test_read_state_and_registry_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify Monty can read states and registries through HA-native facades."""
    code = """
bedroom = area_registry.async_get_area_by_name("Bedroom")
er = entity_registry
bedroom_states = []
if bedroom is not None:
    bedroom_entries = er.async_entries_for_area(er.async_get(hass), bedroom.id)
    for entry in bedroom_entries:
        if entry.entity_id.split(".")[0] != "light":
            continue
        st = hass.states.get(entry.entity_id)
        if st is not None:
            bedroom_states.append({
                "entity_id": entry.entity_id,
                "state": st.state,
                "attributes": {k: v for k, v in st.attributes.items() if k in ("friendly_name", "brightness")},
            })
result = bedroom_states
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == [
        {
            "entity_id": "light.bedroom",
            "state": "on",
            "attributes": {"friendly_name": "Bedroom Light"},
        }
    ]


async def test_state_machine_sugar_subscript_and_contains(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify optional subscript sugar works alongside strict methods."""
    code = """
result = {
    "get": hass.states.get("light.bedroom").state,
    "subscript": states["light.bedroom"].state,
    "contains": "light.bedroom" in states,
    "len": len(states),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["get"] == "on"
    assert output["subscript"] == "on"
    assert output["contains"] is True
    assert output["len"] >= 2


async def test_snapshot_records_and_llm_context_support_read_only_mapping_reads(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Monty can use mapping reads over records and the request context without mutation APIs."""
    result = await _run_code(
        hass,
        loaded_entry,
        """
state = hass.states.get("light.bedroom")
result = {
    "state_get": state.get("state"),
    "state_keys": "attributes" in state.keys(),
    "state_items": state.items()[0][0],
    "state_values": "on" in state.values(),
    "context_id": llm_context.context.get("id"),
    "context_keys": "device_id" in llm_context.keys(),
    "context_items": llm_context.items()[0][0],
    "context_values": "test" in llm_context.values(),
    "mutation_api": hasattr(state, "update") or hasattr(llm_context, "update"),
}
""",
    )

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert isinstance(output, dict)
    assert output["state_get"] == "on"
    assert output["state_keys"] is True
    assert output["state_items"] == "entity_id"
    assert output["state_values"] is True
    assert isinstance(output["context_id"], str)
    assert output["context_keys"] is True
    assert output["context_items"] == "platform"
    assert output["context_values"] is True
    assert output["mutation_api"] is False


async def test_print_only_execution_keeps_null_output(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Print-only code keeps its result distinct from captured output lines."""
    result = await _run_code(hass, loaded_entry, 'print("Bedroom light is on")')

    assert result["execution"]["status"] == "ok"
    assert result["output"] is None
    assert result["printed"] == ["Bedroom light is on"]


async def test_hass_query_reads_visible_states(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """hass.query exposes bounded read-only SQL over the visible snapshot states table."""
    result = await _run_code(
        hass,
        loaded_entry,
        """
result = await hass.query("select entity_id, state from states where entity_id = 'light.bedroom'")
""",
    )

    assert result["execution"]["status"] == "ok"
    assert result["output"] == [{"entity_id": "light.bedroom", "state": "on"}]


async def test_hass_query_rejects_writes(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """hass.query reports a helper error for non-read SQL."""
    result = await _run_code(hass, loaded_entry, "result = await hass.query('delete from states')")

    assert result["execution"]["status"] == "helper_error"
    assert result["execution"]["kind"] == "sql_read_only"


async def test_hass_history_recorder_unavailable_is_helper_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorder-backed facade helpers map unavailable recorder to a stable helper key."""
    monkeypatch.setattr("custom_components.llm_sandbox.llm_api.tools.code.recorder_available", lambda _hass: False)

    result = await _run_code(hass, loaded_entry, "result = await hass.history(entity_ids='light.bedroom')")

    assert result["execution"]["status"] == "helper_error"
    assert result["execution"]["kind"] == "recorder_unavailable"


async def test_hass_query_loads_additional_history_scope_in_same_run() -> None:
    """A second SQL history query with a different visible scope loads additional rows."""
    base = _snapshot()
    temp_state = base.states["sensor.temp"]
    snapshot = replace(
        base,
        states=base.states
        | {"sensor.other": replace(temp_state, entity_id="sensor.other", object_id="other", state="30")},
    )
    fetch_calls: list[tuple[str, ...]] = []

    async def _fetch_history(entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        fetch_calls.append(tuple(entity_ids))
        return [
            {
                "entity_id": entity_id,
                "domain": "sensor",
                "area_id": None,
                "floor_id": None,
                "device_id": None,
                "when": "2026-01-01T00:00:00+00:00",
                "state": "20" if entity_id == "sensor.temp" else "30",
                "value": 20.0 if entity_id == "sensor.temp" else 30.0,
            }
            for entity_id in entity_ids
        ]

    hass_facade, _runtime = _history_facade(snapshot, _fetch_history)
    try:
        first = await hass_facade.query("select count(*) as count from \"history\" where entity_id = 'sensor.temp'")
        second = await hass_facade.query("select count(*) as count from main.history where entity_id = 'sensor.other'")
    finally:
        clear_runtime()

    assert first == [{"count": 1}]
    assert second == [{"count": 1}]
    assert fetch_calls == [("sensor.temp",), ("sensor.other",)]


async def test_hass_query_literal_scope_appends_transparency_note() -> None:
    """Literal-inferred query scope is surfaced so partial-scope results are not silent."""
    snapshot = _snapshot()

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return [
            {
                "entity_id": "sensor.temp",
                "domain": "sensor",
                "area_id": None,
                "floor_id": None,
                "device_id": None,
                "when": "2026-01-01T00:00:00+00:00",
                "state": "20",
                "value": 20.0,
            }
        ]

    hass_facade, runtime = _history_facade(snapshot, _fetch_history)
    try:
        await hass_facade.query("select count(*) as c from history where entity_id = 'sensor.temp'")
    finally:
        clear_runtime()

    assert len(runtime.state.notes) == 1
    assert "sensor.temp" in runtime.state.notes[0]


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param('select count(*) as c from history where "sensor.temp" = entity_id', id="double-quoted"),
        pytest.param("-- sensor.temp\nselect count(*) as c from history", id="line-comment"),
        pytest.param("/* sensor.temp */ select count(*) as c from history", id="block-comment"),
    ],
)
async def test_hass_query_scope_ignores_identifiers_and_comments(sql: str) -> None:
    """Only single-quoted SQL string literals infer recorder scope."""

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return []

    hass_facade, _runtime = _history_facade(_snapshot(), _fetch_history)
    try:
        with pytest.raises(HelperExecutionError) as err:
            await hass_facade.query(sql)
    finally:
        clear_runtime()

    assert err.value.helper == "query"
    assert err.value.key == "invalid_tool_input"


async def test_hass_query_loads_statistics_tables_independently() -> None:
    """Hourly and short-term statistics SQL references call their distinct fetch seams."""
    stats_calls: list[tuple[str, ...]] = []
    short_calls: list[tuple[str, ...]] = []

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return []

    async def _fetch_statistics(
        entity_ids: Sequence[str], _start: datetime, _end: datetime
    ) -> list[dict[str, object]]:
        stats_calls.append(tuple(entity_ids))
        return [
            {
                "statistic_id": entity_id,
                "when": "2026-01-01T00:00:00+00:00",
                "mean": 20.0,
            }
            for entity_id in entity_ids
        ]

    async def _fetch_short_term_statistics(
        entity_ids: Sequence[str], _start: datetime, _end: datetime
    ) -> list[dict[str, object]]:
        short_calls.append(tuple(entity_ids))
        return [
            {
                "statistic_id": entity_id,
                "when": "2026-01-01T00:05:00+00:00",
                "mean": 20.5,
            }
            for entity_id in entity_ids
        ]

    hass_facade, _runtime = _history_facade(
        _snapshot(),
        _fetch_history,
        fetch_statistics=_fetch_statistics,
        fetch_short_term_statistics=_fetch_short_term_statistics,
    )
    try:
        result = await hass_facade.query(
            """
            select 'hour' as table_name, mean from statistics where statistic_id = 'sensor.temp'
            union all
            select 'short' as table_name, mean from statistics_short_term where statistic_id = 'sensor.temp'
            order by table_name
            """
        )
    finally:
        clear_runtime()

    assert result == [{"table_name": "hour", "mean": 20.0}, {"table_name": "short", "mean": 20.5}]
    assert stats_calls == [("sensor.temp",)]
    assert short_calls == [("sensor.temp",)]


async def test_hass_query_capped_history_load_does_not_mark_window_complete() -> None:
    """A capped SQL history load stays conservative so later narrower scopes can fetch missing rows."""
    base = _snapshot()
    temp_state = base.states["sensor.temp"]
    snapshot = replace(
        base,
        states=base.states
        | {"sensor.other": replace(temp_state, entity_id="sensor.other", object_id="other", state="30")},
    )
    fetch_calls: list[tuple[str, ...]] = []

    async def _fetch_history(entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        fetch_calls.append(tuple(entity_ids))
        if tuple(entity_ids) == ("sensor.other",):
            return [
                {
                    "entity_id": "sensor.other",
                    "domain": "sensor",
                    "area_id": None,
                    "floor_id": None,
                    "device_id": None,
                    "when": "2026-01-01T00:00:00+00:00",
                    "state": "30",
                    "value": 30.0,
                }
            ]
        return [
            {
                "entity_id": "sensor.temp",
                "domain": "sensor",
                "area_id": None,
                "floor_id": None,
                "device_id": None,
                "when": f"2026-01-01T{index // 3600:02d}:{(index // 60) % 60:02d}:{index % 60:02d}+00:00",
                "state": str(index),
                "value": float(index),
            }
            for index in range(MAX_HISTORY_LOAD_ROWS + 1)
        ]

    hass_facade, runtime = _history_facade(snapshot, _fetch_history)
    try:
        await hass_facade.query("select count(*) as count from history", entity_ids=["sensor.other", "sensor.temp"])
        second = await hass_facade.query("select count(*) as count from history where entity_id = 'sensor.other'")
    finally:
        clear_runtime()

    assert second == [{"count": 1}]
    assert fetch_calls == [("sensor.other", "sensor.temp"), ("sensor.other",)]
    assert runtime.state.notes == [
        f"history load capped at {MAX_HISTORY_LOAD_ROWS} rows",
        "query scope inferred from SQL literals: sensor.other",
    ]


async def test_hass_history_raw_caps_rows_and_adds_note() -> None:
    """Raw hass.history output is bounded and reports the cap transparently."""
    snapshot = _snapshot()
    rows = [
        {
            "entity_id": "sensor.temp",
            "when": f"2026-01-01T00:{index % 60:02d}:00+00:00",
            "state": str(index),
            "value": float(index),
        }
        for index in range(MAX_HISTORY_STATES + 1)
    ]

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return rows

    hass_facade, runtime = _history_facade(snapshot, _fetch_history)
    try:
        result = await hass_facade.history(entity_ids="sensor.temp")
    finally:
        clear_runtime()

    assert isinstance(result, dict)
    assert len(result["rows"]) == MAX_HISTORY_STATES
    assert result["overflow"] == {
        "truncated": True,
        "limit": MAX_HISTORY_STATES,
        "returned": MAX_HISTORY_STATES,
        "omitted": 1,
    }
    assert runtime.state.notes == [f"history result capped at {MAX_HISTORY_STATES} rows"]


async def test_hass_history_analytics_error_uses_helper_key() -> None:
    """Analytics validation errors inside hass.history surface as helper errors."""

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return []

    hass_facade, _runtime = _history_facade(_snapshot(), _fetch_history)
    try:
        with pytest.raises(HelperExecutionError) as err:
            await hass_facade.history(entity_ids="sensor.temp", aggregate="on_duration", bucket="bad")
    finally:
        clear_runtime()

    assert err.value.helper == "history"
    assert err.value.key == "analytics_bad_bucket"


async def test_hass_query_clamp_error_uses_helper_key() -> None:
    """Recorder window clamp errors inside hass.query surface as helper errors."""

    async def _fetch_history(_entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        return []

    hass_facade, _runtime = _history_facade(_snapshot(), _fetch_history)
    try:
        with pytest.raises(HelperExecutionError) as err:
            await hass_facade.query("select * from history", hours=10_000)
    finally:
        clear_runtime()

    assert err.value.helper == "query"
    assert err.value.key == "time_window_too_large"


async def test_hass_query_scopes_history_by_area() -> None:
    """Area-scoped SQL history queries load only the area's visible entities."""
    base = _snapshot()
    state = base.states["sensor.temp"]
    area_id = "area-main"
    area_entity_ids = ("sensor.area_0", "sensor.area_1")
    states = {
        f"sensor.area_{index}": replace(
            state,
            entity_id=f"sensor.area_{index}",
            object_id=f"area_{index}",
            area_id=area_id,
        )
        for index in range(2)
    } | {
        f"sensor.other_{index}": replace(
            state,
            entity_id=f"sensor.other_{index}",
            object_id=f"other_{index}",
            area_id="area-other",
        )
        for index in range(MAX_RECORDER_ENTITY_IDS)
    }
    snapshot = replace(
        base,
        states=states,
        areas={
            area_id: SafeAreaEntry(
                id=area_id,
                area_id=area_id,
                name="Main Area",
                aliases=(),
                floor_id=None,
                labels=(),
                icon=None,
                picture=None,
                humidity_entity_id=None,
                temperature_entity_id=None,
                created_at=None,
                modified_at=None,
            )
        },
        indexes=SnapshotIndexes(
            {},
            {
                area_id: area_entity_ids,
                "area-other": tuple(f"sensor.other_{index}" for index in range(MAX_RECORDER_ENTITY_IDS)),
            },
            {},
            {},
            {},
            {},
            {},
        ),
    )
    fetch_calls: list[tuple[str, ...]] = []

    async def _fetch_history(entity_ids: Sequence[str], _start: datetime, _end: datetime) -> list[dict[str, object]]:
        fetch_calls.append(tuple(entity_ids))
        return [
            {
                "entity_id": entity_id,
                "domain": "sensor",
                "area_id": area_id,
                "floor_id": None,
                "device_id": None,
                "when": "2026-01-01T00:00:00+00:00",
                "state": "20",
                "value": 20.0,
            }
            for entity_id in entity_ids
        ]

    hass_facade, _runtime = _history_facade(snapshot, _fetch_history)
    try:
        result = await hass_facade.query("select count(*) as c from history", area_id=area_id)
    finally:
        clear_runtime()

    assert result == [{"c": 2}]
    assert fetch_calls == [area_entity_ids]


async def test_missing_entity_read_attaches_note(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """An empty literal state read adds a repair note without training weak memory."""
    code = """
result = hass.states.get("light.kitchen_main")
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] is None
    note = result["note"]
    assert isinstance(note, str)
    assert note
    assert "light.kitchen_main" in note
    assert "light.bedroom" in note

    follow_up = await _run_code(
        hass,
        loaded_entry,
        """
state = hass.states.get("light.kitchen_main")
result = state.entity_id if state is not None else None
""",
    )

    assert follow_up["execution"]["status"] == "ok"
    assert follow_up["output"] is None
    assert "resolutions" not in follow_up


async def test_missing_entity_diagnostic_output_attaches_note(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A missing literal state read that returns an absence diagnostic still gets a repair note."""
    code = """
s = hass.states.get("light.kitchen_main")
result = {"found": s is not None, "state": str(s)}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == {"found": False, "state": "None"}
    note = result["note"]
    assert isinstance(note, str)
    assert "light.kitchen_main" in note
    assert "light.bedroom" in note


async def test_missing_entity_note_suppressed_for_mixed_visible_data(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A mixed missing+visible literal read that returns real data does not get a missing-only note."""
    code = """
missing = hass.states.get("light.kitchen_main")
visible = hass.states.get("light.bedroom")
result = {"missing": missing is not None, "visible": visible.state if visible is not None else None}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == {"missing": False, "visible": "on"}
    assert "note" not in result


async def test_map_filter_normalize_and_run(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """map()/filter() run natively and return lists end-to-end."""
    code = """
result = {
    "map": map(lambda x: x * 2, [1, 2, 3]),
    "filter": filter(lambda x: x > 1, [1, 2, 3]),
    "filter_none": filter(None, [0, 1, 2]),
    "map_multi": map(lambda a, b: a + b, [1, 2], [10, 20]),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["map"] == [2, 4, 6]
    assert output["filter"] == [2, 3]
    assert output["filter_none"] == [1, 2]
    assert output["map_multi"] == [11, 22]


@pytest.mark.parametrize(
    ("method", "id_expr"),
    [
        pytest.param("async_entries_for_area", "bedroom_id", id="area"),
        pytest.param("async_entries_for_device", "device_id", id="device"),
    ],
)
async def test_registry_traversal_one_and_two_arg_forms_match(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    method: str,
    id_expr: str,
) -> None:
    """One-arg and two-arg registry traversal forms resolve identically."""
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=loaded_entry.entry_id,
        identifiers={("test", "traversal-device")},
    )
    er.async_get(hass).async_update_entity("light.bedroom", device_id=device.id)
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")

    code = f"""
bedroom_id = "{bedroom.id}"
device_id = "{device.id}"
one_arg = [e.entity_id for e in er.{method}({id_expr})]
two_arg = [e.entity_id for e in er.{method}(er.async_get(hass), {id_expr})]
result = {{"one_arg": one_arg, "two_arg": two_arg}}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["one_arg"] == output["two_arg"]
    assert output["one_arg"] == ["light.bedroom"]


async def test_entity_entry_domain_field_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The derived domain field is readable through the entity registry facade."""
    code = """
entry = entity_registry.async_get("light.bedroom")
result = entry.domain
"""
    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "light"


async def test_service_catalog_reads(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify the service catalog snapshot is accessible."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    code = """
result = {
    "has_turn_on": hass.services.has_service("light", "turn_on"),
    "has_missing": hass.services.has_service("light", "nonexistent"),
    "domain_count": len(hass.services.async_services()),
    "services": hass.services.async_services(),
    "light_services": hass.services.async_services_for_domain("light"),
    "missing_services": hass.services.async_services_for_domain("missing"),
    "turn_on_response": hass.services.supports_response("light", "turn_on"),
    "required_response": hass.services.supports_response("test_response", "required"),
    "missing_response": hass.services.supports_response("missing", "missing"),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["has_turn_on"] is True
    assert output["has_missing"] is False
    assert output["domain_count"] >= 1
    assert output["services"]["light"]["turn_on"]["supports_response"] == "none"
    assert output["services"]["light"]["turn_on"]["fields"] == []
    assert isinstance(output["services"]["light"]["turn_on"]["fields"], list)
    assert isinstance(output["services"]["light"]["turn_on"]["dynamic"], bool)
    assert output["services"]["test_response"]["required"]["supports_response"] == "only"
    assert output["light_services"]["turn_on"]["supports_response"] == "none"
    assert output["light_services"]["turn_on"]["fields"] == []
    assert isinstance(output["light_services"]["turn_on"]["fields"], list)
    assert isinstance(output["light_services"]["turn_on"]["dynamic"], bool)
    assert output["missing_services"] == {}
    assert output["turn_on_response"] == "none"
    assert output["required_response"] == "only"
    assert output["missing_response"] == "none"


async def test_config_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify safe hass.config values are readable through the Monty root."""
    hass.config.location_name = "Test Home"
    hass.config.elevation = 42
    hass.config.country = "NL"
    code = """
result = {
    "location_name": hass.config.location_name,
    "elevation": hass.config.elevation,
    "time_zone": hass.config.time_zone,
    "country": hass.config.country,
    "temperature_unit": hass.config.units.temperature_unit,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["location_name"] == "Test Home"
    assert output["elevation"] == 42
    assert output["country"] == "NL"
    assert isinstance(output["time_zone"], str)
    assert isinstance(output["temperature_unit"], str)


async def test_device_label_and_entity_lookup_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify device label and entity module lookup helpers use the snapshot."""
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=loaded_entry.entry_id,
        identifiers={("test", "dev1")},
        name="Labelled Device",
    )
    device_registry.async_update_device(device.id, labels={"fav"})
    # Link the visible Bedroom light to this device so the device survives the
    # snapshot's visibility filtering (entity-less devices are dropped unless
    # they are the anchor device).
    er.async_get(hass).async_update_entity("light.bedroom", device_id=device.id)
    code = """
result = {
    "device_label_count": len(dr.async_entries_for_label(dr.async_get(hass), "fav")),
    "entity_by_id": er.async_get_entity(er.async_get(hass), "light", "test", "bedroom"),
    "entity_entries_count": len(er.async_entries(er.async_get(hass))),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["device_label_count"] == 1
    assert output["entity_by_id"] == "light.bedroom"
    assert output["entity_entries_count"] >= 2


async def test_service_action_executes_and_records_outcome(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify async_call executes and records a successful action outcome."""
    events: list[str] = []
    hass.bus.async_listen("call_service", lambda event: events.append(event.data.get("service", "")))

    code = """
ret = await hass.services.async_call(
    "light",
    "turn_on",
    {"brightness_pct": 80},
    target={"entity_id": "light.bedroom"},
)
result = ret
"""

    result = await _run_code(hass, loaded_entry, code)
    await hass.async_block_till_done()

    assert result["execution"]["status"] == "ok"
    # Return value matches HA: None (not return_response).
    assert result["output"] is None
    actions = result["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["service"] == "light.turn_on"
    assert action["target"]["entity_id"] == ["light.bedroom"]
    assert action["status"] == "ok"
    assert "service_data" not in action
    assert "response" not in action
    assert "error" not in action
    assert "frozen snapshot" in result["notes"][0]
    assert "do not reread state" in result["notes"][0]
    assert events == ["turn_on"]


async def test_mapping_fourth_positional_service_target_is_normalized(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify a fourth positional mapping target executes through the facade."""
    events: list[str] = []
    hass.bus.async_listen("call_service", lambda event: events.append(event.data.get("service", "")))
    code = """
await hass.services.async_call(
    "light",
    "turn_on",
    {"brightness_pct": 80},
    {"entity_id": "light.bedroom"},
    blocking=True,
)
result = "done"
"""

    result = await _run_code(hass, loaded_entry, code)
    await hass.async_block_till_done()

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "done"
    assert result["actions"][0]["target"]["entity_id"] == ["light.bedroom"]
    assert events == ["turn_on"]


async def test_helper_registry_import_is_normalized(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify a supported helper-registry import resolves to its facade global."""
    code = """
from homeassistant.helpers import entity_registry as er
result = er.async_get(hass).async_get("light.bedroom").entity_id
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "light.bedroom"


async def test_policy_blocked_service_call_adds_action_failure_metadata(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Policy-blocked calls keep status ok but expose unmistakable action failure metadata."""
    mock_config_entry.add_to_hass(hass)
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={**mock_config_entry.options, CONF_ACTIONS_ENABLED: False},
    )
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await _run_code(
        hass,
        mock_config_entry,
        """
await hass.services.async_call("light", "turn_on", target={"entity_id": "light.bedroom"})
result = {"status": "unexpectedly_succeeded"}
""",
    )

    assert result["execution"]["status"] == "ok"
    assert result["execution"]["action_status"] == "error"
    assert result["execution"]["action_failures"] == ["actions_disabled"]
    assert result["output"] == {"status": "unexpectedly_succeeded"}
    assert result["actions"][0]["status"] == "error"
    assert result["notes"][0].startswith("1 of 1 service calls were blocked or failed")


async def test_action_payload_is_json_safe(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify nested non-primitive service action data is serialized safely."""
    code = """
state = hass.states.get("light.bedroom")
await hass.services.async_call(
    "light",
    "turn_on",
    {
        7: {
            "levels": (1, 2),
            "labels": {"cozy", "night"},
            "state": state,
        },
    },
    target={"entity_id": ("light.bedroom",)},
)
result = "ok"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    json.dumps(result["actions"])
    action = result["actions"][0]
    assert action["service"] == "light.turn_on"
    assert action["target"]["entity_id"] == ["light.bedroom"]
    assert "service_data" not in action


async def test_large_allocation_fails_with_monty_resource_limit(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify Monty memory limits bound runaway allocation before tool success."""
    result = await _run_code(hass, loaded_entry, "result = [0] * 10000000")

    assert result["execution"]["status"] == "code_error"
    assert result["execution"]["kind"] == "MemoryError"


async def test_timeout_returns_code_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify execution timeouts are runtime code errors, not setup errors."""

    async def _raise_timeout(_awaitable: Awaitable[Any], **_kwargs: object) -> Any:
        close = getattr(_awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    monkeypatch.setattr(executor.asyncio, "wait_for", _raise_timeout)

    result = await _run_code(hass, loaded_entry, "result = 1")

    assert result["execution"]["status"] == "code_error"
    assert result["execution"]["kind"] == "TimeoutError"
    assert result["output"] is None


async def test_positional_response_service_call_records_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify HA-style positional async_call arguments are accepted."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    events: list[str] = []
    hass.bus.async_listen("call_service", lambda event: events.append(event.data.get("service", "")))
    code = """
result = await hass.services.async_call(
    "test_response",
    "required",
    None,
    True,
    None,
    {"entity_id": "light.bedroom"},
    True,
)
"""

    result = await _run_code(hass, loaded_entry, code)
    await hass.async_block_till_done()

    assert result["execution"]["status"] == "ok"
    assert result["output"] == {}
    actions = result["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["service"] == "test_response.required"
    assert action["target"]["entity_id"] == ["light.bedroom"]
    assert action["status"] == "ok"
    assert action["response"] == result["output"]
    assert "service_data" not in action
    assert "blocking" not in action
    assert "return_response" not in action
    assert events == ["required"]


async def test_conditional_dead_result_branch_promotes_trailing_expression(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A ``result =`` in a dead branch must not suppress a trailing expression."""
    code = """
if False:
    result = 1
2 + 3
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == 5


async def test_conditional_result_never_bound_defaults_to_none(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A conditional ``result =`` that never binds yields None, not a NameError."""
    code = """
if False:
    result = 1
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] is None


@pytest.mark.parametrize(
    ("code", "expects_time_component"),
    [
        pytest.param("result = datetime.now().isoformat()", True, id="datetime-now-isoformat"),
        pytest.param("result = datetime.utcnow().isoformat()", True, id="datetime-utcnow-isoformat"),
        pytest.param("result = datetime.now()", True, id="datetime-direct-return"),
        pytest.param("result = date.today().isoformat()", False, id="date-today-isoformat"),
        pytest.param("result = datetime.now().date().isoformat()", False, id="datetime-date-method"),
        pytest.param("result = date.today()", False, id="date-direct-return"),
    ],
)
async def test_datetime_facade_values_serialize_stably(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    code: str,
    expects_time_component: bool,
) -> None:
    """Datetime/date facade values serialize to stable JSON-safe strings."""
    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert isinstance(output, str)
    if expects_time_component:
        assert "T" in output
    else:
        assert len(output) == 10


async def test_date_today_fields(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """date.today() exposes year, month, day, weekday fields."""
    code = """
d = date.today()
result = {
    "year": d.year,
    "month": d.month,
    "day": d.day,
    "weekday": d.weekday,
    "year_matches_snapshot": d.year == int(now[:4]),
}
"""
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["year_matches_snapshot"] is True
    assert isinstance(output["month"], int)
    assert isinstance(output["day"], int)
    assert isinstance(output["weekday"], int)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "from datetime import datetime\nresult = datetime.now().isoformat()[:4] == now[:4]",
            id="from-datetime-import",
        ),
        pytest.param(
            "import datetime as dt\nresult = dt.datetime.now().year == int(now[:4])",
            id="module-datetime-alias",
        ),
        pytest.param(
            "from datetime import date as d\nresult = len(d.today().isoformat()) == 10",
            id="from-date-alias",
        ),
    ],
)
async def test_datetime_imports_normalize_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    code: str,
) -> None:
    """Supported datetime/date imports normalize to the sandbox facades end to end."""
    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] is True


@pytest.mark.parametrize(
    ("code", "expected_output"),
    [
        pytest.param(
            "result = datetime.fromisoformat('2025-01-15T08:30:00+00:00').year",
            2025,
            id="datetime-fromisoformat",
        ),
        pytest.param("result = date.fromisoformat('2025-03-20').month", 3, id="date-fromisoformat"),
        pytest.param(
            """
s = hass.states.get('light.bedroom')
result = datetime.fromisoformat(s.last_changed).year == int(now[:4])
""",
            True,
            id="state-timestamp-fromisoformat",
        ),
    ],
)
async def test_datetime_fromisoformat_parses_expected_values(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    code: str,
    expected_output: object,
) -> None:
    """fromisoformat parsing works for explicit literals and state timestamps."""
    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == expected_output


async def test_now_global_unchanged(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The now global remains an ISO string, not a facade object."""
    code = "result = now"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert "T" in result["output"]
