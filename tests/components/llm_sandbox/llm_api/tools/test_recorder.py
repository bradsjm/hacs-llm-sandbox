"""Behavior tests for recorder-backed LLM tools."""

from datetime import timedelta
from typing import cast

import pytest
from custom_components.llm_sandbox.const import TOOL_GET_HISTORY, TOOL_GET_LOGBOOK, TOOL_GET_STATISTICS
from custom_components.llm_sandbox.llm_api.tools import recorder
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, GetLogbookTool, GetStatisticsTool
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.db_schema import Statistics
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType, StatisticMetaData
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_history_returns_states_for_visible_entity(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """History returns recorded state rows for a visible entity."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()

    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    assert "light.bedroom" in result["entities"]
    assert isinstance(result["window"]["start"], str)
    assert isinstance(result["window"]["end"], str)
    assert "next_cursor" not in result
    entity = result["entities"]["light.bedroom"]
    assert set(entity) == {"rows"}
    row = entity["rows"][-1]
    assert isinstance(row, list)
    assert len(row) == 2
    assert isinstance(row[0], str)
    assert isinstance(row[1], str)


async def test_history_includes_requested_attributes(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Requested attributes append a per-row dict; absent and unrequested names are omitted."""
    start = dt_util.utcnow().isoformat()
    hass.states.async_set(
        "light.bedroom",
        "off",
        {"friendly_name": "Bedroom Light", "brightness": 64, "color_mode": "rgb"},
    )
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()
    hass.states.async_set(
        "light.bedroom",
        "on",
        {"friendly_name": "Bedroom Light", "brightness": 128, "color_mode": "rgb"},
    )
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()

    result = await _call_history(
        hass,
        recorder_entry,
        {
            "entity_ids": ["light.bedroom"],
            "start": start,
            "attributes": ["brightness", "color_mode", "missing_attr"],
        },
    )

    row = result["entities"]["light.bedroom"]["rows"][-1]
    assert isinstance(row, list)
    assert len(row) == 3
    assert isinstance(row[2], dict)
    assert row[2] == {"brightness": 128, "color_mode": "rgb"}

    missing_only = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "start": start, "attributes": ["missing_attr"]},
    )
    missing_row = missing_only["entities"]["light.bedroom"]["rows"][-1]
    assert isinstance(missing_row, list)
    assert len(missing_row) == 3
    assert missing_row[2] == {}


async def test_statistics_returns_rows_for_visible_statistic(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Statistics returns long-term statistic rows for a visible statistic ID."""
    monkeypatch.setattr(recorder, "MAX_STATISTICS_ROWS", 2)
    er.async_get(hass).async_get_or_create(
        "sensor",
        "test",
        "energy",
        suggested_object_id="energy",
    )
    hass.states.async_set("sensor.energy", "12", {"friendly_name": "Energy", "state_class": "total"})
    await hass.async_block_till_done()
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=5)
    get_instance(hass).async_import_statistics(
        StatisticMetaData(
            has_mean=True,
            has_sum=True,
            mean_type=StatisticMeanType.NONE,
            name="Energy",
            source="sensor",
            statistic_id="sensor.energy",
            unit_class=None,
            unit_of_measurement="kWh",
        ),
        [
            StatisticData(
                start=start + timedelta(hours=index),
                mean=12.0 + index,
                min=12.0 + index,
                max=12.0 + index,
                sum=120.0 + index,
            )
            for index in range(4)
        ],
        Statistics,
    )
    await get_instance(hass).async_block_till_done()

    result = await _call_statistics(hass, recorder_entry, {"statistic_ids": ["sensor.energy"], "period": "hour"})

    assert result["period"] == "hour"
    assert "sensor.energy" in result["statistics"]
    row = result["statistics"]["sensor.energy"]["rows"][-1]
    assert isinstance(row, list)
    assert len(row) == 2
    assert isinstance(row[0], str)
    assert isinstance(row[1], dict)
    assert result["statistics"]["sensor.energy"]["fields"] == ["sum"]

    sum_only = await _call_statistics(
        hass,
        recorder_entry,
        {"statistic_ids": ["sensor.energy"], "period": "hour", "types": ["sum"]},
    )
    sum_rows = sum_only["statistics"]["sensor.energy"]["rows"]
    assert sum_rows
    assert all(list(row[1]) == ["sum"] for row in sum_rows)
    assert sum_only["statistics"]["sensor.energy"]["fields"] == ["sum"]
    cursor = cast(str, sum_only["next_cursor"])
    older_sums = await _call_statistics(
        hass,
        recorder_entry,
        {"statistic_ids": ["sensor.energy"], "cursor": cursor},
    )
    older_rows = older_sums["statistics"]["sensor.energy"]["rows"]
    assert older_rows
    assert all(list(row[1]) == ["sum"] for row in older_rows)
    assert older_sums["statistics"]["sensor.energy"]["fields"] == ["sum"]

    invalid = await _call_statistics(hass, recorder_entry, {"statistic_ids": ["sensor.energy"], "types": ["bogus"]})
    assert invalid["status"] == "error"
    assert invalid["error"]["key"] == "invalid_tool_input"


async def test_logbook_returns_entries_for_visible_entity(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Logbook returns real recorder-backed entries for a visible entity."""
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()

    result = await _call_logbook(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    assert isinstance(result["entries"], list)
    assert len(result["entries"]) >= 1
    row = result["entries"][-1]
    assert isinstance(row["when"], str)
    assert row["entity_id"] == "light.bedroom"


async def test_logbook_unavailable_returns_error_key(
    hass: HomeAssistant,
    recorder_without_logbook_entry: MockConfigEntry,
) -> None:
    """Logbook returns a stable error when the logbook component is absent."""
    result = await _call_logbook(hass, recorder_without_logbook_entry, {"entity_ids": ["light.bedroom"]})

    assert result["status"] == "error"
    assert result["error"]["key"] == "logbook_unavailable"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args"),
    [
        pytest.param(GetHistoryTool, TOOL_GET_HISTORY, {"entity_ids": ["light.bedroom"]}, id="history"),
        pytest.param(GetStatisticsTool, TOOL_GET_STATISTICS, {"statistic_ids": ["light.bedroom"]}, id="statistics"),
        pytest.param(GetLogbookTool, TOOL_GET_LOGBOOK, {"entity_ids": ["light.bedroom"]}, id="logbook"),
    ],
)
async def test_recorder_absent_returns_error_key(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
) -> None:
    """Recorder-backed tools return a stable error when recorder is absent."""
    result = await _call_tool(tool_cls(loaded_entry.entry_id), tool_name, hass, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "recorder_unavailable"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args"),
    [
        pytest.param(GetHistoryTool, TOOL_GET_HISTORY, {"entity_ids": ["light.living_room"]}, id="history"),
        pytest.param(
            GetStatisticsTool, TOOL_GET_STATISTICS, {"statistic_ids": ["light.living_room"]}, id="statistics"
        ),
        pytest.param(GetLogbookTool, TOOL_GET_LOGBOOK, {"entity_ids": ["light.living_room"]}, id="logbook"),
    ],
)
async def test_non_visible_entity_rejected(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
) -> None:
    """Hidden entities are rejected by the fresh snapshot visibility check."""
    er.async_get(hass).async_update_entity("light.living_room", hidden_by=er.RegistryEntryHider.USER)

    result = await _call_tool(tool_cls(recorder_entry.entry_id), tool_name, hass, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "entity_not_visible"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]
    fix = result["error"]["fix"]
    assert isinstance(fix, list)
    assert fix


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args", "days"),
    [
        pytest.param(GetHistoryTool, TOOL_GET_HISTORY, {"entity_ids": ["light.bedroom"]}, 30, id="history"),
        pytest.param(
            GetStatisticsTool, TOOL_GET_STATISTICS, {"statistic_ids": ["light.bedroom"]}, 400, id="statistics"
        ),
        pytest.param(GetLogbookTool, TOOL_GET_LOGBOOK, {"entity_ids": ["light.bedroom"]}, 30, id="logbook"),
    ],
)
async def test_window_too_large_rejected(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
    days: int,
) -> None:
    """Windows beyond the recorder lookback cap return the stable error key."""
    end = dt_util.utcnow()
    start = end - timedelta(days=days)
    args = tool_args | {"start": start.isoformat(), "end": end.isoformat()}

    result = await _call_tool(
        tool_cls(recorder_entry.entry_id),
        tool_name,
        hass,
        args,
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "time_window_too_large"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]
    assert isinstance(result["error"]["fix"], list)


async def test_history_window_clamped_to_default_when_omitted(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Omitting start/end queries the default one-hour history window."""
    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    start = dt_util.parse_datetime(cast(str, result["window"]["start"]))
    end = dt_util.parse_datetime(cast(str, result["window"]["end"]))
    assert start is not None
    assert end is not None
    assert timedelta(hours=1) - timedelta(seconds=5) <= end - start <= timedelta(hours=1) + timedelta(seconds=5)


async def test_history_hours_sizes_window(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """The hours argument sizes the window without ISO/timedelta math."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()

    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "hours": 2})

    start = dt_util.parse_datetime(cast(str, result["window"]["start"]))
    end = dt_util.parse_datetime(cast(str, result["window"]["end"]))
    assert start is not None
    assert end is not None
    assert timedelta(hours=2) - timedelta(seconds=5) <= end - start <= timedelta(hours=2) + timedelta(seconds=5)


async def test_history_area_and_domain_selectors(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """HA-native selectors resolve to visible entities without enumerating IDs."""
    from homeassistant.helpers import area_registry as ar

    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await hass.async_block_till_done()
    await get_instance(hass).async_block_till_done()

    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    assert bedroom is not None

    by_area = await _call_history(hass, recorder_entry, {"area_id": bedroom.id})
    assert "light.bedroom" in by_area["entities"]

    by_domain = await _call_history(hass, recorder_entry, {"domain": "light"})
    assert "light.bedroom" in by_domain["entities"]


async def test_history_paginates_large_result(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History output keeps the newest page and exposes a cursor."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 3)
    start = dt_util.utcnow().isoformat()
    for index in range(6):
        hass.states.async_set("light.bedroom", str(index), {"friendly_name": "Bedroom Light"})
        await hass.async_block_till_done()
        await get_instance(hass).async_block_till_done()

    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "start": start})

    assert "next_cursor" in result
    assert "truncated" not in result
    assert _row_states(result["entities"]["light.bedroom"]["rows"]) == ["3", "4", "5"]


async def test_history_pagination_walk_returns_older_page(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History cursor walks from newest rows to older rows."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 3)
    start = dt_util.utcnow().isoformat()
    for index in range(6):
        hass.states.async_set("light.bedroom", str(index), {"friendly_name": "Bedroom Light"})
        await hass.async_block_till_done()
        await get_instance(hass).async_block_till_done()

    first = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "start": start})
    cursor = cast(str, first["next_cursor"])
    second = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "cursor": cursor})

    assert _row_states(first["entities"]["light.bedroom"]["rows"]) == ["3", "4", "5"]
    assert _row_states(second["entities"]["light.bedroom"]["rows"]) == ["0", "1", "2"]
    assert "next_cursor" in second


async def test_history_multi_entity_paginates_independently(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History cursor keeps independent page boundaries per entity."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 6)
    start = dt_util.utcnow().isoformat()
    for index in range(6):
        hass.states.async_set("light.bedroom", str(index), {"friendly_name": "Bedroom Light"})
        hass.states.async_set("light.living_room", str(index), {"friendly_name": "Living Room Light"})
        await hass.async_block_till_done()
        await get_instance(hass).async_block_till_done()

    first = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom", "light.living_room"], "start": start},
    )
    cursor = cast(str, first["next_cursor"])
    second = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom", "light.living_room"], "cursor": cursor},
    )

    assert _row_states(first["entities"]["light.bedroom"]["rows"]) == ["3", "4", "5"]
    assert _row_states(first["entities"]["light.living_room"]["rows"]) == ["3", "4", "5"]
    assert _row_states(second["entities"]["light.bedroom"]["rows"]) == ["0", "1", "2"]
    assert _row_states(second["entities"]["light.living_room"]["rows"]) == ["0", "1", "2"]
    assert "next_cursor" in second


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args"),
    [
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {"entity_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor"},
            id="history",
        ),
        pytest.param(
            GetStatisticsTool,
            TOOL_GET_STATISTICS,
            {"statistic_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor"},
            id="statistics",
        ),
        pytest.param(
            GetLogbookTool,
            TOOL_GET_LOGBOOK,
            {"entity_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor"},
            id="logbook",
        ),
    ],
)
async def test_malformed_cursor_returns_invalid_cursor(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
) -> None:
    """A malformed cursor returns the stable invalid_cursor error key."""
    result = await _call_tool(tool_cls(recorder_entry.entry_id), tool_name, hass, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_cursor"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"entity_ids": []}, id="empty-entity-list"),
        pytest.param({"entity_ids": ["not-an-entity"]}, id="malformed-entity-id"),
        pytest.param(
            {
                "entity_ids": ["light.bedroom"],
                "attributes": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            },
            id="too-many-attributes",
        ),
    ],
)
async def test_invalid_input_returns_invalid_tool_input(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> None:
    """Schema violations return the shared invalid-tool-input envelope."""
    result = await _call_history(hass, recorder_entry, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]


async def _call_history(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Call GetHistoryTool with a standard test LLM context."""
    return await _call_tool(GetHistoryTool(entry.entry_id), TOOL_GET_HISTORY, hass, tool_args)


def _row_states(rows: list[list[object]]) -> list[str]:
    """Return the state/value field from compact recorder rows."""
    return [str(row[1]) for row in rows]


async def _call_statistics(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Call GetStatisticsTool with a standard test LLM context."""
    return await _call_tool(GetStatisticsTool(entry.entry_id), TOOL_GET_STATISTICS, hass, tool_args)


async def _call_logbook(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Call GetLogbookTool with a standard test LLM context."""
    return await _call_tool(GetLogbookTool(entry.entry_id), TOOL_GET_LOGBOOK, hass, tool_args)


async def _call_tool(
    tool: GetHistoryTool | GetStatisticsTool | GetLogbookTool,
    tool_name: str,
    hass: HomeAssistant,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Call one recorder tool with a standard test LLM context."""
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    return await tool.async_call(
        hass,
        llm.ToolInput(tool_name=tool_name, tool_args=tool_args),
        llm_context,
    )
