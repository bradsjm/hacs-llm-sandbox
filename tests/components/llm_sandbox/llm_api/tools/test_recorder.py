"""Behavior tests for recorder-backed LLM tools."""

import base64
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
import json
import time
from typing import cast

from custom_components.llm_sandbox.const import TOOL_GET_HISTORY, TOOL_GET_LOGBOOK, TOOL_GET_STATISTICS
from custom_components.llm_sandbox.llm_api.data.history import HistoryRow
from custom_components.llm_sandbox.llm_api.tools import recorder
from custom_components.llm_sandbox.llm_api.tools._recorder_runtime import RecorderSource
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, GetLogbookTool, GetStatisticsTool
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.db_schema import Statistics
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType, StatisticMetaData
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .test_analytics import _snapshot


def _raw_cursor(payload: dict[str, object]) -> str:
    """Encode a raw cursor payload for malformed-cursor behavior tests."""
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")


def _compact_json_bytes(payload: Mapping[str, object]) -> int:
    """Measure a payload with the recorder response's compact UTF-8 encoding."""
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))


async def test_history_returns_states_for_visible_entity(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """History returns recorded state rows for a visible entity."""
    start = dt_util.utcnow().isoformat()
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "start": start})

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
    start = (dt_util.utcnow() - timedelta(seconds=1)).isoformat()
    hass.states.async_set(
        "light.bedroom",
        "off",
        {"friendly_name": "Bedroom Light", "brightness": 64, "color_mode": "rgb"},
    )
    await _sync_recorder(hass)
    hass.states.async_set(
        "light.bedroom",
        "on",
        {"friendly_name": "Bedroom Light", "brightness": 128, "color_mode": "rgb"},
    )
    await _sync_recorder(hass)

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


async def test_history_page_uses_compact_utf8_byte_limit(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Multibyte row values are measured as UTF-8 bytes in the full response."""
    start = dt_util.utcnow().isoformat()
    for state in ("one", "two"):
        hass.states.async_set(
            "light.bedroom",
            state,
            {"friendly_name": "Bedroom Light", "payload": "é" * 3000},
        )
    await _sync_recorder(hass)

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "start": start, "attributes": ["payload"]},
    )

    assert _row_states(result["entities"]["light.bedroom"]["rows"])[-2:] == ["one", "two"]
    assert _compact_json_bytes(result) <= recorder.MAX_RECORDER_PAGE_BYTES
    assert set(result) == {"window", "entities"}


async def test_history_oversized_first_row_is_intact_and_cursor_progresses(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized newest row is returned alone and does not hide older rows."""
    # Keep the payload below recorder's own storage cap while exercising the page cap.
    monkeypatch.setattr(recorder, "MAX_RECORDER_PAGE_BYTES", 8 * 1024)
    start = dt_util.utcnow().isoformat()
    hass.states.async_set("light.bedroom", "older", {"friendly_name": "Bedroom Light", "payload": "small"})
    hass.states.async_set(
        "light.bedroom",
        "newest",
        {"friendly_name": "Bedroom Light", "payload": "x" * (recorder.MAX_RECORDER_PAGE_BYTES + 1)},
    )
    await _sync_recorder(hass)

    first = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "start": start, "attributes": ["payload"]},
    )
    second = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "cursor": first["next_cursor"]},
    )

    assert _row_states(first["entities"]["light.bedroom"]["rows"]) == ["newest"]
    assert _compact_json_bytes(first) > recorder.MAX_RECORDER_PAGE_BYTES
    assert "older" in _row_states(second["entities"]["light.bedroom"]["rows"])
    assert "newest" not in _row_states(second["entities"]["light.bedroom"]["rows"])


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
    await _sync_recorder(hass)
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
    await _sync_recorder(hass)

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
        {"statistic_ids": ["sensor.energy"], "period": "hour", "types": ["sum"] * 6},
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
    await _sync_recorder(hass)

    result = await _call_logbook(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    assert result["scope"] == {"entity_ids": ["light.bedroom"]}
    assert isinstance(result["entries"], list)
    assert len(result["entries"]) >= 1
    row = result["entries"][-1]
    assert isinstance(row["when"], str)
    assert row["entity_id"] == "light.bedroom"


async def test_logbook_same_scope_cursor_returns_older_page(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logbook cursors continue when replayed against the same entity scope."""
    monkeypatch.setattr(recorder, "MAX_LOGBOOK_ENTRIES", 1)
    for state in ("off", "on", "off"):
        hass.states.async_set("light.bedroom", state, {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    first = await _call_logbook(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})
    second = await _call_logbook(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "cursor": first["next_cursor"]},
    )

    assert len(first["entries"]) == 1
    assert len(second["entries"]) == 1
    assert first["scope"] == second["scope"] == {"entity_ids": ["light.bedroom"]}
    assert first["entries"] != second["entries"]


async def test_logbook_scope_is_sorted_and_matches_selector_resolution(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Explicit and selector scopes expose the same normalized visible IDs."""
    from homeassistant.helpers import area_registry as ar

    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    assert bedroom is not None

    explicit = await _call_logbook(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})
    selected = await _call_logbook(hass, recorder_entry, {"area_id": bedroom.id})

    assert explicit["scope"] == selected["scope"] == {"entity_ids": ["light.bedroom"]}


async def test_logbook_empty_result_retains_resolved_scope(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """An authorized empty logbook result still identifies its resolved scope."""
    result = await _call_logbook(
        hass,
        recorder_entry,
        {
            "entity_ids": ["light.bedroom"],
            "start": "2000-01-01T00:00:00+00:00",
            "end": "2000-01-02T00:00:00+00:00",
        },
    )

    assert result["scope"] == {"entity_ids": ["light.bedroom"]}
    assert result["entries"] == []


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
    assert _guidance_candidate_ids(result["error"]["guidance"])


async def test_recorder_snapshot_visibility_is_fresh_per_call(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Recorder tools rebuild visibility on each call instead of reusing an old snapshot."""
    registry = er.async_get(hass)
    registry.async_get_or_create("light", "test", "fresh_visibility", suggested_object_id="fresh_visibility")
    hass.states.async_set("light.fresh_visibility", "on", {"friendly_name": "Fresh Visibility"})
    await _sync_recorder(hass)

    first = await _call_history(hass, recorder_entry, {"entity_ids": ["light.fresh_visibility"]})
    registry.async_update_entity("light.fresh_visibility", hidden_by=er.RegistryEntryHider.USER)
    second = await _call_history(hass, recorder_entry, {"entity_ids": ["light.fresh_visibility"]})

    assert "light.fresh_visibility" in first["entities"]
    assert second["status"] == "error"
    assert second["error"]["key"] == "entity_not_visible"


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args", "days", "max_hours"),
    [
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {"entity_ids": ["light.bedroom"]},
            30,
            recorder.MAX_RECORDER_LOOKBACK_HOURS,
            id="history",
        ),
        pytest.param(
            GetStatisticsTool,
            TOOL_GET_STATISTICS,
            {"statistic_ids": ["light.bedroom"]},
            400,
            recorder.MAX_STATISTICS_LOOKBACK_HOURS,
            id="statistics",
        ),
        pytest.param(
            GetLogbookTool,
            TOOL_GET_LOGBOOK,
            {"entity_ids": ["light.bedroom"]},
            30,
            recorder.MAX_RECORDER_LOOKBACK_HOURS,
            id="logbook",
        ),
    ],
)
async def test_window_too_large_rejected(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
    days: int,
    max_hours: int,
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
    assert str(max_hours) in result["error"]["message"]
    assert "hours" in result["error"]["message"]


@pytest.mark.parametrize(
    ("key", "placeholders", "tokens"),
    [
        pytest.param(
            "analytics_unknown_op", {"op": "median", "valid": "count, sum"}, ("median", "count, sum"), id="op"
        ),
        pytest.param(
            "analytics_unknown_group_key",
            {"group_key": "room", "valid": "domain, entity_id"},
            ("room", "domain, entity_id"),
            id="group-key",
        ),
        pytest.param("analytics_bad_bucket", {"bucket": "7x", "examples": "15m, 1h"}, ("7x", "15m, 1h"), id="bucket"),
        pytest.param(
            "invalid_tool_input",
            {"error": "aggregate cannot be combined with cursor"},
            ("aggregate", "cursor"),
            id="invalid-input",
        ),
    ],
)
def test_recorder_error_messages_include_rejected_values(
    key: str,
    placeholders: dict[str, str],
    tokens: tuple[str, ...],
) -> None:
    """Recorder fallback envelopes name rejected analytics/input values."""
    result = recorder.recorder_error_envelope(key, placeholders)

    assert result["status"] == "error"
    assert result["error"]["key"] == key
    assert all(token in str(result["error"]["message"]) for token in tokens)


@pytest.mark.parametrize(
    ("tool_args", "expected_hours"),
    [
        pytest.param({"entity_ids": ["light.bedroom"]}, 1, id="default-hour-window"),
        pytest.param({"entity_ids": ["light.bedroom"], "hours": 2}, 2, id="hours-argument-window"),
    ],
)
async def test_history_window_sizing(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_args: dict[str, object],
    expected_hours: int,
) -> None:
    """History windows default to one hour or honor the explicit hours selector."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(hass, recorder_entry, tool_args)

    start = dt_util.parse_datetime(cast(str, result["window"]["start"]))
    end = dt_util.parse_datetime(cast(str, result["window"]["end"]))
    assert start is not None
    assert end is not None
    expected_window = timedelta(hours=expected_hours)
    assert expected_window - timedelta(seconds=5) <= end - start <= expected_window + timedelta(seconds=5)


async def test_history_area_and_domain_selectors(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """HA-native selectors resolve to visible entities without enumerating IDs."""
    from homeassistant.helpers import area_registry as ar

    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    assert bedroom is not None

    by_area = await _call_history(hass, recorder_entry, {"area_id": bedroom.id})
    assert "light.bedroom" in by_area["entities"]

    by_domain = await _call_history(hass, recorder_entry, {"domain": "light"})
    assert "light.bedroom" in by_domain["entities"]


async def test_history_pure_domain_still_expands(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """A bare domain with no IDs and no location selectors still widens to all matches."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(hass, recorder_entry, {"domain": "light"})

    assert "light.bedroom" in result["entities"]


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"area_id": "kichen-typo", "domain": "light"}, id="bad-area-with-domain"),
        pytest.param({"area_id": "kichen-typo"}, id="bad-area-without-domain"),
    ],
)
async def test_history_bad_area_errors_with_candidates(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> None:
    """A typo'd area selector errors with concrete fixes instead of widening the scope."""
    from homeassistant.helpers import area_registry as ar

    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room Light"})
    await _sync_recorder(hass)

    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    assert bedroom is not None

    result = await _call_history(hass, recorder_entry, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "selector_no_match"
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]
    assert "light.bedroom" in _guidance_candidate_ids(result["error"]["guidance"])


async def test_history_mixed_selector_hit_and_miss_returns_selector_no_match(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """One bad location selector value is rejected even when another value matches."""
    from homeassistant.helpers import area_registry as ar

    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    assert bedroom is not None

    result = await _call_history(
        hass,
        recorder_entry,
        {"area_id": [bedroom.id, "kichen-typo"], "domain": "light"},
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "selector_no_match"
    assert result["error"]["message"]


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
    await _sync_recorder(hass)

    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "start": start})

    assert "next_cursor" in result
    assert result["overflow"]["truncated"] is True
    assert result["overflow"]["next_cursor"] == result["next_cursor"]
    assert _row_states(result["entities"]["light.bedroom"]["rows"]) == ["3", "4", "5"]
    assert len(result["entities"]["light.bedroom"]["rows"]) <= recorder.MAX_HISTORY_STATES


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
    await _sync_recorder(hass)

    first = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "start": start})
    cursor = cast(str, first["next_cursor"])
    second = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "cursor": cursor})

    assert _row_states(first["entities"]["light.bedroom"]["rows"]) == ["3", "4", "5"]
    assert _row_states(second["entities"]["light.bedroom"]["rows"]) == ["0", "1", "2"]
    assert "next_cursor" in second
    walked = _row_states(first["entities"]["light.bedroom"]["rows"]) + _row_states(
        second["entities"]["light.bedroom"]["rows"]
    )
    assert len(walked) == len(set(walked)) == 6
    assert set(walked) == {str(index) for index in range(6)}


async def test_cursor_rejects_different_tool(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor from one recorder tool cannot continue a different tool."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 1)
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)
    first = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    result = await _call_logbook(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "cursor": first["next_cursor"]},
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_cursor"


async def test_cursor_rejects_different_scope(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor from one resolved scope cannot continue a different scope."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 1)
    hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room Light"})
    await _sync_recorder(hass)
    first = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"]})

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.living_room"], "cursor": first["next_cursor"]},
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_cursor"


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
    await _sync_recorder(hass)

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


async def test_history_multi_entity_asymmetric_exhaustion_no_duplicates(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An early-exhausted stream yields empty later pages instead of duplicate rows."""
    monkeypatch.setattr(recorder, "MAX_HISTORY_STATES", 4)
    start = dt_util.utcnow().isoformat()
    # light.bedroom has many rows (spans pages); light.living_room has one (exhausts page 1).
    for index in range(6):
        hass.states.async_set("light.bedroom", str(index), {"friendly_name": "Bedroom Light"})
    hass.states.async_set("light.living_room", "only", {"friendly_name": "Living Room Light"})
    await _sync_recorder(hass)

    entity_ids = ["light.bedroom", "light.living_room"]
    first = await _call_history(hass, recorder_entry, {"entity_ids": entity_ids, "start": start})
    assert "next_cursor" in first
    # living_room had rows on page 1 and is now exhausted.
    assert first["entities"]["light.living_room"]["rows"]

    second = await _call_history(hass, recorder_entry, {"entity_ids": entity_ids, "cursor": first["next_cursor"]})

    # The exhausted stream must return an empty page, not re-emit its newest rows as duplicates.
    assert second["entities"]["light.living_room"]["rows"] == []
    # The non-exhausted stream still advances.
    assert second["entities"]["light.bedroom"]["rows"]


async def test_history_declarative_limit_above_cap_clamps(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Declarative analytics accepts oversized positive limits and clamps internally."""
    hass.states.async_set("light.bedroom", "on")
    await _sync_recorder(hass)

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "aggregate": "state_counts", "group_by": ["domain"], "limit": 10_000},
    )

    assert "status" not in result
    rows = result["rows"]
    assert isinstance(rows, list)
    assert len(rows) <= 500
    assert rows[0]["domain"] == "light"
    assert "state_counts" in rows[0]


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param(
            {"entity_ids": ["light.bedroom"], "aggregate": {"value": ["mean"]}},
            id="aggregate-object",
        ),
        pytest.param(
            {"entity_ids": ["light.bedroom"], "aggregate": "state_counts", "value_operations": ["mean"]},
            id="aggregate-and-value-operations",
        ),
        pytest.param({"entity_ids": ["light.bedroom"], "agg": "state_counts"}, id="agg-alias"),
        pytest.param({"entity_ids": ["light.bedroom"], "groupby": ["domain"]}, id="groupby-alias"),
        pytest.param({"entity_ids": ["light.bedroom"], "resample": "1h"}, id="resample-alias"),
        pytest.param({"entity_ids": ["light.bedroom"], "interval": "1h"}, id="interval-alias"),
    ],
)
async def test_history_rejects_legacy_aggregate_inputs(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> None:
    """Aggregate objects, mixed forms, and removed aliases fail schema validation."""
    result = await _call_history(hass, recorder_entry, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"


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
    "cursor",
    [
        pytest.param(
            _raw_cursor(
                {
                    "v": 2,
                    "k": 123,
                    "ids": ["light.bedroom"],
                    "s": "2026-01-01T00:00:00+00:00",
                    "e": "2026-01-01T01:00:00+00:00",
                    "c": {},
                }
            ),
            id="malformed-kind",
        ),
        pytest.param(
            _raw_cursor(
                {
                    "v": 2,
                    "k": "history",
                    "ids": [123],
                    "s": "2026-01-01T00:00:00+00:00",
                    "e": "2026-01-01T01:00:00+00:00",
                    "c": {},
                }
            ),
            id="malformed-scope-ids",
        ),
    ],
)
async def test_malformed_cursor_kind_or_scope_returns_invalid_cursor(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    cursor: str,
) -> None:
    """Malformed cursor kind/scope fields return the stable invalid_cursor key."""
    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "cursor": cursor})

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_cursor"


@pytest.mark.parametrize(
    "cursor",
    [
        pytest.param(
            _raw_cursor(
                {
                    "v": 2,
                    "k": "history",
                    "ids": ["light.bedroom"],
                    "s": "2026-01-01T01:00:00+00:00",
                    "e": "2026-01-01T00:00:00+00:00",
                    "c": {},
                }
            ),
            id="start-after-end",
        ),
        pytest.param(
            _raw_cursor(
                {
                    "v": 2,
                    "k": "history",
                    "ids": ["light.bedroom"],
                    "s": "2026-01-01T00:00:00+00:00",
                    "e": "2026-01-03T00:00:00+00:00",
                    "c": {},
                }
            ),
            id="oversized-window",
        ),
    ],
)
async def test_tampered_cursor_window_returns_invalid_cursor(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    cursor: str,
) -> None:
    """A cursor with a tampered but parseable window returns invalid_cursor."""
    result = await _call_history(hass, recorder_entry, {"entity_ids": ["light.bedroom"], "cursor": cursor})

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_cursor"


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args"),
    [
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {"entity_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor", "hours": 1},
            id="history-hours",
        ),
        pytest.param(
            GetStatisticsTool,
            TOOL_GET_STATISTICS,
            {
                "statistic_ids": ["light.bedroom"],
                "cursor": "not-a-valid-cursor",
                "start": dt_util.utcnow().isoformat(),
            },
            id="statistics-start",
        ),
        pytest.param(
            GetLogbookTool,
            TOOL_GET_LOGBOOK,
            {"entity_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor", "end": dt_util.utcnow().isoformat()},
            id="logbook-end",
        ),
        pytest.param(
            GetStatisticsTool,
            TOOL_GET_STATISTICS,
            {"statistic_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor", "types": ["sum"]},
            id="statistics-types",
        ),
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {"entity_ids": ["light.bedroom"], "cursor": "not-a-valid-cursor", "attributes": ["brightness"]},
            id="history-attributes",
        ),
    ],
)
async def test_cursor_cannot_be_combined_with_window_args(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
) -> None:
    """Cursor calls reject explicit window arguments before decoding the cursor."""
    result = await _call_tool(tool_cls(recorder_entry.entry_id), tool_name, hass, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    assert result["error"]["message"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "entity_ids": ["sensor.temp"],
                "to_state": "on",
            },
            id="to-state-without-aggregate",
        ),
        pytest.param(
            {
                "entity_ids": ["sensor.temp"],
                "aggregate": "state_counts",
                "from_state": "off",
            },
            id="from-state-with-incompatible-aggregate",
        ),
    ],
)
async def test_history_analytics_validation_precedes_recorder_fetch(
    data: dict[str, object],
) -> None:
    """Invalid analytics combinations return envelopes without recorder I/O."""
    fetch_calls = 0

    async def _run_in_executor(fn: Callable[[], object]) -> object:
        return fn()

    async def _fetch_history(
        _entity_ids: list[str],
        _start: datetime,
        _end: datetime,
    ) -> dict[str, list[HistoryRow]]:
        nonlocal fetch_calls
        fetch_calls += 1
        return {}

    async def _fetch_statistics(
        _statistic_ids: list[str],
        _start: datetime,
        _end: datetime,
        _period: str,
        _types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        return {}

    async def _fetch_logbook(
        _entity_ids: list[str],
        _start: datetime,
        _end: datetime,
    ) -> list[dict[str, object]]:
        return []

    source = RecorderSource(
        now=datetime(2026, 1, 1, 1, tzinfo=UTC),
        logbook_available=True,
        run_in_executor=_run_in_executor,
        fetch_history=_fetch_history,
        fetch_statistics=_fetch_statistics,
        fetch_logbook=_fetch_logbook,
    )

    result = await GetHistoryTool("eval").run_query(_snapshot(), data, source)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    assert fetch_calls == 0


async def test_history_rejects_hours_with_explicit_start(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Window resolution rejects hours with start instead of ignoring hours."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "start": dt_util.utcnow().isoformat(), "hours": 1},
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"


async def test_history_accepts_hours_with_explicit_end(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """A relative hours window may still anchor against an explicit end."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "end": dt_util.utcnow().isoformat(), "hours": 2},
    )

    assert "status" not in result
    start = dt_util.parse_datetime(cast(str, result["window"]["start"]))
    end = dt_util.parse_datetime(cast(str, result["window"]["end"]))
    assert start is not None
    assert end is not None
    assert timedelta(hours=2) - timedelta(seconds=5) <= end - start <= timedelta(hours=2) + timedelta(seconds=5)


def test_recorder_schemas_keep_runtime_scalar_id_tolerance() -> None:
    """Provider arrays remain forgiving singleton inputs at the runtime boundary."""
    history = cast(
        dict[str, object],
        GetHistoryTool("eval").parameters(
            {
                "entity_ids": "sensor.temp",
                "start": "2026-01-01T00:00:00+00:00",
            }
        ),
    )
    statistics = cast(
        dict[str, object],
        GetStatisticsTool("eval").parameters({"statistic_ids": "sensor.temp"}),
    )
    logbook = cast(
        dict[str, object],
        GetLogbookTool("eval").parameters({"entity_ids": "sensor.temp"}),
    )

    assert history["entity_ids"] == ["sensor.temp"]
    assert history["start"] == datetime(2026, 1, 1, tzinfo=UTC)
    assert statistics["statistic_ids"] == ["sensor.temp"]
    assert logbook["entity_ids"] == ["sensor.temp"]


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"entity_ids": []}, id="empty-entity-list"),
        pytest.param({"entity_ids": ["not-an-entity"]}, id="malformed-entity-id"),
        pytest.param(
            {"entity_ids": {"value": ["light.bedroom"]}},
            id="entity-ids-wrapper",
        ),
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


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "tool_args"),
    [
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {
                "entity_ids": ["light.bedroom"],
                "start": "",
                "end": "",
                "area_id": "",
                "device_id": "",
                "from_state": "",
                "to_state": "",
                "bucket": "",
                "order_by": "",
                "cursor": "",
                "attributes": [],
                "group_by": [],
                "where": [],
                "hours": None,
                "aggregate": None,
                "value_operations": None,
                "limit": None,
            },
            id="history-null",
        ),
        pytest.param(
            GetHistoryTool,
            TOOL_GET_HISTORY,
            {
                "entity_ids": ["light.bedroom"],
                "start": "",
                "end": "",
                "area_id": "",
                "device_id": "",
                "from_state": "",
                "to_state": "",
                "bucket": "",
                "order_by": "",
                "cursor": "",
                "attributes": [],
                "group_by": [],
                "where": [],
                "hours": None,
                "aggregate": None,
                "value_operations": [],
                "limit": None,
            },
            id="history-empty-value-operations",
        ),
        pytest.param(
            GetStatisticsTool,
            TOOL_GET_STATISTICS,
            {
                "statistic_ids": ["light.bedroom"],
                "start": "",
                "end": "",
                "types": [],
                "cursor": "",
                "hours": None,
                "period": None,
            },
            id="statistics",
        ),
        pytest.param(
            GetLogbookTool,
            TOOL_GET_LOGBOOK,
            {
                "entity_ids": ["light.bedroom"],
                "start": "",
                "end": "",
                "cursor": "",
                "hours": None,
            },
            id="logbook",
        ),
    ],
)
async def test_empty_optional_args_omitted(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
    tool_cls: type[GetHistoryTool | GetStatisticsTool | GetLogbookTool],
    tool_name: str,
    tool_args: dict[str, object],
) -> None:
    """Empty/null optional values are ignored as if omitted (Postel's law)."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_tool(tool_cls(recorder_entry.entry_id), tool_name, hass, tool_args)

    assert "status" not in result


async def test_history_empty_entity_ids_ignored_with_domain_scope(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """An empty entity_ids list is ignored when a domain selector supplies scope."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(hass, recorder_entry, {"entity_ids": [], "domain": "light"})

    assert "light.bedroom" in result["entities"]


async def test_history_non_empty_attributes_with_analytics_rejected(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Non-empty attributes combined with analytics still returns invalid_tool_input."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    await _sync_recorder(hass)

    result = await _call_history(
        hass,
        recorder_entry,
        {"entity_ids": ["light.bedroom"], "attributes": ["brightness"], "aggregate": "state_counts"},
    )

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"
    assert "aggregate" in str(result["error"]["message"])
    assert "attributes" in str(result["error"]["message"])


async def _call_history(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> JsonObjectType:
    """Call GetHistoryTool with a standard test LLM context."""
    return await _call_tool(GetHistoryTool(entry.entry_id), TOOL_GET_HISTORY, hass, tool_args)


async def _sync_recorder(hass: HomeAssistant) -> None:
    """Deterministically commit recorder writes before direct history assertions."""
    # Test state writes are followed by immediate recorder-backed reads; use the
    # same unconditional commit-before barrier as production to avoid TOCTOU gaps.
    await recorder._sync_recorder_for_query(hass, get_instance(hass), time.monotonic() + 10)


def _row_states(rows: list[list[object]]) -> list[str]:
    """Return the state/value field from compact recorder rows."""
    return [str(row[1]) for row in rows]


def _guidance_candidate_ids(guidance: object) -> set[str]:
    """Return candidate ids from a serialized recorder-error guidance payload."""
    assert isinstance(guidance, Mapping)
    candidates = guidance["candidates"]
    assert isinstance(candidates, list)
    return {str(candidate["id"]) for candidate in candidates if isinstance(candidate, Mapping)}


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
