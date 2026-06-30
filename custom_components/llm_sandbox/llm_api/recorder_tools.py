"""Recorder-backed read-only LLM tools."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast, final, override

import voluptuous as vol
from homeassistant.components.logbook import DOMAIN as LOGBOOK_DOMAIN
from homeassistant.components.logbook.helpers import async_determine_event_types
from homeassistant.components.logbook.processor import EventProcessor
from homeassistant.components.recorder import get_instance, history, statistics
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm
from homeassistant.helpers.recorder import DATA_INSTANCE
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from ..const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    DEFAULT_LOGBOOK_WINDOW_HOURS,
    DEFAULT_STATISTICS_WINDOW_HOURS,
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_RECORDER_ENTITY_IDS,
    MAX_RECORDER_LOOKBACK_HOURS,
    MAX_STATISTICS_LOOKBACK_HOURS,
    MAX_STATISTICS_ROWS,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from ..runtime import SandboxSettings, settings_from_entry
from ..snapshot import build_snapshot
from ..snapshot.models import HomeSnapshot
from ..types import TranslationPlaceholders
from .errors import tool_error_envelope, tool_error_from_exception
from .executor_support import json_safe
from .prompts import build_get_history_description, build_get_logbook_description, build_get_statistics_description

RECORDER_UNAVAILABLE = "recorder_unavailable"
ENTITY_NOT_VISIBLE = "entity_not_visible"
TIME_WINDOW_TOO_LARGE = "time_window_too_large"
QUERY_FAILED = "query_failed"


@dataclass(frozen=True, slots=True)
class _RecoverableToolError(Exception):
    """Controlled recorder-tool failure converted to an error envelope."""

    key: str
    placeholders: TranslationPlaceholders

    def __post_init__(self) -> None:
        """Initialize Exception with the stable error key."""
        Exception.__init__(self, self.key)


def _iso_datetime(value: object) -> datetime:
    """Validate an ISO datetime string or datetime object and return UTC-aware datetime."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise vol.Invalid("expected an ISO datetime")


class _RecorderTool(llm.Tool):
    """Shared base for recorder-backed read-only tools."""

    name: str
    description: str
    parameters: vol.Schema

    def __init__(self, entry_id: str) -> None:
        """Initialize the recorder tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        try:
            data = cast(dict[str, object], self.parameters(tool_input.tool_args))
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            return tool_error_envelope(*mapped)

        if not _recorder_available(hass):
            return tool_error_envelope(RECORDER_UNAVAILABLE, {})

        from .api import _require_loaded_entry, _require_loaded_entry_error

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            key, placeholders = setup_error
            return tool_error_envelope(key, placeholders)
        entry = _require_loaded_entry(hass, self.entry_id)
        settings = settings_from_entry(entry)
        # Build a fresh visible snapshot for every recorder tool call.
        snapshot = build_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        deadline = time.monotonic() + settings.execution_timeout_seconds

        try:
            return await self._query(hass, snapshot, settings, deadline, data)
        except _RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)
        except Exception as err:  # noqa: BLE001 - recorder tools map unexpected query failures to envelopes
            mapped = tool_error_from_exception(err)
            if mapped is None:
                return tool_error_envelope(QUERY_FAILED, {"error": type(err).__name__})
            return tool_error_envelope(*mapped)

    async def _query(
        self,
        hass: HomeAssistant,
        snapshot: HomeSnapshot,
        settings: SandboxSettings,
        deadline: float,
        data: dict[str, object],
    ) -> JsonObjectType:
        """Run the concrete recorder query."""
        raise NotImplementedError


def _recorder_available(hass: HomeAssistant) -> bool:
    """Return whether the recorder integration has an active instance."""
    return DATA_INSTANCE in hass.data


def _validate_visibility(snapshot: HomeSnapshot, ids: list[str]) -> None:
    """Require all requested IDs to exist in the fresh visible snapshot."""
    for entity_id in ids:
        if entity_id not in snapshot.states:
            raise _RecoverableToolError(ENTITY_NOT_VISIBLE, {"entity_id": entity_id})


def _clamp_window(
    start_in: datetime | None,
    end_in: datetime | None,
    *,
    default_hours: int,
    max_hours: int,
) -> tuple[datetime, datetime]:
    """Resolve optional start/end values and enforce the recorder lookback cap."""
    now = dt_util.utcnow()
    end = dt_util.as_utc(end_in or now)
    start = dt_util.as_utc(start_in or (end - timedelta(hours=default_hours)))
    if start > end:
        raise _RecoverableToolError("invalid_tool_input", {"error": "start after end"})
    if end - start > timedelta(hours=max_hours):
        raise _RecoverableToolError(TIME_WINDOW_TOO_LARGE, {"max_hours": str(max_hours)})
    return start, end


async def _run_query[T](
    hass: HomeAssistant,
    deadline: float,
    fn: Callable[..., T],
    /,
    *args: object,
) -> T:
    """Run a blocking recorder query on the recorder executor with the tool deadline."""
    remaining = max(0.1, deadline - time.monotonic())
    return await asyncio.wait_for(get_instance(hass).async_add_executor_job(fn, *args), timeout=remaining)


def _truncate[T](values: list[T], limit: int) -> tuple[list[T], bool]:
    """Keep the most recent values when a result list exceeds its limit."""
    return (values[-limit:], True) if len(values) > limit else (values, False)


def _state_row_to_dict(row: State | dict[str, object]) -> dict[str, object]:
    """Convert a recorder history row to the SafeState-compatible history shape."""
    if isinstance(row, State):
        return cast(
            dict[str, object],
            json_safe(
                {
                    "entity_id": row.entity_id,
                    "state": row.state,
                    "attributes": dict(row.attributes),
                    "last_changed": row.last_changed.isoformat(),
                    "last_updated": row.last_updated.isoformat(),
                }
            ),
        )

    shaped = dict(row)
    for key in ("last_changed", "last_updated"):
        value = shaped.get(key)
        if isinstance(value, datetime):
            shaped[key] = value.isoformat()
    return cast(dict[str, object], json_safe(shaped))


def _statistic_row_to_dict(row: dict[str, object]) -> dict[str, object]:
    """Convert recorder statistic timestamps to ISO strings before JSON shaping."""
    shaped = dict(row)
    for key in ("start", "end", "last_reset"):
        value = shaped.get(key)
        if isinstance(value, datetime):
            shaped[key] = dt_util.as_utc(value).isoformat()
        elif isinstance(value, int | float):
            shaped[key] = datetime.fromtimestamp(value, UTC).isoformat()
    return cast(dict[str, object], json_safe(shaped))


@final
class GetHistoryTool(_RecorderTool):
    """Return recorded state history for visible entities."""

    name = TOOL_GET_HISTORY
    description = build_get_history_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required("entity_ids", description="One or up to 20 entity IDs."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            vol.Optional("start", description="Window start (ISO-8601). Default now-1h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
        }
    )

    @override
    async def _query(
        self,
        hass: HomeAssistant,
        snapshot: HomeSnapshot,
        settings: SandboxSettings,
        deadline: float,
        data: dict[str, object],
    ) -> JsonObjectType:
        entity_ids = [entity_id.lower() for entity_id in cast(list[str], data["entity_ids"])]
        _validate_visibility(snapshot, entity_ids)
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        result = await _run_query(
            hass,
            deadline,
            history.get_significant_states,
            hass,
            start,
            end,
            entity_ids,
            None,
            True,
            True,
            False,
            False,
            False,
        )

        budget = max(1, MAX_HISTORY_STATES // len(entity_ids))
        truncated = False
        entities: dict[str, list[dict[str, object]]] = {}
        for entity_id, states in result.items():
            rows = [_state_row_to_dict(row) for row in states]
            rows, cut = _truncate(rows, budget)
            truncated = truncated or cut
            entities[entity_id] = rows

        return cast(
            JsonObjectType,
            json_safe(
                {
                    "status": "ok",
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "entities": entities,
                    "truncated": truncated,
                }
            ),
        )


@final
class GetStatisticsTool(_RecorderTool):
    """Return long-term recorder statistics for visible statistic IDs."""

    name = TOOL_GET_STATISTICS
    description = build_get_statistics_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required(
                "statistic_ids",
                description="One or up to 20 statistic IDs (usually entity IDs).",
            ): vol.All(
                cv.ensure_list,
                [str],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            vol.Optional("start", description="Window start (ISO-8601). Default now-24h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
            vol.Optional("period", default="hour", description="Aggregation bucket."): vol.In(
                ("5minute", "hour", "day")
            ),
        }
    )

    @override
    async def _query(
        self,
        hass: HomeAssistant,
        snapshot: HomeSnapshot,
        settings: SandboxSettings,
        deadline: float,
        data: dict[str, object],
    ) -> JsonObjectType:
        statistic_ids = [statistic_id.lower() for statistic_id in cast(list[str], data["statistic_ids"])]
        _validate_visibility(snapshot, statistic_ids)
        period = cast(Literal["5minute", "hour", "day"], data["period"])
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            default_hours=DEFAULT_STATISTICS_WINDOW_HOURS,
            max_hours=MAX_STATISTICS_LOOKBACK_HOURS,
        )
        result = await _run_query(
            hass,
            deadline,
            statistics.statistics_during_period,
            hass,
            start,
            end,
            set(statistic_ids),
            period,
            None,
            {"last_reset", "max", "mean", "min", "state", "sum"},
        )

        budget = max(1, MAX_STATISTICS_ROWS // len(statistic_ids))
        truncated = False
        shaped_statistics: dict[str, list[dict[str, object]]] = {}
        for statistic_id, rows in result.items():
            shaped_rows = [_statistic_row_to_dict(cast(dict[str, object], row)) for row in rows]
            shaped_rows, cut = _truncate(shaped_rows, budget)
            truncated = truncated or cut
            shaped_statistics[statistic_id] = shaped_rows

        return cast(
            JsonObjectType,
            json_safe(
                {
                    "status": "ok",
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "period": period,
                    "statistics": shaped_statistics,
                    "truncated": truncated,
                }
            ),
        )


@final
class GetLogbookTool(_RecorderTool):
    """Return logbook events for visible entities."""

    name = TOOL_GET_LOGBOOK
    description = build_get_logbook_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required("entity_ids", description="One or up to 20 entity IDs to scope the logbook."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            vol.Optional("start", description="Window start (ISO-8601). Default now-24h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
        }
    )

    @override
    async def _query(
        self,
        hass: HomeAssistant,
        snapshot: HomeSnapshot,
        settings: SandboxSettings,
        deadline: float,
        data: dict[str, object],
    ) -> JsonObjectType:
        entity_ids = [entity_id.lower() for entity_id in cast(list[str], data["entity_ids"])]
        _validate_visibility(snapshot, entity_ids)
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        if LOGBOOK_DOMAIN not in hass.data:
            raise _RecoverableToolError("logbook_unavailable", {})

        event_types = async_determine_event_types(hass, entity_ids, None)
        processor = EventProcessor(
            hass, event_types, entity_ids, None, None, timestamp=False, include_entity_name=True
        )
        entries = await _run_query(hass, deadline, processor.get_events, start, end)
        entries, truncated = _truncate(entries, MAX_LOGBOOK_ENTRIES)

        return cast(
            JsonObjectType,
            json_safe(
                {
                    "status": "ok",
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "entries": entries,
                    "truncated": truncated,
                }
            ),
        )
