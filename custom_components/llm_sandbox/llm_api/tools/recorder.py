"""Recorder-backed read-only LLM tools."""

import asyncio
import functools
import time
from collections.abc import Callable, Mapping
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

from ...const import (
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
from ...runtime import SandboxSettings
from ...snapshot import build_snapshot
from ...snapshot.models import HomeSnapshot
from ...types import TranslationPlaceholders
from ..errors import tool_error_envelope, tool_error_from_exception
from ..executor_support import json_safe
from ..prompts import build_get_history_description, build_get_logbook_description, build_get_statistics_description
from ._support import _require_loaded_entry, _require_loaded_entry_error

RECORDER_UNAVAILABLE = "recorder_unavailable"
ENTITY_NOT_VISIBLE = "entity_not_visible"
TIME_WINDOW_TOO_LARGE = "time_window_too_large"
QUERY_FAILED = "query_failed"

# Relative window size in hours, accepted by every recorder tool as an
# alternative to absolute ISO start/end (the sandbox forbids timedelta math).
_HOURS_ARG = vol.All(vol.Coerce(float), vol.Range(min=0))

# Actionable guidance keyed by the recoverable error key. Message/hints are
# surfaced inline to the LLM so a follow-up call can succeed; stable keys stay
# translated in en.json for the human-facing contract.
_RECORDER_GUIDANCE: dict[str, tuple[str, list[str]]] = {
    ENTITY_NOT_VISIBLE: (
        "Only snapshot-visible entities are queryable.",
        ["List visible entities via execute_home_code (hass.states.async_all()) and retry with visible IDs."],
    ),
    TIME_WINDOW_TOO_LARGE: (
        "The requested time window is too large.",
        ["Reduce the window to at most {max_hours} hours.", "Pass hours=<n> or a smaller start/end range."],
    ),
    RECORDER_UNAVAILABLE: (
        "The recorder integration is not available.",
        ["Ask the user to enable the recorder integration, or query live state via execute_home_code instead."],
    ),
    "logbook_unavailable": (
        "The logbook integration is not available.",
        ["Ask the user to enable the logbook integration, or query history via get_history instead."],
    ),
    QUERY_FAILED: (
        "The recorder query failed.",
        ["Check the argument values; the recorder error was: {error}."],
    ),
    "invalid_tool_input": (
        "Invalid tool input.",
        ["Check argument names and types; the validation error was: {error}."],
    ),
}


class _SafeHintDict(dict[str, str]):
    """dict that keeps unknown ``{placeholder}`` tokens verbatim instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _error_guidance(key: str, placeholders: Mapping[str, str]) -> tuple[str | None, list[str] | None]:
    """Return (message, hints) for a recorder error key, formatting placeholders into hints."""
    entry = _RECORDER_GUIDANCE.get(key)
    if entry is None:
        return None, None
    message, templates = entry
    values = _SafeHintDict({str(k): str(v) for k, v in placeholders.items()})
    return message, [template.format_map(values) for template in templates]


def recorder_error_envelope(key: str, placeholders: TranslationPlaceholders) -> JsonObjectType:
    """Build a recoverable recorder error envelope with actionable guidance."""
    message, hints = _error_guidance(key, placeholders)
    return tool_error_envelope(key, placeholders, message=message, hints=hints)


@dataclass(frozen=True, slots=True)
class RecoverableToolError(Exception):
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
            return recorder_error_envelope(*mapped)

        if not _recorder_available(hass):
            return recorder_error_envelope(RECORDER_UNAVAILABLE, {})

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            key, placeholders = setup_error
            return recorder_error_envelope(key, placeholders)
        entry = _require_loaded_entry(hass, self.entry_id)
        runtime_data = entry.runtime_data
        assert runtime_data is not None
        settings = runtime_data.settings
        # Build a fresh visible snapshot for every recorder tool call.
        snapshot = build_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        deadline = time.monotonic() + settings.execution_timeout_seconds

        try:
            return await self._query(hass, snapshot, settings, deadline, data)
        except RecoverableToolError as err:
            return recorder_error_envelope(err.key, err.placeholders)
        except Exception as err:  # noqa: BLE001 - recorder tools map unexpected query failures to envelopes
            mapped = tool_error_from_exception(err)
            if mapped is None:
                return recorder_error_envelope(QUERY_FAILED, {"error": type(err).__name__})
            return recorder_error_envelope(*mapped)

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
            raise RecoverableToolError(ENTITY_NOT_VISIBLE, {"entity_id": entity_id})


def _as_list(value: object) -> list[str]:
    """Normalize a scalar/list selector value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


# HA-native target selectors accepted as an alternative to enumerated IDs.
RECORDER_SELECTOR_FIELD_NAMES = ("area_id", "device_id", "floor_id", "label_id", "domain")
_SELECTOR_FIELD_DESCRIPTIONS: dict[str, str] = {
    "area_id": "Area ID(s) to scope the query.",
    "device_id": "Device ID(s) to scope the query.",
    "floor_id": "Floor ID(s) to scope the query.",
    "label_id": "Label ID(s) to scope the query.",
    "domain": "Domain(s) (e.g. light) to keep from the resolved set.",
}
_SELECTOR_FIELDS: dict[vol.Optional, object] = {
    vol.Optional(field_name, description=_SELECTOR_FIELD_DESCRIPTIONS[field_name]): vol.All(cv.ensure_list, [str])
    for field_name in RECORDER_SELECTOR_FIELD_NAMES
}


def resolve_entity_ids(snapshot: HomeSnapshot, data: dict[str, object], id_key: str) -> list[str]:
    """Resolve explicit IDs plus HA-native selectors to visible entity IDs.

    Explicit IDs are validated for visibility (an invisible one names itself in
    the error). Selector expansion (area/device/floor/label) keeps only
    snapshot-visible entities; ``domain`` filters the selector expansion and,
    when no IDs are given, expands across all visible states of that domain.
    """
    indexes = snapshot.indexes
    explicit = [entity_id.lower() for entity_id in _as_list(data.get(id_key))]
    # Explicit IDs must each be visible (named in the error so the LLM can correct).
    _validate_visibility(snapshot, explicit)
    domains = {domain.lower() for domain in _as_list(data.get("domain"))}

    selector_ids: list[str] = []
    for area_id in _as_list(data.get("area_id")):
        selector_ids.extend(indexes.entity_ids_by_area_id.get(area_id, ()))
    for device_id in _as_list(data.get("device_id")):
        selector_ids.extend(indexes.entity_ids_by_device_id.get(device_id, ()))
    for label_id in _as_list(data.get("label_id")):
        selector_ids.extend(indexes.entity_ids_by_label.get(label_id, ()))
    for floor_id in _as_list(data.get("floor_id")):
        for area_id in indexes.area_ids_by_floor_id.get(floor_id, ()):
            selector_ids.extend(indexes.entity_ids_by_area_id.get(area_id, ()))

    def _domain_matches(entity_id: str) -> bool:
        return not domains or entity_id.split(".", 1)[0].lower() in domains

    seen: set[str] = set()
    resolved: list[str] = []
    # Explicit IDs are kept as-is (visibility already validated).
    for entity_id in explicit:
        if entity_id not in seen:
            seen.add(entity_id)
            resolved.append(entity_id)
    # Selector expansion keeps only visible entities honoring the domain filter.
    for entity_id in selector_ids:
        if entity_id in seen or entity_id not in snapshot.states or not _domain_matches(entity_id):
            continue
        seen.add(entity_id)
        resolved.append(entity_id)
    # Pure-domain scope with no IDs expands across all visible matching states.
    if not resolved and domains:
        for entity_id in snapshot.states:
            if _domain_matches(entity_id):
                resolved.append(entity_id)

    if not resolved:
        raise RecoverableToolError(
            "invalid_tool_input",
            {"error": "no visible entity IDs or scope selectors resolved"},
        )
    if len(resolved) > MAX_RECORDER_ENTITY_IDS:
        raise RecoverableToolError(
            "invalid_tool_input",
            {"error": f"scope resolves to {len(resolved)} entities; narrow it to at most {MAX_RECORDER_ENTITY_IDS}"},
        )
    return resolved


def _clamp_window(
    start_in: datetime | None,
    end_in: datetime | None,
    *,
    hours: float | None = None,
    default_hours: int,
    max_hours: int,
) -> tuple[datetime, datetime]:
    """Resolve start/end values, honoring an explicit window or a relative ``hours`` size.

    Precedence: explicit ``start``/``end`` win; otherwise a relative ``hours``
    size is applied against ``end``; otherwise the tool default window is used.
    The recorder lookback cap is always enforced.
    """
    now = dt_util.utcnow()
    end = dt_util.as_utc(end_in or now)
    if start_in is not None:
        start = dt_util.as_utc(start_in)
    elif hours is not None:
        start = end - timedelta(hours=hours)
    else:
        start = end - timedelta(hours=default_hours)
    if start > end:
        raise RecoverableToolError("invalid_tool_input", {"error": "start after end"})
    if end - start > timedelta(hours=max_hours):
        raise RecoverableToolError(TIME_WINDOW_TOO_LARGE, {"max_hours": str(max_hours)})
    return start, end


async def _run_query[T](
    hass: HomeAssistant,
    deadline: float,
    fn: Callable[[], T],
) -> T:
    """Run a blocking recorder query on the recorder executor with the tool deadline."""
    remaining = max(0.1, deadline - time.monotonic())
    return await asyncio.wait_for(get_instance(hass).async_add_executor_job(fn), timeout=remaining)


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
            vol.Optional("entity_ids", description="One or up to 20 entity IDs."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
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
        entity_ids = resolve_entity_ids(snapshot, data, "entity_ids")
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            hours=cast(float | None, data.get("hours")),
            default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        result = await _run_query(
            hass,
            deadline,
            functools.partial(
                history.get_significant_states,
                hass=hass,
                start_time=start,
                end_time=end,
                entity_ids=entity_ids,
                filters=None,
                include_start_time_state=True,
                significant_changes_only=True,
                minimal_response=False,
                no_attributes=False,
                compressed_state_format=False,
            ),
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
            vol.Optional(
                "statistic_ids",
                description="One or up to 20 statistic IDs (usually entity IDs).",
            ): vol.All(
                cv.ensure_list,
                [str],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
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
        statistic_ids = resolve_entity_ids(snapshot, data, "statistic_ids")
        period = cast(Literal["5minute", "hour", "day"], data["period"])
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            hours=cast(float | None, data.get("hours")),
            default_hours=DEFAULT_STATISTICS_WINDOW_HOURS,
            max_hours=MAX_STATISTICS_LOOKBACK_HOURS,
        )
        result = await _run_query(
            hass,
            deadline,
            functools.partial(
                statistics.statistics_during_period,
                hass=hass,
                start_time=start,
                end_time=end,
                statistic_ids=set(statistic_ids),
                period=period,
                units=None,
                types={"last_reset", "max", "mean", "min", "state", "sum"},
            ),
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
            vol.Optional("entity_ids", description="One or up to 20 entity IDs to scope the logbook."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
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
        entity_ids = resolve_entity_ids(snapshot, data, "entity_ids")
        start, end = _clamp_window(
            cast(datetime | None, data.get("start")),
            cast(datetime | None, data.get("end")),
            hours=cast(float | None, data.get("hours")),
            default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        if LOGBOOK_DOMAIN not in hass.data:
            raise RecoverableToolError("logbook_unavailable", {})

        event_types = async_determine_event_types(hass, entity_ids, None)
        processor = EventProcessor(
            hass, event_types, entity_ids, None, None, timestamp=False, include_entity_name=True
        )
        entries = await _run_query(
            hass,
            deadline,
            functools.partial(processor.get_events, start_day=start, end_day=end),
        )
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
