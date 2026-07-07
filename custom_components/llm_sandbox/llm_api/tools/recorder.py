"""Recorder-backed read-only LLM tools."""

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Mapping
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
    MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    MAX_HISTORY_ATTRIBUTES,
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
from ...snapshot import build_recorder_snapshot
from ...snapshot.models import HomeSnapshot
from ...types import TranslationPlaceholders
from .._hinting import error_guidance
from ..errors import RecoverableToolError, tool_error_envelope, tool_error_from_exception
from ..executor_support import json_safe
from ..prompts import build_get_history_description, build_get_logbook_description, build_get_statistics_description
from ..resolution import _DISCOVERY_LIMIT, bounded_strings, candidates_for_domain, resolve_target_entity
from ..selector_expansion import expand_aggregate_selectors
from ._analytics import (
    AGGREGATORS,
    AggregateFilters,
    AggregateMode,
    HistoryRow,
    analytics_spec_from_data,
    flat_history_rows,
    run_analytics,
)
from ._cursor import _LOGBOOK_CURSOR_KEY, INVALID_CURSOR, Cursor, decode_cursor, encode_cursor, paginate_stream
from ._support import _require_loaded_entry_error, _require_sandbox_runtime

RECORDER_UNAVAILABLE = "recorder_unavailable"
ENTITY_NOT_VISIBLE = "entity_not_visible"
SELECTOR_NO_MATCH = "selector_no_match"
TIME_WINDOW_TOO_LARGE = "time_window_too_large"
QUERY_FAILED = "query_failed"
type StatisticValueType = Literal["mean", "min", "max", "state", "sum"]
type StatisticQueryType = Literal["change", "last_reset", "max", "mean", "min", "state", "sum"]

STATISTIC_VALUE_TYPES: tuple[StatisticValueType, ...] = ("mean", "min", "max", "state", "sum")

_DEFAULT_STATISTIC_VALUE_PRIORITY: tuple[StatisticValueType, ...] = ("state", "mean", "sum", "min", "max")
_ALL_STAT_QUERY_TYPES: frozenset[StatisticQueryType] = frozenset({"last_reset", "max", "mean", "min", "state", "sum"})

# Relative window size in hours, accepted by every recorder tool as an
# alternative to absolute ISO start/end (the sandbox forbids timedelta math).
_HOURS_ARG = vol.All(vol.Coerce(float), vol.Range(min=0))

# Actionable guidance keyed by the recoverable error key. Entity visibility
# errors are snapshot-specific and are handled separately so they can include
# concrete visible candidates for the requested domain.
_RECORDER_GUIDANCE: dict[str, tuple[str, list[str]]] = {
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
    INVALID_CURSOR: (
        "The pagination cursor is invalid or expired.",
        ["Re-issue the original query without a cursor to start a new page sequence."],
    ),
    "invalid_tool_input": (
        "Invalid tool input.",
        ["Check argument names and types; the validation error was: {error}."],
    ),
    "analytics_unknown_op": (
        "The requested analytics operation is not supported.",
        ["Use one of: {valid}."],
    ),
    "analytics_unknown_group_key": (
        "The requested analytics group key is not supported.",
        ["Use one of: {valid}."],
    ),
    "analytics_bad_bucket": (
        "The requested analytics bucket is invalid.",
        ["Use bucket examples like {examples}."],
    ),
}


def _candidate_ids(snapshot: HomeSnapshot, requested_entity_id: str) -> list[str] | None:
    """Return deterministic visible candidates for an invisible requested entity."""
    domain = requested_entity_id.split(".", 1)[0]
    resolution = resolve_target_entity(snapshot, requested_entity_id, domain)
    if resolution.resolved is not None:
        ids = [resolution.resolved]
    elif resolution.candidates:
        ids = [candidate.entity_id for candidate in resolution.candidates]
    else:
        candidates = candidates_for_domain(snapshot, domain, limit=_DISCOVERY_LIMIT + 1)
        ids = [candidate.entity_id for candidate in candidates]
    if not ids:
        return None
    return bounded_strings(sorted(ids))


def _selector_candidate_ids(snapshot: HomeSnapshot, selectors: str) -> list[str] | None:
    """Return bounded candidate ids for the provided location selector field names."""
    id_pools: list[str] = []
    for field in selectors.split(", "):
        # Match the snapshot id universe for each requested selector type.
        if field == "area_id":
            id_pools.extend(snapshot.areas)
        elif field == "device_id":
            id_pools.extend(snapshot.devices)
        elif field == "floor_id":
            id_pools.extend(snapshot.floors)
        elif field == "label_id":
            id_pools.extend(snapshot.labels)
    if not id_pools:
        return None
    return bounded_strings(sorted(set(id_pools)))


def recorder_error_envelope(
    key: str,
    placeholders: TranslationPlaceholders,
    snapshot: HomeSnapshot | None = None,
) -> JsonObjectType:
    """Build a recoverable recorder error envelope with actionable guidance."""
    if key == ENTITY_NOT_VISIBLE:
        entity_id = placeholders.get("entity_id", "the requested entity")
        fix = _candidate_ids(snapshot, entity_id) if snapshot is not None else None
        return tool_error_envelope(
            key,
            placeholders,
            message=f"Entity '{entity_id}' is not visible to this LLM tool.",
            fix=fix,
        )
    if key == SELECTOR_NO_MATCH:
        selectors = placeholders.get("selectors", "")
        fix = _selector_candidate_ids(snapshot, selectors) if snapshot is not None else None
        return tool_error_envelope(
            key,
            placeholders,
            message=f"Selector(s) {selectors or 'requested'} matched no visible entities.",
            fix=fix,
        )
    message, fix = error_guidance(_RECORDER_GUIDANCE, key, placeholders)
    return tool_error_envelope(key, placeholders, message=message, fix=fix)


def recorder_available(hass: HomeAssistant) -> bool:
    """Return whether the recorder integration has a live recorder instance."""
    return DATA_INSTANCE in hass.data


def logbook_available(hass: HomeAssistant) -> bool:
    """Return whether the logbook integration has registered its runtime config."""
    return LOGBOOK_DOMAIN in hass.data


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
            data = cast(dict[str, object], self.parameters(self._normalize_args(tool_input.tool_args)))
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            return recorder_error_envelope(*mapped)

        if not recorder_available(hass):
            return recorder_error_envelope(RECORDER_UNAVAILABLE, {})

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            key, placeholders = setup_error
            return recorder_error_envelope(key, placeholders)
        settings = _require_sandbox_runtime(hass, self.entry_id).settings
        # Build a fresh visible snapshot for every recorder tool call.
        snapshot = build_recorder_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        deadline = time.monotonic() + settings.execution_timeout_seconds

        try:
            return await self._query(hass, snapshot, settings, deadline, data)
        except RecoverableToolError as err:
            return recorder_error_envelope(err.key, err.placeholders, snapshot)
        except Exception as err:  # noqa: BLE001 - recorder tools map unexpected query failures to envelopes
            mapped = tool_error_from_exception(err)
            if mapped is None:
                return recorder_error_envelope(QUERY_FAILED, {"error": type(err).__name__})
            return recorder_error_envelope(*mapped)

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]:
        """Normalize tool-specific input aliases before voluptuous validation."""
        return dict(args)

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
# Location-backed selectors resolved through snapshot indexes; ``domain`` is a
# filter, not a location selector, so it is excluded from selector-presence checks.
_LOCATION_SELECTOR_FIELDS = ("area_id", "device_id", "floor_id", "label_id")
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
    the error). Location selectors (area/device/floor/label) expand to visible
    entities and union across selector types. A selector that is present but
    matches nothing raises ``selector_no_match`` with candidate ids rather than
    widening (e.g. a typo'd ``area_id`` plus ``domain`` would otherwise silently
    expand to every matching-domain entity in the home). ``domain`` filters the
    resolved set and, when no IDs or selectors are given, expands across all
    visible states of that domain.
    """
    explicit = [entity_id.lower() for entity_id in _as_list(data.get(id_key))]
    # Explicit IDs must each be visible (named in the error so the LLM can correct).
    _validate_visibility(snapshot, explicit)
    domains = {domain.lower() for domain in _as_list(data.get("domain"))}
    provided_selectors = [field for field in _LOCATION_SELECTOR_FIELDS if _as_list(data.get(field))]
    selector_present = bool(provided_selectors)

    selector_ids: list[str] = []
    for requested_expansions in expand_aggregate_selectors(
        snapshot,
        data,
        selector_keys=("area_id", "device_id", "label_id", "floor_id"),
    ).values():
        for _requested, expanded_ids in requested_expansions:
            selector_ids.extend(expanded_ids)

    # A present selector resolving to nothing is a naming error (e.g. a typo'd
    # area_id), not a cue to widen. Name the selector and surface candidate ids
    # so the LLM can correct, mirroring the explicit-id visibility error.
    if selector_present and not selector_ids:
        raise RecoverableToolError(
            SELECTOR_NO_MATCH,
            {"selectors": ", ".join(provided_selectors)},
        )

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
    # Pure-domain scope with no IDs and no selectors expands across all visible matching states.
    if not resolved and domains and not selector_present:
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


def _resolve_window(
    data: dict[str, object],
    *,
    default_hours: int,
    max_hours: int,
) -> tuple[datetime, datetime, Cursor]:
    """Return the stable query window and decoded cursor for one recorder page."""
    if (cursor_in := data.get("cursor")) is not None:
        # Cursor window wins while paging so the continuation is stable.
        cursor = decode_cursor(cursor_in)
        return cursor.start, cursor.end, cursor
    start, end = _clamp_window(
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        hours=cast(float | None, data.get("hours")),
        default_hours=default_hours,
        max_hours=max_hours,
    )
    return start, end, Cursor(start=start, end=end, cutoffs={})


async def fetch_visible_history_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[HistoryRow]]:
    """Fetch significant recorder history for visible snapshot entity ids."""
    _validate_visibility(snapshot, entity_ids)
    return cast(
        dict[str, list[HistoryRow]],
        await _run_query(
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
        ),
    )


async def fetch_flat_history_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Fetch JSON-compatible flat history rows for facade helpers and SQL."""
    result = await fetch_visible_history_rows(
        hass,
        snapshot,
        deadline,
        entity_ids,
        start,
        end,
    )
    return flat_history_rows(result, snapshot)


async def fetch_flat_statistics_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    statistic_ids: list[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Fetch JSON-compatible long-term statistics rows for facade SQL."""
    _validate_visibility(snapshot, statistic_ids)
    result = await _run_query(
        hass,
        deadline,
        functools.partial(
            statistics.statistics_during_period,
            hass=hass,
            start_time=start,
            end_time=end,
            statistic_ids=set(statistic_ids),
            period="hour",
            units=None,
            types=set(_ALL_STAT_QUERY_TYPES),
        ),
    )
    rows: list[dict[str, object]] = []
    for statistic_id, values in cast(Mapping[str, list[dict[str, object]]], result).items():
        for row in values:
            timestamp = row.get("start") or row.get("end") or row.get("last_reset")
            if isinstance(timestamp, datetime):
                timestamp = dt_util.as_utc(timestamp).isoformat()
            rows.append(
                {
                    "statistic_id": statistic_id,
                    "when": str(timestamp),
                    "mean": row.get("mean"),
                    "min": row.get("min"),
                    "max": row.get("max"),
                    "state": row.get("state"),
                    "sum": row.get("sum"),
                }
            )
    return rows


def _paginate_streams(
    streams: Mapping[str, list[list[object]]],
    *,
    budget: int,
    cutoffs: Mapping[str, str],
) -> tuple[dict[str, list[list[object]]], dict[str, str]]:
    """Page each timestamp-first stream independently and return next cutoffs.

    Exhausted streams carry the ``""`` sentinel instead of being dropped, so a
    later page treats them as empty (``ts < ""`` is always false) rather than
    re-querying them as brand-new and re-emitting their newest rows as
    duplicates. When every stream is sentinel-exhausted the map collapses to
    ``{}`` so callers' ``if next_cutoffs`` checks stop paging.
    """
    pages: dict[str, list[list[object]]] = {}
    next_cutoffs: dict[str, str] = {}
    for stream_id, rows in streams.items():
        page, next_cutoff = paginate_stream(
            rows,
            ts_of=_row_timestamp,
            budget=budget,
            cutoff_iso=cutoffs.get(stream_id),
        )
        pages[stream_id] = page
        # "" sentinel marks an exhausted stream so it yields an empty page next
        # time instead of being re-queried from scratch.
        next_cutoffs[stream_id] = next_cutoff if next_cutoff is not None else ""
    # Collapse to {} once every stream is exhausted so paging terminates.
    if next_cutoffs and all(cutoff == "" for cutoff in next_cutoffs.values()):
        return pages, {}
    return pages, next_cutoffs


def _windowed_payload(
    start: datetime,
    end: datetime,
    body: Mapping[str, object],
    next_cursor: Cursor | None,
) -> JsonObjectType:
    """Return a JSON-safe recorder payload with a stable window envelope."""
    payload: dict[str, object] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        **body,
    }
    if next_cursor is not None:
        payload["next_cursor"] = encode_cursor(next_cursor)
    return cast(JsonObjectType, json_safe(payload))


async def _run_query[T](
    hass: HomeAssistant,
    deadline: float,
    fn: Callable[[], T],
) -> T:
    """Run a blocking recorder query on the recorder executor with the tool deadline."""
    recorder_instance = get_instance(hass)
    # Drain the HA event loop first so state-change events are dispatched to the
    # recorder listener and queued; otherwise the recorder sync below could see an
    # empty queue, early-return, and skip the commit that makes writes visible.
    await _await_deadline(hass.async_block_till_done(), deadline)
    # async_block_till_done queues a SynchronizeTask (commit_before=True), forcing a
    # session.commit() and resolving only after the recorder thread has run it, so the
    # fresh read session sees all writes. The sync block_till_done must NOT be used
    # here: it queues WaitTask (commit_before=False) and drains without committing.
    await _await_deadline(recorder_instance.async_block_till_done(), deadline)
    # Drain once more in case the commit resolution scheduled further loop work.
    await _await_deadline(hass.async_block_till_done(), deadline)
    # A final public recorder sync catches state-change events that were queued
    # by the post-commit loop drain above, preserving read-after-write visibility
    # without private recorder APIs or sync block_till_done.
    await _await_deadline(recorder_instance.async_block_till_done(), deadline)
    await _await_deadline(hass.async_block_till_done(), deadline)
    return await _await_deadline(recorder_instance.async_add_executor_job(fn), deadline)


async def _await_deadline[T](awaitable: Awaitable[T], deadline: float) -> T:
    """Await one recorder sync/query step within the remaining tool deadline."""
    return await asyncio.wait_for(awaitable, timeout=max(0, deadline - time.monotonic()))


def _state_row_to_dict(
    row: State | dict[str, object],
    attributes: list[str] | None = None,
) -> tuple[list[object], str | None]:
    """Convert a recorder history row to compact ``(row, unit)`` shape."""
    if isinstance(row, State):
        unit = row.attributes.get("unit_of_measurement") or row.attributes.get("unit")
        shaped_row: list[object] = [row.last_changed.isoformat(), row.state]
        if attributes is not None:
            # Attribute rows use a stable third element, omitting absent requested keys.
            shaped_row.append({name: row.attributes[name] for name in attributes if name in row.attributes})
        return shaped_row, str(unit) if unit is not None else None

    shaped = dict(row)
    timestamp = shaped.get("last_changed") or shaped.get("last_updated")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    row_attributes = shaped.get("attributes")
    unit = None
    if isinstance(row_attributes, Mapping):
        unit = row_attributes.get("unit_of_measurement") or row_attributes.get("unit")
    shaped_row = [str(timestamp), shaped.get("state")]
    if attributes is not None:
        # Keep the third row slot present even when none of the requested names exist.
        present: dict[str, object] = {}
        if isinstance(row_attributes, Mapping):
            present = {name: row_attributes[name] for name in attributes if name in row_attributes}
        shaped_row.append(present)
    return shaped_row, str(unit) if unit is not None else None


def _row_timestamp(row: list[object]) -> str:
    """Return the ISO timestamp in a compact timestamp-first row."""
    return str(row[0])


def _statistic_row_to_dict(
    row: dict[str, object], requested_types: tuple[StatisticValueType, ...] | None = None
) -> list[object]:
    """Convert one recorder statistic row to a compact timestamp/keyed-values array."""
    shaped = dict(row)
    timestamp = shaped.get("start") or shaped.get("end") or shaped.get("last_reset")
    if isinstance(timestamp, datetime):
        timestamp = dt_util.as_utc(timestamp).isoformat()
    elif isinstance(timestamp, int | float):
        timestamp = datetime.fromtimestamp(timestamp, UTC).isoformat()
    value_keys = requested_types or _DEFAULT_STATISTIC_VALUE_PRIORITY
    values: dict[str, object] = {}
    for key in value_keys:
        value = shaped.get(key)
        if value is None:
            continue
        values[key] = value
        if requested_types is None:
            break
    return [str(timestamp), values]


def _statistic_fields(rows: list[list[object]]) -> list[str]:
    """Return sorted statistic value keys present in one shaped page."""
    fields: set[str] = set()
    for row in rows:
        values = row[1]
        if isinstance(values, Mapping):
            fields.update(str(key) for key in values)
    return sorted(fields)


def _logbook_when(entry: dict[str, object]) -> str:
    """Return the ISO timestamp of a logbook entry for pagination."""
    when = entry["when"]
    if isinstance(when, datetime):
        return dt_util.as_utc(when).isoformat()
    return str(when)


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
            vol.Optional(
                "attributes",
                description=(
                    "Optional attribute names to include per row. Each row then appends "
                    "{name: value} for requested attributes present on that row; absent names are omitted."
                ),
            ): vol.All(cv.ensure_list, [str], vol.Length(min=1, max=MAX_HISTORY_ATTRIBUTES)),
            vol.Optional(
                "aggregate",
                description="Optional server-side summary mode instead of raw rows.",
            ): vol.Any(vol.In(tuple(AGGREGATORS)), dict),
            vol.Optional("group_by", description="Optional analytics group key(s)."): vol.All(
                cv.ensure_list, [str], vol.Length(min=1, max=4)
            ),
            vol.Optional("bucket", description="Optional analytics bucket, e.g. 15m, 1h, 1d."): str,
            vol.Optional("where", description="Optional analytics row filters."): vol.All(cv.ensure_list, [dict]),
            vol.Optional("order_by", description="Optional analytics result sort field; prefix '-' for desc."): str,
            vol.Optional("limit", description="Maximum analytics result rows."): vol.All(
                vol.Coerce(int), vol.Range(min=1)
            ),
            vol.Optional(
                "from_state",
                description="Optional count_transitions filter for the previous state.",
            ): str,
            vol.Optional(
                "to_state",
                description="Optional count_transitions filter for the next state.",
            ): str,
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from a prior next_cursor; pass it to fetch the next older page. "
                    "Omit on the first call."
                ),
            ): str,
        }
    )

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]:
        """Accept the requested history analytics input-key synonyms."""
        data = dict(args)
        if "agg" in data and "aggregate" not in data:
            data["aggregate"] = data.pop("agg")
        if "groupby" in data and "group_by" not in data:
            data["group_by"] = data.pop("groupby")
        if "resample" in data and "bucket" not in data:
            data["bucket"] = data.pop("resample")
        if "interval" in data and "bucket" not in data:
            data["bucket"] = data.pop("interval")
        return data

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
        requested_attributes = cast(list[str] | None, data.get("attributes"))
        aggregate = cast(AggregateMode | None, data.get("aggregate"))
        analytics_requested = any(
            key in data for key in ("aggregate", "group_by", "bucket", "where", "order_by", "limit")
        )
        if isinstance(aggregate, str) and not any(
            key in data for key in ("group_by", "bucket", "where", "order_by", "limit")
        ):
            return await _aggregate_history(hass, deadline, entity_ids, data, aggregate)
        if analytics_requested:
            return await _declarative_history(hass, snapshot, deadline, entity_ids, data)

        start, end, cursor = _resolve_window(
            data,
            default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        result = await fetch_visible_history_rows(
            hass,
            snapshot,
            deadline,
            entity_ids,
            start,
            end,
        )
        budget = max(1, MAX_HISTORY_STATES // len(entity_ids))
        stream_rows: dict[str, list[list[object]]] = {}
        stream_units: dict[str, str] = {}
        for entity_id, states in result.items():
            converted = [_state_row_to_dict(row, requested_attributes) for row in states]
            stream_rows[entity_id] = [row for row, _unit in converted]
            unit = next((unit for _row, unit in converted if unit), None)
            if unit:
                stream_units[entity_id] = unit
        pages, next_cutoffs = _paginate_streams(stream_rows, budget=budget, cutoffs=cursor.cutoffs)
        entities: dict[str, dict[str, object]] = {}
        for entity_id, page in pages.items():
            entity: dict[str, object] = {"rows": page}
            if (unit := stream_units.get(entity_id)) is not None:
                entity["unit"] = unit
            entities[entity_id] = entity

        return _windowed_payload(
            start,
            end,
            {"entities": entities},
            Cursor(start=start, end=end, cutoffs=next_cutoffs) if next_cutoffs else None,
        )


async def _aggregate_history(
    hass: HomeAssistant,
    deadline: float,
    entity_ids: list[str],
    data: dict[str, object],
    aggregate: AggregateMode,
) -> JsonObjectType:
    """Return a compact server-side history summary for aggregate mode."""
    if data.get("attributes") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "aggregate cannot be combined with attributes"})
    if data.get("cursor") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "aggregate cannot be combined with cursor"})

    start, end = _clamp_window(
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        hours=cast(float | None, data.get("hours")),
        default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
        max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    )
    filters = AggregateFilters(
        from_state=cast(str | None, data.get("from_state")),
        to_state=cast(str | None, data.get("to_state")),
    )

    def _fetch_and_aggregate() -> dict[str, dict[str, object]]:
        result = history.get_significant_states(
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
        )
        aggregator = AGGREGATORS[aggregate]
        return {
            entity_id: aggregator(cast(list[HistoryRow], list(result.get(entity_id, ()))), start, end, filters)
            for entity_id in entity_ids
        }

    summary = await _run_query(hass, deadline, _fetch_and_aggregate)
    payload: dict[str, object] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "mode": aggregate,
        "summary": summary,
    }
    return cast(
        JsonObjectType,
        json_safe(payload),
    )


async def _declarative_history(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    data: dict[str, object],
) -> JsonObjectType:
    """Return flat declarative analytics results for get_history."""
    if data.get("attributes") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "analytics cannot be combined with attributes"})
    if data.get("cursor") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "analytics cannot be combined with cursor"})
    start, end = _clamp_window(
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        hours=cast(float | None, data.get("hours")),
        default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
        max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    )
    raw = await fetch_visible_history_rows(hass, snapshot, deadline, entity_ids, start, end)
    flat = flat_history_rows(raw, snapshot)
    spec = analytics_spec_from_data(data)
    rows = run_analytics(cast(list[HistoryRow], flat), spec, (start, end), snapshot)
    return cast(
        JsonObjectType,
        json_safe({"window": {"start": start.isoformat(), "end": end.isoformat()}, "rows": rows}),
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
            vol.Optional(
                "types",
                description="Optional statistic value fields to include per row.",
            ): vol.All(
                cv.ensure_list,
                [vol.In(STATISTIC_VALUE_TYPES)],
                vol.Length(min=1, max=len(STATISTIC_VALUE_TYPES)),
            ),
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from a prior next_cursor; pass it to fetch the next older page. "
                    "Omit on the first call."
                ),
            ): str,
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
        start, end, cursor = _resolve_window(
            data,
            default_hours=DEFAULT_STATISTICS_WINDOW_HOURS,
            max_hours=MAX_STATISTICS_LOOKBACK_HOURS,
        )
        if data.get("cursor") is not None:
            # Cursor carries period and selected fields so pagination stays consistent.
            if cursor.period not in (None, "5minute", "hour", "day"):
                raise RecoverableToolError(INVALID_CURSOR, {})
            period = cast(Literal["5minute", "hour", "day"], cursor.period or "hour")
            requested_types = cast(tuple[StatisticValueType, ...] | None, cursor.statistic_types)
            if requested_types is not None and (
                not requested_types or set(requested_types) - set(STATISTIC_VALUE_TYPES)
            ):
                raise RecoverableToolError(INVALID_CURSOR, {})
        else:
            period = cast(Literal["5minute", "hour", "day"], data["period"])
            requested_types = cast(
                tuple[StatisticValueType, ...] | None,
                tuple(cast(list[str] | None, data.get("types")) or ()) or None,
            )
            cursor = Cursor(start=start, end=end, cutoffs={}, period=period, statistic_types=requested_types)
        query_types: set[StatisticQueryType] = (
            set(cast(tuple[StatisticQueryType, ...], requested_types))
            if requested_types is not None
            else set(_ALL_STAT_QUERY_TYPES)
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
                types=query_types,
            ),
        )

        budget = max(1, MAX_STATISTICS_ROWS // len(statistic_ids))
        stream_rows = {
            statistic_id: [_statistic_row_to_dict(cast(dict[str, object], row), requested_types) for row in rows]
            for statistic_id, rows in result.items()
        }
        pages, next_cutoffs = _paginate_streams(stream_rows, budget=budget, cutoffs=cursor.cutoffs)
        shaped_statistics = {
            statistic_id: {"rows": page, "fields": _statistic_fields(page)} for statistic_id, page in pages.items()
        }

        return _windowed_payload(
            start,
            end,
            {"period": period, "statistics": shaped_statistics},
            Cursor(start=start, end=end, cutoffs=next_cutoffs, period=period, statistic_types=requested_types)
            if next_cutoffs
            else None,
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
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from a prior next_cursor; pass it to fetch the next older page. "
                    "Omit on the first call."
                ),
            ): str,
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
        start, end, cursor = _resolve_window(
            data,
            default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
        if not logbook_available(hass):
            raise RecoverableToolError("logbook_unavailable", {})

        event_types = async_determine_event_types(hass, entity_ids, None)
        processor = EventProcessor(
            hass, event_types, entity_ids, None, None, timestamp=False, include_entity_name=True
        )
        raw_entries = await _run_query(
            hass,
            deadline,
            functools.partial(processor.get_events, start_day=start, end_day=end),
        )
        entries = [dict(entry) for entry in raw_entries]
        # A logbook page is one flat stream keyed by the sentinel cutoff.
        page_entries, next_cutoff = paginate_stream(
            entries,
            ts_of=_logbook_when,
            budget=MAX_LOGBOOK_ENTRIES,
            cutoff_iso=cursor.cutoffs.get(_LOGBOOK_CURSOR_KEY),
        )
        return _windowed_payload(
            start,
            end,
            {"entries": page_entries},
            Cursor(start=start, end=end, cutoffs={_LOGBOOK_CURSOR_KEY: next_cutoff})
            if next_cutoff is not None
            else None,
        )
