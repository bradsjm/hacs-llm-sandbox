"""Runtime query helpers for recorder-backed LLM tools."""

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from homeassistant.components.logbook import DOMAIN as LOGBOOK_DOMAIN
from homeassistant.components.logbook.helpers import async_determine_event_types
from homeassistant.components.logbook.processor import EventProcessor
from homeassistant.components.recorder import get_instance, history, statistics
from homeassistant.components.recorder.core import Recorder
from homeassistant.components.recorder.tasks import SynchronizeTask
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.recorder import DATA_INSTANCE
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from ...const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
)
from ...snapshot.models import HomeSnapshot
from ..data.history import (
    AGGREGATORS,
    AggregateFilters,
    AggregateMode,
    HistoryRow,
    analytics_spec_from_data,
    flat_history_rows,
    run_analytics,
)
from ..data.recorder_scope import _clamp_window, _validate_visibility
from ..errors import RecoverableToolError
from ..executor_support import json_safe, overflow_metadata
from ._cursor import INVALID_CURSOR, Cursor, decode_cursor, encode_cursor, paginate_stream

type StatisticValueType = Literal["mean", "min", "max", "state", "sum"]
type StatisticQueryType = Literal["change", "last_reset", "max", "mean", "min", "state", "sum"]

STATISTIC_VALUE_TYPES: tuple[StatisticValueType, ...] = ("mean", "min", "max", "state", "sum")

_DEFAULT_STATISTIC_VALUE_PRIORITY: tuple[StatisticValueType, ...] = ("state", "mean", "sum", "min", "max")
_ALL_STAT_QUERY_TYPES: frozenset[StatisticQueryType] = frozenset({"last_reset", "max", "mean", "min", "state", "sum"})

# Window anchor + async row fetchers. Production closures over a live hass
# (via _production_recorder_source); eval closures over frozen fixtures.
type HistoryRowStream = dict[str, list[HistoryRow]]
type StatisticsRowStream = Mapping[str, list[dict[str, object]]]
type LogbookEntryStream = list[dict[str, object]]

type HistoryFetcher = Callable[[list[str], datetime, datetime], Awaitable[HistoryRowStream]]
type StatisticsFetcher = Callable[[list[str], datetime, datetime, str, set[str]], Awaitable[StatisticsRowStream]]
type LogbookFetcher = Callable[[list[str], datetime, datetime], Awaitable[LogbookEntryStream]]
type BlockingRunner = Callable[[Callable[[], object]], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class RecorderSource:
    """Hass-free boundary carrying the window anchor and async row fetchers.

    ``now`` is the window anchor (production = ``dt_util.utcnow()``; eval =
    parsed ``snapshot.created_at``). Fetchers return the SAME row shapes the
    recorder cores consume today. The pure cores read ``now`` and call the
    fetchers without ever touching hass, the recorder instance, the executor,
    or the event loop.
    """

    now: datetime
    logbook_available: bool
    # Production uses the HA executor; eval uses asyncio.to_thread.
    run_in_executor: BlockingRunner
    fetch_history: HistoryFetcher
    fetch_statistics: StatisticsFetcher
    fetch_logbook: LogbookFetcher


def recorder_available(hass: HomeAssistant) -> bool:
    """Return whether the recorder integration has a live recorder instance."""
    return DATA_INSTANCE in hass.data


def logbook_available(hass: HomeAssistant) -> bool:
    """Return whether the logbook integration has registered its runtime config."""
    return LOGBOOK_DOMAIN in hass.data


def _resolve_window(
    data: dict[str, object],
    *,
    now: datetime,
    default_hours: int,
    max_hours: int,
    expected_kind: str,
    expected_scope_ids: tuple[str, ...],
    cursor_conflicts: tuple[str, ...] = (),
) -> tuple[datetime, datetime, Cursor]:
    """Return the stable query window and decoded cursor for one recorder page."""
    if (cursor_in := data.get("cursor")) is not None:
        conflicts = tuple(key for key in ("start", "end", "hours", *cursor_conflicts) if key in data)
        if conflicts:
            raise RecoverableToolError(
                "invalid_tool_input",
                {"error": f"cursor cannot be combined with {', '.join(conflicts)}"},
            )
        # Cursor window wins while paging so the continuation is stable.
        cursor = decode_cursor(cursor_in, expected_kind=expected_kind, expected_scope_ids=expected_scope_ids)
        if cursor.start > cursor.end or cursor.end - cursor.start > timedelta(hours=max_hours):
            raise RecoverableToolError(INVALID_CURSOR, {})
        return cursor.start, cursor.end, cursor
    start, end = _clamp_window(
        now,
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        hours=cast(float | None, data.get("hours")),
        default_hours=default_hours,
        max_hours=max_hours,
    )
    return start, end, Cursor(kind=expected_kind, scope_ids=expected_scope_ids, start=start, end=end, cutoffs={})


async def fetch_visible_history_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    sync: bool = True,
) -> dict[str, list[HistoryRow]]:
    """Fetch significant recorder history for visible snapshot entity ids.

    ``sync`` controls the read-after-write recorder synchronization barrier;
    standalone tools keep the default (True), facade fetches pass the run's
    ``live_write_dispatched`` flag so only runs that dispatched live writes pay
    the global loop-drain cost.
    """
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
            sync=sync,
        ),
    )


async def fetch_flat_history_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    sync: bool = True,
) -> list[dict[str, object]]:
    """Fetch JSON-compatible flat history rows for facade helpers and SQL."""
    result = await fetch_visible_history_rows(
        hass,
        snapshot,
        deadline,
        entity_ids,
        start,
        end,
        sync=sync,
    )
    return flat_history_rows(result, snapshot)


async def fetch_visible_logbook_entries(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    sync: bool = True,
) -> list[dict[str, object]]:
    """Fetch copied logbook entries for visible snapshot entity ids.

    The recorder query stays entirely host-side. ``sync`` controls the shared
    read-after-write barrier, matching the history and statistics fetch seams.
    """
    # Availability gates preserve the stable recorder/logbook helper errors.
    if not recorder_available(hass):
        raise RecoverableToolError("recorder_unavailable", {})
    if not logbook_available(hass):
        raise RecoverableToolError("logbook_unavailable", {})
    _validate_visibility(snapshot, entity_ids)
    event_types = async_determine_event_types(hass, entity_ids, None)
    processor = EventProcessor(hass, event_types, entity_ids, None, None, timestamp=False, include_entity_name=True)
    raw_entries = await _run_query(
        hass,
        deadline,
        functools.partial(processor.get_events, start_day=start, end_day=end),
        sync=sync,
    )
    # Copy host-produced mappings before they cross the private runtime seam.
    return [dict(entry) for entry in raw_entries]


async def fetch_flat_statistics_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    statistic_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    sync: bool = True,
) -> list[dict[str, object]]:
    """Fetch JSON-compatible long-term statistics rows for facade SQL."""
    return await _fetch_flat_statistics_rows_for_period(
        hass,
        snapshot,
        deadline,
        statistic_ids,
        start,
        end,
        period="hour",
        sync=sync,
    )


async def fetch_flat_short_term_statistics_rows(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    statistic_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    sync: bool = True,
) -> list[dict[str, object]]:
    """Fetch JSON-compatible 5-minute short-term statistics rows for facade SQL."""
    return await _fetch_flat_statistics_rows_for_period(
        hass,
        snapshot,
        deadline,
        statistic_ids,
        start,
        end,
        period="5minute",
        sync=sync,
    )


async def _fetch_flat_statistics_rows_for_period(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    statistic_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    period: Literal["5minute", "hour"],
    sync: bool,
) -> list[dict[str, object]]:
    """Fetch and flatten recorder statistics rows for one supported SQL period."""
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
            period=period,
            units=None,
            types=set(_ALL_STAT_QUERY_TYPES),
        ),
        sync=sync,
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


def _production_recorder_source(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
) -> RecorderSource:
    """Build a RecorderSource backed by the live recorder and logbook."""
    recorder_synced = False

    async def _fetch_history(entity_ids: list[str], start: datetime, end: datetime) -> dict[str, list[HistoryRow]]:
        nonlocal recorder_synced
        # Reuses the existing visibility-validating helper unchanged so the
        # code.py facade seam and the tool path share one fetch implementation.
        result = await fetch_visible_history_rows(
            hass, snapshot, deadline, entity_ids, start, end, sync=not recorder_synced
        )
        # One source is built per tool invocation; later grouped reads share its barrier.
        recorder_synced = True
        return result

    async def _fetch_statistics(
        statistic_ids: list[str], start: datetime, end: datetime, period: str, types: set[str]
    ) -> Mapping[str, list[dict[str, object]]]:
        nonlocal recorder_synced
        result = cast(
            Mapping[str, list[dict[str, object]]],
            await _run_query(
                hass,
                deadline,
                functools.partial(
                    statistics.statistics_during_period,
                    hass=hass,
                    start_time=start,
                    end_time=end,
                    statistic_ids=set(statistic_ids),
                    period=cast(Literal["5minute", "day", "hour", "week", "month", "year"], period),
                    units=None,
                    types=cast(set[StatisticQueryType], types),
                ),
                sync=not recorder_synced,
            ),
        )
        # One source is built per tool invocation; later grouped reads share its barrier.
        recorder_synced = True
        return result

    async def _fetch_logbook(entity_ids: list[str], start: datetime, end: datetime) -> list[dict[str, object]]:
        nonlocal recorder_synced
        result = await fetch_visible_logbook_entries(
            hass, snapshot, deadline, entity_ids, start, end, sync=not recorder_synced
        )
        # One source is built per tool invocation; later grouped reads share its barrier.
        recorder_synced = True
        return result

    return RecorderSource(
        now=dt_util.utcnow(),
        logbook_available=logbook_available(hass),
        run_in_executor=lambda fn: hass.async_add_executor_job(fn),
        fetch_history=_fetch_history,
        fetch_statistics=_fetch_statistics,
        fetch_logbook=_fetch_logbook,
    )


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
    *,
    returned: int = 0,
    limit: int | None = None,
) -> JsonObjectType:
    """Return a JSON-safe recorder payload with a stable window envelope."""
    payload: dict[str, object] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        **body,
    }
    if next_cursor is not None:
        encoded_cursor = encode_cursor(next_cursor)
        payload["next_cursor"] = encoded_cursor
        payload["overflow"] = overflow_metadata(
            truncated=True,
            limit=limit,
            returned=returned,
            next_cursor=encoded_cursor,
        )
    return cast(JsonObjectType, json_safe(payload))


async def _run_query[T](
    hass: HomeAssistant,
    deadline: float,
    fn: Callable[[], T],
    *,
    sync: bool = True,
) -> T:
    """Run a blocking recorder query on the recorder executor with the tool deadline.

    ``sync`` runs the read-after-write barrier (commit pending recorder writes
    before the read). Standalone recorder tools keep the default True so they
    always observe prior writes; facade fetches pass the run's
    ``live_write_dispatched`` flag so only runs that dispatched live writes pay
    the global loop-drain cost of the barrier.
    """
    recorder_instance = get_instance(hass)
    if sync:
        await _sync_recorder_for_query(hass, recorder_instance, deadline)
    return await _await_deadline(recorder_instance.async_add_executor_job(fn), deadline)


async def _sync_recorder_for_query(
    hass: HomeAssistant,
    recorder_instance: Recorder,
    deadline: float,
) -> None:
    """Commit recorder writes that were dispatched before a recorder-backed query."""
    # Drain the HA event loop first so state-change events are dispatched to the
    # recorder listener and queued before the commit-before barrier is enqueued.
    await _await_deadline(hass.async_block_till_done(), deadline)
    future: asyncio.Future[None] = hass.loop.create_future()
    # The public recorder async_block_till_done() only queues this SynchronizeTask
    # when it observes backlog or pending writes. A recorder thread can pop a
    # state_changed task before marking pending writes, letting the public helper
    # return without forcing the commit needed for immediate history/logbook reads.
    # Queue the commit-before task unconditionally behind already-dispatched work.
    recorder_instance.queue_task(SynchronizeTask(future))
    await _await_deadline(future, deadline)
    # The SynchronizeTask resolves its Future back on the HA loop; drain once so
    # callback-side loop work settles before the query enters the recorder executor.
    await _await_deadline(hass.async_block_till_done(), deadline)


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


async def _aggregate_history(
    source: RecorderSource,
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
        source.now,
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

    raw = await source.fetch_history(entity_ids, start, end)

    def _summarize() -> dict[str, object]:
        aggregator = AGGREGATORS[aggregate]
        return {
            entity_id: aggregator(cast(list[HistoryRow], list(raw.get(entity_id, ()))), start, end, filters)
            for entity_id in entity_ids
        }

    # Keep CPU aggregation off the event loop; recorder row fetch remains deadline-bound inside the fetcher.
    summary = await source.run_in_executor(_summarize)
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
    snapshot: HomeSnapshot,
    source: RecorderSource,
    entity_ids: list[str],
    data: dict[str, object],
) -> JsonObjectType:
    """Return flat declarative analytics results for get_history."""
    if data.get("attributes") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "analytics cannot be combined with attributes"})
    if data.get("cursor") is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "analytics cannot be combined with cursor"})
    start, end = _clamp_window(
        source.now,
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        hours=cast(float | None, data.get("hours")),
        default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
        max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    )
    raw = await source.fetch_history(entity_ids, start, end)
    spec = analytics_spec_from_data(data)

    def _analyze() -> list[dict[str, object]]:
        flat = flat_history_rows(raw, snapshot)
        return run_analytics(cast(list[HistoryRow], flat), spec, (start, end), snapshot)

    # Keep CPU analytics off the event loop; recorder row fetch remains deadline-bound inside the fetcher.
    rows = await source.run_in_executor(_analyze)
    return cast(
        JsonObjectType,
        json_safe({"window": {"start": start.isoformat(), "end": end.isoformat()}, "rows": rows}),
    )
