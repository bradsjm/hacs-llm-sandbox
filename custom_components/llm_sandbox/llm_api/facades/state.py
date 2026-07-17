"""State, date/time, and root Home Assistant facades for Monty."""

# ruff: noqa: D105, ANN401

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Any, cast

from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonValueType

from ...const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    DEFAULT_LOGBOOK_WINDOW_HOURS,
    MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_RECORDER_ENTITY_IDS,
    MAX_RECORDER_LOOKBACK_HOURS,
)
from ...snapshot.models import HomeSnapshot, SafeConfig, SafeState
from ..data.history import (
    AnalyticsSpec,
    HistoryRow,
    analytics_spec_from_data,
    history_analytics_requested,
    run_analytics,
)
from ..data.home_db import MAX_HISTORY_LOAD_ROWS, HomeDatabase, QueryResult, ensure_sql_allowed
from ..data.recorder_scope import _clamp_window, resolve_entity_ids
from ..errors import HelperExecutionError, RecoverableToolError
from ..executor_support import helper_response, overflow_metadata
from ..guidance import FailureContext, Intent, advise
from ..sandbox_context import require_runtime, require_snapshot
from ..tools.energy import validate_energy_args
from .services import SafeServiceRegistry


@dataclass(frozen=True, slots=True)
class SafeStateMachine:
    """Read-only Home Assistant StateMachine facade.

    Mirrors HA's ``hass.states`` read API. All methods are synchronous
    callbacks (the ``async_`` prefix denotes loop-safe, not coroutine).
    Optional subscript sugar (``states["light.x"]``, ``"light.x" in states``)
    is provided in addition to the strict ``get``/``async_all`` methods.
    """

    states: Mapping[str, SafeState]
    type: str = "states"

    def get(self, entity_id: str) -> SafeState | None:
        """Return the state for ``entity_id``, or None if it does not exist."""
        return self.states.get(entity_id)

    def async_all(self, domain_filter: str | None = None) -> list[SafeState]:
        """Return all states, optionally filtered by domain."""
        if domain_filter is None:
            return list(self.states.values())
        return [s for s in self.states.values() if s.domain == domain_filter]

    def is_state(self, entity_id: str, state: str) -> bool:
        """Return True if ``entity_id`` exists and its state equals ``state``."""
        st = self.states.get(entity_id)
        return st is not None and st.state == state

    def async_entity_ids(self, domain_filter: str | None = None) -> list[str]:
        """Return all entity IDs, optionally filtered by domain."""
        if domain_filter is None:
            return list(self.states.keys())
        return [eid for eid, st in self.states.items() if st.domain == domain_filter]

    def entity_ids(self, domain_filter: str | None = None) -> list[str]:
        """Sync alias for async_entity_ids (HA parity)."""
        return self.async_entity_ids(domain_filter)

    # --- Optional subscript/containment sugar (additive; strict API still works) ---

    def __getitem__(self, entity_id: str) -> SafeState:
        st = self.states.get(entity_id)
        if st is None:
            raise KeyError(entity_id)
        return st

    def __contains__(self, entity_id: object) -> bool:
        return isinstance(entity_id, str) and entity_id in self.states

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Any:
        return iter(self.states.values())

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, {"type": self.type, "entity_count": len(self.states)})


def _recorder_entity_ids(
    snapshot: HomeSnapshot,
    entity_ids: str | list[str] | None,
    helper: str,
) -> list[str]:
    """Resolve explicit facade recorder ids or default to all visible states."""
    if entity_ids is None:
        ids = sorted(snapshot.states)
    elif isinstance(entity_ids, str):
        ids = [entity_ids]
    else:
        ids = [str(entity_id) for entity_id in entity_ids]
    missing = [entity_id for entity_id in ids if entity_id not in snapshot.states]
    if missing:
        domain = missing[0].split(".", 1)[0] if "." in missing[0] else ""
        guidance = advise(
            snapshot,
            FailureContext(intent=Intent.QUERY_HISTORY, requested=missing[0], domain=domain),
        ).to_payload()
        raise HelperExecutionError(helper, "entity_not_visible", {"entity_id": missing[0]}, guidance=guidance)
    if not ids or len(ids) > MAX_RECORDER_ENTITY_IDS:
        raise HelperExecutionError(
            helper,
            "invalid_tool_input",
            {"reason": f"scope must resolve to 1..{MAX_RECORDER_ENTITY_IDS} visible entities"},
        )
    return ids


def _coerce_id_list(value: str | list[str] | None) -> list[str] | None:
    """Normalize optional query entity ids into recorder resolver input."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _coerce_history_datetime(
    value: str | _datetime | SafeDateTime | None,
    field: str,
) -> _datetime | None:
    """Normalize one facade history boundary to a UTC datetime."""
    if value is None:
        return None
    if isinstance(value, _datetime):
        return dt_util.as_utc(value)
    source = value.iso if isinstance(value, SafeDateTime) else value
    if isinstance(source, str) and (parsed := dt_util.parse_datetime(source)) is not None:
        return dt_util.as_utc(parsed)
    raise RecoverableToolError(
        "invalid_tool_input",
        {"error": f"{field} must be an ISO datetime"},
    )


def _resolve_facade_recorder_scope(
    snapshot: HomeSnapshot,
    *,
    helper: str,
    entity_ids: str | list[str] | None,
    area_id: str | list[str] | None,
    floor_id: str | list[str] | None,
    device_id: str | list[str] | None,
    label_id: str | list[str] | None,
    domain: str | list[str] | None,
) -> list[str]:
    """Resolve explicit facade recorder scope against the fresh snapshot."""
    data: dict[str, object] = {
        "entity_ids": _coerce_id_list(entity_ids),
        "area_id": _coerce_id_list(area_id),
        "floor_id": _coerce_id_list(floor_id),
        "device_id": _coerce_id_list(device_id),
        "label_id": _coerce_id_list(label_id),
        "domain": _coerce_id_list(domain),
    }
    # An explicitly supplied empty scope must not widen to the resolver's
    # all-visible default.
    if not any(data.values()):
        raise HelperExecutionError(
            helper,
            "invalid_tool_input",
            {"reason": f"scope must resolve to 1..{MAX_RECORDER_ENTITY_IDS} visible entities"},
        )
    try:
        return resolve_entity_ids(snapshot, data, "entity_ids")
    except RecoverableToolError as err:
        guidance = _recorder_scope_guidance(snapshot, data, err.placeholders)
        raise HelperExecutionError(
            helper,
            err.key,
            err.placeholders,
            guidance=guidance,
        ) from err


def _logbook_entry_when(entry: Mapping[str, object]) -> str:
    """Return a comparable ISO timestamp for one copied logbook entry."""
    when = entry["when"]
    if isinstance(when, _datetime):
        return dt_util.as_utc(when).isoformat()
    return str(when)


def _query_scope(
    snapshot: HomeSnapshot,
    sql: str,
    entity_ids: str | list[str] | None,
    area_id: str | list[str] | None,
    floor_id: str | list[str] | None,
    device_id: str | list[str] | None,
    label_id: str | list[str] | None,
    domain: str | list[str] | None,
) -> tuple[list[str], bool]:
    """Resolve query recorder scope from explicit ids, HA-native selectors, or SQL literals.

    Explicit entity ids and any selector (area/floor/device/label/domain) route through
    the recorder resolver so scope is bounded without an all-visible fallback. With none
    given, fall back to entity-id string literals quoted in the SQL, accepting only ids
    that actually exist in the snapshot (so unrelated quoted strings do not widen scope).

    Returns the resolved ids and a flag that is True when scope was inferred from SQL
    literals (the fallback path), so the caller can surface that to the LLM.
    """
    scope_values = (entity_ids, area_id, floor_id, device_id, label_id, domain)
    # Branch boundary: any supplied id or selector routes through the recorder
    # resolver, including explicit empty lists that must fail closed.
    if any(value is not None for value in scope_values):
        return (
            _resolve_facade_recorder_scope(
                snapshot,
                helper="query",
                entity_ids=entity_ids,
                area_id=area_id,
                floor_id=floor_id,
                device_id=device_id,
                label_id=label_id,
                domain=domain,
            ),
            False,
        )
    # Advisory literal scan: only accept single-quoted values outside comments
    # that are real visible entity ids. Quoted identifiers and comments do not
    # describe recorder scope.
    literal_ids = sorted(set(_single_quoted_sql_literals(sql)) & set(snapshot.states))
    if literal_ids:
        if len(literal_ids) > MAX_RECORDER_ENTITY_IDS:
            raise HelperExecutionError(
                "query",
                "invalid_tool_input",
                {
                    "reason": f"query SQL references {len(literal_ids)} entities; narrow it to at most {MAX_RECORDER_ENTITY_IDS}"
                },
            )
        return literal_ids, True
    raise HelperExecutionError(
        "query",
        "invalid_tool_input",
        {
            "reason": "narrow the query with entity_ids/area_id/floor_id/device_id/label_id/domain, or quote entity ids in SQL"
        },
    )


async def _load_query_table(
    runtime: Any,
    table: str,
    ids: Sequence[str],
    start: _datetime,
    end: _datetime,
    needs_load: Callable[[Sequence[str], _datetime, _datetime], bool],
    fetch_rows: Callable[[Sequence[str], _datetime, _datetime], Awaitable[list[dict[str, object]]]],
    load_rows: Callable[[Sequence[Mapping[str, object]]], bool],
    record_loaded: Callable[[Sequence[str], _datetime, _datetime], None],
) -> None:
    """Load one recorder table only when needed, retaining conservative cap semantics."""
    if not needs_load(ids, start, end):
        return
    rows = await fetch_rows(ids, start, end)
    truncated = cast(bool, await runtime.run_blocking(lambda: load_rows(rows)))
    if truncated:
        runtime.state.notes.append(f"{table} load capped at {MAX_HISTORY_LOAD_ROWS} rows")
        # Branch boundary: capped data is not a complete window and must be refetched later.
        return
    # State mutation boundary: only an uncapped fetch can satisfy the requested window.
    record_loaded(ids, start, end)


@dataclass(frozen=True, slots=True)
class SafeHass:
    """Root Home Assistant facade exposed to Monty.

    Exposes only frozen ``states``, ``services``, and ``config`` snapshots. The
    live ``hass`` object's ``bus``, ``config_entries``, ``auth``, ``loop``, ``helpers``,
    and ``data`` are intentionally absent — they are never reachable from
    the sandbox.
    """

    states: SafeStateMachine
    services: SafeServiceRegistry
    config: SafeConfig
    type: str = "hass"

    async def history(
        self,
        entity_ids: str | list[str] | None = None,
        hours: float | None = None,
        aggregate: str | None = None,
        value_operations: str | list[str] | None = None,
        group_by: str | list[str] | None = None,
        bucket: str | None = None,
        limit: int | None = None,
        start: str | _datetime | SafeDateTime | None = None,
        end: str | _datetime | SafeDateTime | None = None,
        area_id: str | list[str] | None = None,
        device_id: str | list[str] | None = None,
        floor_id: str | list[str] | None = None,
        label_id: str | list[str] | None = None,
        domain: str | list[str] | None = None,
        where: dict[str, object] | list[dict[str, object]] | None = None,
        order_by: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
    ) -> list[dict[str, object]]:
        """Return raw or aggregated recorder history for visible snapshot entities."""

        async def _call() -> object:
            runtime = require_runtime(None)
            snapshot = require_snapshot()
            analytics_data: dict[str, object] = {
                "aggregate": aggregate,
                "value_operations": value_operations,
                "group_by": group_by,
                "bucket": bucket,
                "where": where,
                "order_by": order_by,
                "limit": limit,
                "from_state": from_state,
                "to_state": to_state,
            }
            analytics = history_analytics_requested(analytics_data)
            start_value = _coerce_history_datetime(start, "start")
            end_value = _coerce_history_datetime(end, "end")
            window_start, window_end = _clamp_window(
                runtime._utcnow(),
                start_value,
                end_value,
                hours=hours,
                default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS if analytics else MAX_RECORDER_LOOKBACK_HOURS,
            )
            scope_values = (entity_ids, area_id, floor_id, device_id, label_id, domain)
            if any(value is not None for value in scope_values):
                ids = _resolve_facade_recorder_scope(
                    snapshot,
                    helper="history",
                    entity_ids=entity_ids,
                    area_id=area_id,
                    floor_id=floor_id,
                    device_id=device_id,
                    label_id=label_id,
                    domain=domain,
                )
            else:
                ids = _recorder_entity_ids(snapshot, None, "history")
            spec: AnalyticsSpec | None = analytics_spec_from_data(analytics_data) if analytics else None
            rows = await runtime.fetch_history(ids, window_start, window_end)
            if spec is None:
                if len(rows) > MAX_HISTORY_STATES:
                    runtime.state.notes.append(f"history result capped at {MAX_HISTORY_STATES} rows")
                    capped_rows = rows[:MAX_HISTORY_STATES]
                    runtime.state.overflow["history"] = overflow_metadata(
                        truncated=True,
                        limit=MAX_HISTORY_STATES,
                        returned=len(capped_rows),
                        omitted=len(rows) - len(capped_rows),
                    )
                    return [
                        {
                            "entity_id": row["entity_id"],
                            "when": row["when"],
                            "state": row["state"],
                            "value": row["value"],
                        }
                        for row in capped_rows
                    ]
                return [
                    {
                        "entity_id": row["entity_id"],
                        "when": row["when"],
                        "state": row["state"],
                        "value": row["value"],
                    }
                    for row in rows
                ]
            return run_analytics(
                cast(list[HistoryRow], rows),
                spec,
                (window_start, window_end),
                snapshot,
            )

        return cast(
            list[dict[str, object]],
            await helper_response(require_runtime(None).state, "history", _call),
        )

    async def logbook(
        self,
        entity_ids: str | list[str] | None = None,
        hours: float | None = None,
    ) -> JsonValueType:
        """Return bounded logbook activity for visible snapshot entities."""

        async def _call() -> object:
            runtime = require_runtime(None)
            snapshot = require_snapshot()
            start, end = _clamp_window(
                runtime._utcnow(),
                None,
                None,
                hours=hours,
                default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
                max_hours=MAX_RECORDER_LOOKBACK_HOURS,
            )
            ids = _recorder_entity_ids(snapshot, entity_ids, "logbook")
            entries = await runtime.fetch_logbook(ids, start, end)
            # Logbook queries may return source order; expose chronological activity.
            ordered_entries = sorted(entries, key=_logbook_entry_when)
            if len(ordered_entries) > MAX_LOGBOOK_ENTRIES:
                # Keep the newest bounded slice while preserving ascending order.
                capped_entries = ordered_entries[-MAX_LOGBOOK_ENTRIES:]
                runtime.state.notes.append(f"logbook result capped at {MAX_LOGBOOK_ENTRIES} entries")
                return {
                    "entries": capped_entries,
                    "overflow": overflow_metadata(
                        truncated=True,
                        limit=MAX_LOGBOOK_ENTRIES,
                        returned=len(capped_entries),
                        omitted=len(ordered_entries) - len(capped_entries),
                    ),
                }
            return ordered_entries

        return await helper_response(require_runtime(None).state, "logbook", _call)

    async def energy(
        self,
        hours: float | None = None,
        period: str = "auto",
        source_types: str | list[str] | None = None,
        device_statistic_ids: str | list[str] | None = None,
        include: str | list[str] | None = None,
        compare: str | None = None,
        start: str | _datetime | SafeDateTime | None = None,
        end: str | _datetime | SafeDateTime | None = None,
    ) -> JsonValueType:
        """Return Energy dashboard data through the shared validated core."""

        async def _call() -> object:
            runtime = require_runtime(None)
            start_value = _coerce_history_datetime(start, "start")
            end_value = _coerce_history_datetime(end, "end")
            args: dict[str, object] = {
                "hours": hours,
                "period": period,
                "source_types": source_types,
                "device_statistic_ids": device_statistic_ids,
                "include": include,
                "compare": compare,
                "start": start_value.isoformat() if start_value is not None else None,
                "end": end_value.isoformat() if end_value is not None else None,
            }
            return await runtime.fetch_energy(validate_energy_args(args))

        return await helper_response(require_runtime(None).state, "energy", _call)

    async def query(
        self,
        sql: str,
        hours: float | None = None,
        entity_ids: str | list[str] | None = None,
        area_id: str | list[str] | None = None,
        floor_id: str | list[str] | None = None,
        device_id: str | list[str] | None = None,
        label_id: str | list[str] | None = None,
        domain: str | list[str] | None = None,
    ) -> JsonValueType:
        """Run bounded read-only SQLite over states plus optional recorder rows."""

        async def _call() -> object:
            runtime = require_runtime(None)
            snapshot = require_snapshot()
            ensure_sql_allowed(sql)
            if runtime.state.home_db is None:
                # Lazy per-run DB creation stores only frozen snapshot records;
                # lifecycle cleanup in executor closes it before context reset.
                db = HomeDatabase(snapshot)
                await runtime.run_blocking(db.initialize)
                runtime.state.home_db = db
            db = runtime.state.home_db
            # Exact table detection via SQLite's own preparation: view aliases
            # resolve to base tables, and a CTE named ``history`` shadows it.
            tables = cast(set[str], await runtime.run_blocking(lambda: db.referenced_base_tables(sql)))
            referenced_tables = cast(
                frozenset[str], await runtime.run_blocking(lambda: db.referenced_schema_tables(sql))
            )
            needs_history = "history" in tables
            needs_statistics = "statistics" in tables
            needs_short_term_statistics = "statistics_short_term" in tables
            if needs_history or needs_statistics or needs_short_term_statistics:
                # Branch boundary: when history is referenced it forces the 24h recorder cap;
                # statistics-only may use the longer analytics lookback. One window, one scope.
                max_hours = MAX_RECORDER_LOOKBACK_HOURS if needs_history else MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS
                start, end = _clamp_window(
                    runtime._utcnow(),
                    None,
                    None,
                    hours=hours,
                    default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                    max_hours=max_hours,
                )
                ids, inferred = _query_scope(snapshot, sql, entity_ids, area_id, floor_id, device_id, label_id, domain)
                if inferred:
                    runtime.state.notes.append(f"query scope inferred from SQL literals: {', '.join(ids)}")

                await _load_query_table(
                    runtime,
                    "history",
                    ids,
                    start,
                    end,
                    lambda query_ids, query_start, query_end: (
                        needs_history and db.history_needs_load(query_ids, query_start, query_end)
                    ),
                    runtime.fetch_history,
                    db.load_history,
                    db.record_history_loaded,
                )
                await _load_query_table(
                    runtime,
                    "statistics",
                    ids,
                    start,
                    end,
                    lambda query_ids, query_start, query_end: (
                        needs_statistics and db.statistics_needs_load(query_ids, query_start, query_end)
                    ),
                    runtime.fetch_statistics,
                    db.load_statistics,
                    db.record_statistics_loaded,
                )
                await _load_query_table(
                    runtime,
                    "statistics_short_term",
                    ids,
                    start,
                    end,
                    lambda query_ids, query_start, query_end: (
                        needs_short_term_statistics
                        and db.short_term_statistics_needs_load(query_ids, query_start, query_end)
                    ),
                    runtime.fetch_short_term_statistics,
                    db.load_short_term_statistics,
                    db.record_short_term_statistics_loaded,
                )
            result = cast(
                QueryResult,
                await runtime.run_blocking(
                    lambda: db.execute(sql, runtime.deadline, referenced_tables=referenced_tables)
                ),
            )
            if result.truncated:
                runtime.state.notes.append("query result truncated")
                return {
                    "rows": result.rows,
                    "overflow": overflow_metadata(
                        truncated=True,
                        limit=None,
                        returned=len(result.rows),
                    ),
                }
            return result.rows

        return await helper_response(require_runtime(None).state, "query", _call)

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {"type": self.type, "states": self.states, "services": self.services, "config": self.config},
        )


@dataclass(frozen=True, slots=True)
class SafeDate:
    """Frozen date value returned by the ``date`` facade.

    Stores parsed calendar components from a single datetime. All fields are
    JSON-safe primitives.
    """

    iso: str
    year: int
    month: int
    day: int
    weekday: int

    def isoformat(self) -> str:
        """Return the date as an ISO 8601 string (YYYY-MM-DD)."""
        return self.iso

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, self.iso)


@dataclass(frozen=True, slots=True)
class SafeDateTime:
    """Frozen datetime value returned by the ``datetime`` facade.

    Stores parsed datetime components from a single datetime. All fields are
    JSON-safe primitives.
    """

    iso: str
    timestamp: float
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    weekday: int

    def date(self) -> SafeDate:
        """Return the calendar-date portion as a SafeDate."""
        return SafeDate(
            iso=self.iso[:10],
            year=self.year,
            month=self.month,
            day=self.day,
            weekday=self.weekday,
        )

    def isoformat(self) -> str:
        """Return the datetime as an ISO 8601 string."""
        return self.iso

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, self.iso)


@dataclass(frozen=True, slots=True)
class SafeDateFacade:
    """Frozen date class facade exposed as the ``date`` Monty global.

    ``today()`` returns the frozen snapshot date. ``fromisoformat()`` parses
    a caller-supplied ISO date string. No live wall-clock access.
    """

    today_value: SafeDate

    def today(self) -> SafeDate:
        """Return the frozen snapshot date in the configured HA timezone."""
        return self.today_value

    def fromisoformat(self, date_string: str) -> SafeDate:
        """Parse an ISO 8601 date string into a SafeDate.

        Mirrors stdlib date.fromisoformat: a datetime string (containing a time
        component) is rejected rather than silently truncated.
        """
        parsed = _date.fromisoformat(date_string)
        return SafeDate(
            iso=parsed.isoformat(),
            year=parsed.year,
            month=parsed.month,
            day=parsed.day,
            weekday=parsed.weekday(),
        )


@dataclass(frozen=True, slots=True)
class SafeDateTimeFacade:
    """Frozen datetime class facade exposed as the ``datetime`` Monty global.

    ``now()`` returns the frozen snapshot datetime in the HA timezone.
    ``utcnow()`` returns the UTC snapshot datetime. ``fromisoformat()`` parses
    a caller-supplied ISO datetime string. No live wall-clock access.
    """

    now_value: SafeDateTime
    utcnow_value: SafeDateTime

    def now(self, tz: object = None) -> SafeDateTime:
        """Return the frozen snapshot datetime in the configured HA timezone."""
        del tz  # API parity; frozen time cannot honor a caller-supplied timezone.
        return self.now_value

    def utcnow(self) -> SafeDateTime:
        """Return the UTC snapshot datetime."""
        return self.utcnow_value

    def fromisoformat(self, date_string: str) -> SafeDateTime:
        """Parse an ISO 8601 datetime string into a SafeDateTime."""
        return _datetime_from_dt(_datetime.fromisoformat(date_string))


def _date_from_datetime(dt: _datetime) -> SafeDate:
    """Build a SafeDate from a parsed datetime, preserving the calendar date."""
    return SafeDate(
        iso=dt.strftime("%Y-%m-%d"),
        year=dt.year,
        month=dt.month,
        day=dt.day,
        weekday=dt.weekday(),
    )


def _datetime_from_dt(dt: _datetime) -> SafeDateTime:
    """Build a SafeDateTime from a parsed datetime, preserving all components."""
    return SafeDateTime(
        iso=dt.isoformat(),
        timestamp=dt.timestamp(),
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        microsecond=dt.microsecond,
        weekday=dt.weekday(),
    )


def _recorder_scope_guidance(
    snapshot: HomeSnapshot,
    data: Mapping[str, object],
    placeholders: Mapping[str, str],
) -> Mapping[str, object] | None:
    """Return QUERY_HISTORY guidance for facade recorder selector failures."""
    requested = str(placeholders.get("entity_id") or placeholders.get("selector_id") or "")
    entity_ids = data.get("entity_ids")
    if not requested and isinstance(entity_ids, list) and entity_ids:
        requested = str(entity_ids[0])
    if not requested:
        return None
    domain_value = data.get("domain")
    if isinstance(domain_value, str):
        domain = domain_value
    elif isinstance(domain_value, list):
        domain = next((item for item in domain_value if isinstance(item, str)), "")
    else:
        domain = ""
    if not domain and "." in requested:
        domain = requested.split(".", 1)[0]
    return advise(
        snapshot,
        FailureContext(
            intent=Intent.QUERY_HISTORY,
            requested=requested,
            domain=domain,
        ),
    ).to_payload()


def _single_quoted_sql_literals(sql: str) -> list[str]:
    """Return single-quoted SQL string literals outside comments."""
    literals: list[str] = []
    index = 0
    while index < len(sql):
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline == -1 else newline + 1
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            index = len(sql) if end == -1 else end + 2
            continue
        if sql[index] != "'":
            index += 1
            continue
        index += 1
        value: list[str] = []
        while index < len(sql):
            if sql[index] == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                value.append("'")
                index += 2
                continue
            if sql[index] == "'":
                index += 1
                break
            value.append(sql[index])
            index += 1
        literals.append("".join(value))
    return literals
