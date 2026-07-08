"""State, date/time, and root Home Assistant facades for Monty."""

# ruff: noqa: D105, ANN401

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Any, cast

from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonValueType

from ...const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    MAX_HISTORY_STATES,
    MAX_RECORDER_ENTITY_IDS,
    MAX_RECORDER_LOOKBACK_HOURS,
)
from ...snapshot.models import HomeSnapshot, SafeConfig, SafeState
from ..data.history import HistoryRow, analytics_spec_from_data, run_analytics
from ..data.home_db import MAX_HISTORY_LOAD_ROWS, HomeDatabase, QueryResult, ensure_sql_allowed
from ..data.recorder_scope import _clamp_window, resolve_entity_ids
from ..errors import HelperExecutionError, RecoverableToolError
from ..executor_support import helper_response
from ..guidance import FailureContext, Intent, advise
from ..sandbox_context import require_runtime, require_snapshot
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


def _history_entity_ids(snapshot: HomeSnapshot, entity_ids: str | list[str] | None) -> list[str]:
    """Resolve explicit facade history ids or default to all visible states."""
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
        raise HelperExecutionError("history", "entity_not_visible", {"entity_id": missing[0]}, guidance=guidance)
    if not ids or len(ids) > MAX_RECORDER_ENTITY_IDS:
        raise HelperExecutionError(
            "history",
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


def _query_scope(
    snapshot: HomeSnapshot,
    sql: str,
    entity_ids: str | list[str] | None,
    area_id: str | None,
    floor_id: str | None,
    device_id: str | None,
    label_id: str | None,
    domain: str | None,
) -> tuple[list[str], bool]:
    """Resolve query recorder scope from explicit ids, HA-native selectors, or SQL literals.

    Explicit entity ids and any selector (area/floor/device/label/domain) route through
    the recorder resolver so scope is bounded without an all-visible fallback. With none
    given, fall back to entity-id string literals quoted in the SQL, accepting only ids
    that actually exist in the snapshot (so unrelated quoted strings do not widen scope).

    Returns the resolved ids and a flag that is True when scope was inferred from SQL
    literals (the fallback path), so the caller can surface that to the LLM.
    """
    data: dict[str, object] = {
        "entity_ids": _coerce_id_list(entity_ids),
        "area_id": area_id,
        "floor_id": floor_id,
        "device_id": device_id,
        "label_id": label_id,
        "domain": domain,
    }
    # Branch boundary: an explicit id or selector routes through the recorder resolver,
    # which validates visibility, expands selectors, and caps at MAX_RECORDER_ENTITY_IDS.
    if any(data[key] for key in ("entity_ids", "area_id", "floor_id", "device_id", "label_id", "domain")):
        try:
            return resolve_entity_ids(snapshot, data, "entity_ids"), False
        except RecoverableToolError as err:
            guidance = _query_scope_guidance(snapshot, data, err.placeholders)
            raise HelperExecutionError("query", err.key, err.placeholders, guidance=guidance) from err
    # Advisory literal scan: only accept tokens that are real visible entity ids.
    literal_ids = sorted(set(re.findall(r"['\"]([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)['\"]", sql)) & set(snapshot.states))
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
        aggregate: Mapping[str, object] | str | None = None,
        group_by: str | list[str] | None = None,
        bucket: str | None = None,
        limit: int | None = None,
    ) -> JsonValueType:
        """Return raw or aggregated recorder history for visible snapshot entities."""

        async def _call() -> object:
            runtime = require_runtime(None)
            snapshot = require_snapshot()
            analytics = aggregate is not None or group_by is not None or bucket is not None or limit is not None
            start, end = _clamp_window(
                dt_util.utcnow(),
                None,
                None,
                hours=hours,
                default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS if analytics else MAX_RECORDER_LOOKBACK_HOURS,
            )
            ids = _history_entity_ids(snapshot, entity_ids)
            rows = await runtime.fetch_history(ids, start, end)
            if not analytics:
                if len(rows) > MAX_HISTORY_STATES:
                    runtime.state.notes.append(f"history result capped at {MAX_HISTORY_STATES} rows")
                    rows = rows[:MAX_HISTORY_STATES]
                return [
                    {"entity_id": row["entity_id"], "when": row["when"], "state": row["state"], "value": row["value"]}
                    for row in rows
                ]
            spec = analytics_spec_from_data(
                {
                    "aggregate": aggregate,
                    "group_by": group_by,
                    "bucket": bucket,
                    "limit": limit,
                }
            )
            return run_analytics(cast(list[HistoryRow], rows), spec, (start, end), snapshot)

        return await helper_response(require_runtime(None).state, "history", _call)

    async def query(
        self,
        sql: str,
        hours: float | None = None,
        entity_ids: str | list[str] | None = None,
        area_id: str | None = None,
        floor_id: str | None = None,
        device_id: str | None = None,
        label_id: str | None = None,
        domain: str | None = None,
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
            needs_history = "history" in tables
            needs_statistics = "statistics" in tables
            if needs_history or needs_statistics:
                # Branch boundary: when history is referenced it forces the 24h recorder cap;
                # statistics-only may use the longer analytics lookback. One window, one scope.
                max_hours = MAX_RECORDER_LOOKBACK_HOURS if needs_history else MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS
                start, end = _clamp_window(
                    dt_util.utcnow(),
                    None,
                    None,
                    hours=hours,
                    default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                    max_hours=max_hours,
                )
                ids, inferred = _query_scope(snapshot, sql, entity_ids, area_id, floor_id, device_id, label_id, domain)
                if inferred:
                    runtime.state.notes.append(f"query scope inferred from SQL literals: {', '.join(ids)}")

                async def _load_history() -> None:
                    if not needs_history or not db.history_needs_load(ids, start, end):
                        return
                    rows = await runtime.fetch_history(ids, start, end)
                    truncated = cast(bool, await runtime.run_blocking(lambda: db.load_history(rows)))
                    if truncated:
                        runtime.state.notes.append(f"history load capped at {MAX_HISTORY_LOAD_ROWS} rows")
                        # A capped load is intentionally NOT marked complete: the cap keeps
                        # only the newest rows for the fetched scope, so a later narrower
                        # query for an entity whose rows fell outside the cap must still be
                        # allowed to re-fetch. Re-inserts stay cheap via the full-row dedup index.
                        return
                    db.record_history_loaded(ids, start, end)

                async def _load_statistics() -> None:
                    if not needs_statistics or not db.statistics_needs_load(ids, start, end):
                        return
                    rows = await runtime.fetch_statistics(ids, start, end)
                    truncated = cast(bool, await runtime.run_blocking(lambda: db.load_statistics(rows)))
                    if truncated:
                        runtime.state.notes.append(f"statistics load capped at {MAX_HISTORY_LOAD_ROWS} rows")
                        return
                    db.record_statistics_loaded(ids, start, end)

                await _load_history()
                await _load_statistics()
            result = cast(QueryResult, await runtime.run_blocking(lambda: db.execute(sql, runtime.deadline)))
            if result.truncated:
                runtime.state.notes.append("query result truncated")
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


def _query_scope_guidance(
    snapshot: HomeSnapshot,
    data: Mapping[str, object],
    placeholders: Mapping[str, str],
) -> Mapping[str, object] | None:
    """Return QUERY_HISTORY guidance for facade query selector failures."""
    requested = str(placeholders.get("entity_id") or placeholders.get("area_id") or placeholders.get("selector") or "")
    domain = str(data.get("domain") or "")
    entity_ids = data.get("entity_ids")
    if not requested and isinstance(entity_ids, list) and entity_ids:
        requested = str(entity_ids[0])
    if not requested:
        return None
    if not domain and "." in requested:
        domain = requested.split(".", 1)[0]
    return advise(
        snapshot,
        FailureContext(intent=Intent.QUERY_HISTORY, requested=requested, domain=domain),
    ).to_payload()
