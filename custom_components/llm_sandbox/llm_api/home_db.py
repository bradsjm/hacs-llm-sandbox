"""Bounded in-memory SQLite database over the frozen home snapshot."""

import json
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from homeassistant.util import dt as dt_util

from ..snapshot.models import HomeSnapshot, SafeState
from .errors import HelperExecutionError
from .numeric import finite_float

MAX_SQL_RESULT_ROWS = 500
MAX_HISTORY_LOAD_ROWS = 20_000
SQL_PROGRESS_OPCODES = 50_000
MAX_SQL_LENGTH = 4_000

_READ_ONLY_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_PRAGMA,
}
_SCHEMA_TABLE_NAMES: tuple[str, ...] = (
    "states",
    "history",
    "statistics",
    "states_meta",
    "statistics_meta",
    "statistics_short_term",
    "state_history",
    "long_term_statistics",
)
_SQL_IDENTIFIER = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'
_TABLE_PATTERN = re.compile(
    rf"\b(?:from|join)\s+(?P<table>{_SQL_IDENTIFIER}(?:\s*\.\s*{_SQL_IDENTIFIER})?)",
    re.IGNORECASE,
)
_FROM_CLAUSE_PATTERN = re.compile(
    r"\bfrom\s+(?P<body>.*?)(?=\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\bunion\b|\bexcept\b|\bintersect\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_TABLE_PATTERN = re.compile(
    rf"^\s*(?P<table>{_SQL_IDENTIFIER}(?:\s*\.\s*{_SQL_IDENTIFIER})?)",
    re.IGNORECASE,
)
_LEADING_SQL_COMMENT = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/)", re.DOTALL)
_SQL_COLUMN_QUALIFIER = re.compile(
    rf"no such column:\s+(?:(?P<table>{_SQL_IDENTIFIER})\s*\.)?{_SQL_IDENTIFIER}",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Bounded SQL query result."""

    rows: list[dict[str, object]]
    truncated: bool


class HomeDatabase:
    """Per-run SQLite database populated only from safe snapshot/recorder rows."""

    def __init__(self, snapshot: HomeSnapshot) -> None:
        """Create an empty per-run in-memory database for one snapshot."""
        self.snapshot = snapshot
        self._conn: sqlite3.Connection | None = None
        self._history_windows: dict[str, tuple[datetime, datetime]] = {}
        self._statistics_windows: dict[str, tuple[datetime, datetime]] = {}

    def initialize(self) -> None:
        """Connect, create schema, and load snapshot states for this per-run database."""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(False)
        self._conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_LENGTH)
        self._create_schema()
        self.load_states()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the initialized SQLite connection."""
        assert self._conn is not None
        return self._conn

    @property
    def loaded_history(self) -> bool:
        """Whether recorder history was loaded for this run."""
        return bool(self._history_windows)

    @property
    def loaded_statistics(self) -> bool:
        """Whether recorder statistics were loaded for this run."""
        return bool(self._statistics_windows)

    def history_needs_load(self, entity_ids: Sequence[str], start: datetime, end: datetime) -> bool:
        """Return whether history rows for the requested scope/window are missing."""
        return any(not _window_covers(self._history_windows.get(entity_id), start, end) for entity_id in entity_ids)

    def statistics_needs_load(self, statistic_ids: Sequence[str], start: datetime, end: datetime) -> bool:
        """Return whether statistic rows for the requested scope/window are missing."""
        return any(
            not _window_covers(self._statistics_windows.get(statistic_id), start, end)
            for statistic_id in statistic_ids
        )

    def load_states(self) -> None:
        """Load visible snapshot states into SQLite before read-only guard activation."""
        if self._conn is None:
            self.initialize()
            return
        rows = [self._state_row(state) for state in self.snapshot.states.values()]
        self.conn.executemany(
            """
            insert into states(entity_id, domain, object_id, name, state, value, attributes,
                               area_id, floor_id, device_id, platform, unique_id,
                               last_changed, last_changed_ts, last_updated, last_updated_ts)
            values(:entity_id, :domain, :object_id, :name, :state, :value, :attributes,
                   :area_id, :floor_id, :device_id, :platform, :unique_id,
                   :last_changed, :last_changed_ts, :last_updated, :last_updated_ts)
            """,
            rows,
        )
        self.conn.commit()

    def load_history(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Load flat recorder history rows into SQLite before executing user SQL."""
        self.conn.executemany(
            """
            insert or ignore into history(entity_id, domain, area_id, floor_id, device_id, when_iso, when_ts, state, value)
            values(:entity_id, :domain, :area_id, :floor_id, :device_id, :when_iso, :when_ts, :state, :value)
            """,
            [self._history_row(row) for row in rows[:MAX_HISTORY_LOAD_ROWS]],
        )
        self.conn.commit()

    def load_statistics(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Load flat recorder statistics rows into SQLite before executing user SQL."""
        self.conn.executemany(
            """
            insert or ignore into statistics(statistic_id, entity_id, when_iso, when_ts, mean, min, max, state, sum)
            values(:statistic_id, :entity_id, :when_iso, :when_ts, :mean, :min, :max, :state, :sum)
            """,
            [self._statistic_row(row) for row in rows[:MAX_HISTORY_LOAD_ROWS]],
        )
        self.conn.commit()

    def record_history_loaded(self, entity_ids: Sequence[str], start: datetime, end: datetime) -> None:
        """Record the trusted recorder history scope now available in the DB."""
        for entity_id in entity_ids:
            self._history_windows[entity_id] = _merge_window(self._history_windows.get(entity_id), start, end)

    def record_statistics_loaded(self, statistic_ids: Sequence[str], start: datetime, end: datetime) -> None:
        """Record the trusted recorder statistics scope now available in the DB."""
        for statistic_id in statistic_ids:
            self._statistics_windows[statistic_id] = _merge_window(
                self._statistics_windows.get(statistic_id), start, end
            )

    def execute(self, sql: str, deadline: float) -> QueryResult:
        """Execute one bounded read-only SQL statement."""
        self._arm_user_sql_guard(deadline)
        try:
            cursor = self.conn.execute(sql)
            fetched = cursor.fetchmany(MAX_SQL_RESULT_ROWS + 1)
        except sqlite3.OperationalError as err:
            raise self._refine_sql_error(str(err)) from err
        except sqlite3.DatabaseError as err:
            raise HelperExecutionError("query", "sql_read_only", {"reason": str(err)}) from err
        finally:
            self._disarm_user_sql_guard()
        if time.monotonic() > deadline:
            raise HelperExecutionError("query", "sql_timeout", {})
        rows = [dict(row) for row in fetched[:MAX_SQL_RESULT_ROWS]]
        return QueryResult(rows=cast(list[dict[str, object]], rows), truncated=len(fetched) > MAX_SQL_RESULT_ROWS)

    def close(self) -> None:
        """Close the per-run in-memory database."""
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            create table states(
                entity_id text primary key, domain text, object_id text, name text, state text, value real,
                attributes text, area_id text, floor_id text, device_id text, platform text, unique_id text,
                last_changed text, last_changed_ts real, last_updated text, last_updated_ts real
            );
            create table history(
                entity_id text, domain text, area_id text, floor_id text, device_id text,
                when_iso text, when_ts real, state text, value real
            );
            -- Dedup on every loaded column so only byte-identical rows collapse on a
            -- re-load; COALESCE normalizes NULLs (SQLite treats NULLs as distinct in
            -- unique indexes) so two identical rows with nullable fields still dedup.
            create unique index history_row_unique on history(
                entity_id, when_iso, state,
                coalesce(when_ts, -1), coalesce(value, -1e308),
                coalesce(area_id, ''), coalesce(floor_id, ''), coalesce(device_id, ''), coalesce(domain, '')
            );
            create table statistics(
                statistic_id text, entity_id text, when_iso text, when_ts real,
                mean real, min real, max real, state real, sum real
            );
            create unique index statistics_row_unique on statistics(statistic_id, when_iso);
            create view state_history as select * from history;
            create view long_term_statistics as select * from statistics;
            -- Name-only aliases for recorder-schema reflexes.
            create view states_meta as select entity_id, state, attributes, last_updated_ts, last_changed_ts from states;
            create view statistics_meta as select distinct statistic_id, entity_id from statistics;
            create view statistics_short_term as select * from statistics;
            """
        )

    def _arm_user_sql_guard(self, deadline: float) -> None:
        """Temporarily guard one user SQL statement while trusted loads remain possible later."""
        self.conn.execute("pragma query_only=ON")
        self.conn.set_authorizer(_authorize)
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, SQL_PROGRESS_OPCODES)

    def _disarm_user_sql_guard(self) -> None:
        """Restore trusted host write access for lazy internal table loading."""
        self.conn.set_progress_handler(None, 0)
        self.conn.set_authorizer(None)
        self.conn.execute("pragma query_only=OFF")

    def _state_row(self, state: SafeState) -> dict[str, object]:
        return {
            "entity_id": state.entity_id,
            "domain": state.domain,
            "object_id": state.object_id,
            "name": state.name,
            "state": state.state,
            "value": finite_float(state.state),
            "attributes": json.dumps(state.attributes, default=str, sort_keys=True),
            "area_id": state.area_id,
            "floor_id": state.floor_id,
            "device_id": state.device_id,
            "platform": state.platform,
            "unique_id": state.unique_id,
            "last_changed": state.last_changed,
            "last_changed_ts": state.last_changed_timestamp,
            "last_updated": state.last_updated,
            "last_updated_ts": state.last_updated_timestamp,
        }

    def _history_row(self, row: Mapping[str, object]) -> dict[str, object]:
        when = str(row.get("when"))
        when_ts = row.get("when_ts")
        return {
            "entity_id": row.get("entity_id"),
            "domain": row.get("domain"),
            "area_id": row.get("area_id"),
            "floor_id": row.get("floor_id"),
            "device_id": row.get("device_id"),
            "when_iso": when,
            "when_ts": when_ts if isinstance(when_ts, float | int) else _timestamp(when),
            "state": row.get("state"),
            "value": finite_float(row.get("value")),
        }

    def _statistic_row(self, row: Mapping[str, object]) -> dict[str, object]:
        when = str(row.get("when"))
        statistic_id = str(row.get("statistic_id") or row.get("entity_id"))
        return {
            "statistic_id": statistic_id,
            "entity_id": statistic_id,
            "when_iso": when,
            "when_ts": _timestamp(when),
            "mean": finite_float(row.get("mean")),
            "min": finite_float(row.get("min")),
            "max": finite_float(row.get("max")),
            "state": finite_float(row.get("state")),
            "sum": finite_float(row.get("sum")),
        }

    def _refine_sql_error(self, message: str) -> HelperExecutionError:
        """Return a structured SQL helper error with targeted fix candidates."""
        lowered = message.lower()
        if "no such table" in lowered:
            return HelperExecutionError(
                "query",
                "sql_unknown_table",
                {"reason": message},
                fix=_bounded_list(_SCHEMA_TABLE_NAMES),
            )
        if "no such column" in lowered:
            table = None
            if match := _SQL_COLUMN_QUALIFIER.search(message):
                table_ref = match.group("table")
                table = _table_name(table_ref) if table_ref else None
            if table not in _SCHEMA_TABLE_NAMES:
                table = "states"
            columns = tuple(str(row["name"]) for row in self.conn.execute(f"pragma table_info({table})"))
            return HelperExecutionError(
                "query",
                "sql_unknown_column",
                {"reason": message},
                fix=_bounded_list(columns),
            )
        if "not authorized" in lowered or "readonly" in lowered or "only select" in lowered:
            return HelperExecutionError("query", "sql_read_only", {"reason": message})
        if "interrupted" in lowered:
            return HelperExecutionError("query", "sql_timeout", {})
        return HelperExecutionError("query", "sql_syntax_error", {"reason": message})


def referenced_tables(sql: str) -> set[str]:
    """Return user-referenced table names from simple SELECT/JOIN clauses."""
    tables: set[str] = set()
    for match in _TABLE_PATTERN.finditer(sql):
        tables.add(_table_name(match.group("table")))
    for match in _FROM_CLAUSE_PATTERN.finditer(sql):
        for segment in match.group("body").split(","):
            if segment_match := _LEADING_TABLE_PATTERN.match(segment):
                tables.add(_table_name(segment_match.group("table")))
    return tables


def _table_name(table_ref: str) -> str:
    """Return the unqualified, unquoted SQLite table identifier."""
    # The table identifier is the final identifier; any earlier identifier is a
    # schema/catalog qualifier such as ``main``.
    table_name = re.split(r"\s*\.\s*", table_ref)[-1]
    return table_name.strip('"`[]').lower()


def _window_covers(loaded: tuple[datetime, datetime] | None, start: datetime, end: datetime) -> bool:
    """Return whether a previously loaded continuous window covers a request."""
    return loaded is not None and loaded[0] <= start and loaded[1] >= end


def _merge_window(
    loaded: tuple[datetime, datetime] | None, start: datetime, end: datetime
) -> tuple[datetime, datetime]:
    """Return the union window to track trusted rows already inserted."""
    if loaded is None:
        return start, end
    return min(loaded[0], start), max(loaded[1], end)


def ensure_sql_allowed(sql: str) -> None:
    """Validate cheap SQL properties before sending it to SQLite."""
    if not sql.strip():
        raise HelperExecutionError("query", "invalid_tool_input", {"reason": "SQL is required"})
    if len(sql) > MAX_SQL_LENGTH:
        raise HelperExecutionError("query", "sql_too_long", {"max_length": str(MAX_SQL_LENGTH)})
    sql_without_comments = sql
    while match := _LEADING_SQL_COMMENT.match(sql_without_comments):
        sql_without_comments = sql_without_comments[match.end() :]
    first = sql_without_comments.lstrip().split(None, 1)[0].lower() if sql_without_comments.strip() else ""
    if first not in {"select", "with", "pragma"}:
        raise HelperExecutionError(
            "query", "sql_read_only", {"reason": "only SELECT/WITH/approved PRAGMA reads are allowed"}
        )


def _authorize(action: int, arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
    # Read-only category denial is the safety boundary with disabled extensions and query_only.
    if action not in _READ_ONLY_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() not in {"table_info", "table_list"}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _bounded_list(names: tuple[str, ...], limit: int = 8) -> list[str]:
    """Return a deterministic bounded candidate list."""
    bounded = sorted(names)[:limit]
    if len(names) > limit:
        bounded.append("...")
    return bounded


def _timestamp(value: str) -> float | None:
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    return dt_util.as_utc(parsed).timestamp()
