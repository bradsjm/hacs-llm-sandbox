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

from ...snapshot.models import HomeSnapshot, SafeState
from ..errors import HelperExecutionError
from .numeric import finite_float

MAX_SQL_RESULT_ROWS = 500
MAX_HISTORY_LOAD_ROWS = 20_000
SQL_PROGRESS_OPCODES = 50_000
MAX_SQL_LENGTH = 4_000
# Bounds the size of a single SQLite value (string/blob) so LLM-generated
# queries that synthesize large blobs (zeroblob/randomblob) hit this limit
# instead of allocating unbounded host memory before the progress handler
# can interrupt.
MAX_SQL_VALUE_LENGTH = 1_000_000

_READ_ONLY_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_PRAGMA,
}

# Single source of truth for the in-memory SQL surface. Base tables carry their
# column vocabulary with SQLite types so the DDL and the guidance candidates can
# never drift. Views declare a base table plus either an explicit column subset
# or None (select *); the third flag records DISTINCT views.
SCHEMA_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "states": (
        ("entity_id", "text primary key"),
        ("domain", "text"),
        ("object_id", "text"),
        ("name", "text"),
        ("state", "text"),
        ("value", "real"),
        ("attributes", "text"),
        ("area_id", "text"),
        ("floor_id", "text"),
        ("device_id", "text"),
        ("platform", "text"),
        ("unique_id", "text"),
        ("last_changed", "text"),
        ("last_changed_ts", "real"),
        ("last_updated", "text"),
        ("last_updated_ts", "real"),
    ),
    "history": (
        ("entity_id", "text"),
        ("domain", "text"),
        ("area_id", "text"),
        ("floor_id", "text"),
        ("device_id", "text"),
        ("when_iso", "text"),
        ("when_ts", "real"),
        ("state", "text"),
        ("value", "real"),
    ),
    "statistics": (
        ("statistic_id", "text"),
        ("entity_id", "text"),
        ("when_iso", "text"),
        ("when_ts", "real"),
        ("mean", "real"),
        ("min", "real"),
        ("max", "real"),
        ("state", "real"),
        ("sum", "real"),
    ),
}
# View name -> (base table, explicit columns or None for select *, distinct).
SCHEMA_VIEWS: dict[str, tuple[str, tuple[str, ...] | None, bool]] = {
    "state_history": ("history", None, False),
    "long_term_statistics": ("statistics", None, False),
    "states_meta": ("states", ("entity_id", "state", "attributes", "last_updated_ts", "last_changed_ts"), False),
    "statistics_meta": ("statistics", ("statistic_id", "entity_id"), True),
    "statistics_short_term": ("statistics", None, False),
}
_SCHEMA_TABLE_NAMES: tuple[str, ...] = (*SCHEMA_TABLES, *SCHEMA_VIEWS)


def columns_for_table(name: str) -> tuple[str, ...]:
    """Return the column vocabulary for a known table or view (empty when unknown).

    Single-sourced from ``SCHEMA_TABLES``/``SCHEMA_VIEWS`` so the DDL and the
    guidance candidates cannot diverge. An unknown name returns an empty tuple
    rather than masking the error with another table's columns.
    """
    table = SCHEMA_TABLES.get(name)
    if table is not None:
        return tuple(column for column, _type in table)
    view = SCHEMA_VIEWS.get(name)
    if view is not None:
        base, explicit, _distinct = view
        # select * mirrors the base table columns; explicit selects carry their subset.
        return explicit if explicit is not None else columns_for_table(base)
    return ()


def render_query_schema_prompt(*, compact: bool = False, include_heading: bool = True) -> str:
    """Render LLM-facing SQL guidance from the actual in-memory schema."""
    table_parts = [f"{name}({', '.join(columns_for_table(name))})" for name in SCHEMA_TABLES]
    view_parts = [f"{name}({', '.join(columns_for_table(name))})" for name in SCHEMA_VIEWS]
    heading = "## SQL query\n" if include_heading else ""
    if compact:
        return (
            f"{heading}"
            "- await hass.query(sql, hours=N) runs read-only SQLite over one per-run in-memory database, "
            f"not HA's live recorder. Tables: {'; '.join(table_parts)}. Views: {'; '.join(view_parts)}. "
            "states.attributes is JSON text; use json_extract(attributes, '$.<key>'). History/statistics load "
            "on demand when referenced. No registry tables: use facades or denormalized area_id/floor_id/"
            "device_id/domain columns for location filtering."
        )
    return (
        f"{heading}"
        "- await hass.query(sql, hours=N) runs read-only SQLite against a fresh per-run in-memory database "
        "populated only from the frozen visible snapshot and bounded recorder loads; it is not Home Assistant's "
        "live recorder database.\n"
        f"- Base tables: {'; '.join(table_parts)}.\n"
        f"- Compatibility views: {'; '.join(view_parts)}.\n"
        "- states.attributes is JSON text; use SQLite JSON functions such as "
        "json_extract(attributes, '$.<key>') for attribute filters.\n"
        "- history and statistics rows load on demand only when the SQL references history/state_history or "
        "statistics/long_term_statistics/statistics_meta/statistics_short_term; hours=N bounds that load.\n"
        "- There are no registry tables. Use registry facades before querying, or use the denormalized "
        "states/history columns area_id, floor_id, device_id, and domain for location/entity filtering."
    )


_LEADING_SQL_COMMENT = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/)", re.DOTALL)
_SQL_COLUMN_QUALIFIER = re.compile(
    r"no such column:\s+"
    r"(?:(?P<table>\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)\s*\.)?"
    r"(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)",
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
        # check_same_thread=False is safe only because every access to this
        # connection is serialized through sequential run_blocking awaits within
        # a single execute_home_code run; there is no concurrent access.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(False)
        self._conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_LENGTH)
        self._conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_SQL_VALUE_LENGTH)
        self._create_schema()
        self._load_states()

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

    def load_history(self, rows: Sequence[Mapping[str, object]]) -> bool:
        """Load flat recorder history rows, keeping the newest rows when capped.

        Returns True when the row set exceeded the load cap and was truncated.
        """
        kept, truncated = _cap_newest_first([self._history_row(row) for row in rows])
        self.conn.executemany(
            """
            insert or ignore into history(entity_id, domain, area_id, floor_id, device_id, when_iso, when_ts, state, value)
            values(:entity_id, :domain, :area_id, :floor_id, :device_id, :when_iso, :when_ts, :state, :value)
            """,
            kept,
        )
        self.conn.commit()
        return truncated

    def load_statistics(self, rows: Sequence[Mapping[str, object]]) -> bool:
        """Load flat recorder statistics rows, keeping the newest rows when capped.

        Returns True when the row set exceeded the load cap and was truncated.
        """
        kept, truncated = _cap_newest_first([self._statistic_row(row) for row in rows])
        self.conn.executemany(
            """
            insert or ignore into statistics(statistic_id, entity_id, when_iso, when_ts, mean, min, max, state, sum)
            values(:statistic_id, :entity_id, :when_iso, :when_ts, :mean, :min, :max, :state, :sum)
            """,
            kept,
        )
        self.conn.commit()
        return truncated

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

    def referenced_base_tables(self, sql: str) -> set[str]:
        """Return the base tables (history/statistics) the statement reads.

        Uses a recording authorizer over ``EXPLAIN QUERY PLAN`` so the answer
        comes from SQLite's own preparation: view aliases resolve to their base
        tables, and a CTE named ``history`` shadows the base table (no read
        reported). Fails open: any error returns an empty set so ``execute``
        surfaces the real SQL error for the LLM.
        """
        seen: set[str] = set()

        def _record(action: int, arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
            if action == sqlite3.SQLITE_READ and arg1 in {"history", "statistics"}:
                seen.add(str(arg1))
            return sqlite3.SQLITE_OK

        try:
            self.conn.set_authorizer(_record)
            self.conn.execute(f"explain query plan {sql}")
        except Exception:  # noqa: BLE001 - fail open so execute() surfaces the real SQL error
            # A malformed statement surfaces its real error from execute().
            return set()
        finally:
            self.conn.set_authorizer(None)
        return seen

    def execute(self, sql: str, deadline: float) -> QueryResult:
        """Execute one bounded read-only SQL statement."""
        self._arm_user_sql_guard(deadline)
        try:
            cursor = self.conn.execute(sql)
            fetched = cursor.fetchmany(MAX_SQL_RESULT_ROWS + 1)
        except sqlite3.ProgrammingError as err:
            message = str(err)
            reason = "submit exactly one SQL statement" if "one statement at a time" in message.lower() else message
            raise HelperExecutionError("query", "sql_syntax_error", {"reason": reason}) from err
        except sqlite3.OperationalError as err:
            raise self._refine_sql_error(str(err)) from err
        except sqlite3.DatabaseError as err:
            raise HelperExecutionError("query", "sql_read_only", {"reason": str(err)}) from err
        finally:
            self._disarm_user_sql_guard()
        rows = [dict(row) for row in fetched[:MAX_SQL_RESULT_ROWS]]
        return QueryResult(rows=cast(list[dict[str, object]], rows), truncated=len(fetched) > MAX_SQL_RESULT_ROWS)

    def close(self) -> None:
        """Close the per-run in-memory database."""
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def _create_schema(self) -> None:
        statements: list[str] = []
        for table, columns in SCHEMA_TABLES.items():
            body = ", ".join(f"{name} {type_}" for name, type_ in columns)
            statements.append(f"create table {table}({body});")
        for view, (base, explicit, distinct) in SCHEMA_VIEWS.items():
            select = "*" if explicit is None else ", ".join(explicit)
            distinct_sql = "distinct " if distinct else ""
            statements.append(f"create view {view} as select {distinct_sql}{select} from {base};")
        # Dedup on every loaded column so only byte-identical rows collapse on a
        # re-load; COALESCE normalizes NULLs (SQLite treats NULLs as distinct in
        # unique indexes) so two identical rows with nullable fields still dedup.
        statements.append(
            "create unique index history_row_unique on history("
            "entity_id, when_iso, state,"
            "coalesce(when_ts, -1), coalesce(value, -1e308),"
            "coalesce(area_id, ''), coalesce(floor_id, ''), coalesce(device_id, ''), coalesce(domain, ''));"
        )
        statements.append("create unique index statistics_row_unique on statistics(statistic_id, when_iso);")
        self.conn.executescript("\n".join(statements))

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

    def _load_states(self) -> None:
        """Load visible snapshot states into SQLite before read-only guard activation."""
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
        """Return a structured SQL helper error with targeted guidance."""
        from ..guidance import FailureContext, Intent, advise

        lowered = message.lower()
        if "no such table" in lowered:
            requested = message.rsplit(":", 1)[-1].strip() or message
            return HelperExecutionError(
                "query",
                "sql_unknown_table",
                {"reason": message},
                guidance=advise(
                    self.snapshot,
                    FailureContext(intent=Intent.SQL_TABLE, requested=requested),
                ).to_payload(),
            )
        if "no such column" in lowered:
            requested = message.rsplit(":", 1)[-1].strip() or message
            table = None
            if match := _SQL_COLUMN_QUALIFIER.search(message):
                table_ref = match.group("table")
                table = _unquote_identifier(table_ref) if table_ref else None
                requested = _unquote_identifier(match.group(2))
            # SQLite omits the table qualifier for unqualified column errors; default
            # to ``states``, the primary table LLM-queried read paths target.
            table = table if table in _SCHEMA_TABLE_NAMES else "states"
            return HelperExecutionError(
                "query",
                "sql_unknown_column",
                {"reason": message},
                guidance=advise(
                    self.snapshot,
                    FailureContext(intent=Intent.SQL_COLUMN, requested=requested, table_name=table),
                ).to_payload(),
            )
        if "not authorized" in lowered or "readonly" in lowered or "only select" in lowered:
            return HelperExecutionError("query", "sql_read_only", {"reason": message})
        if "interrupted" in lowered:
            return HelperExecutionError("query", "sql_timeout", {})
        return HelperExecutionError("query", "sql_syntax_error", {"reason": message})


def _cap_newest_first(shaped: list[dict[str, object]]) -> tuple[list[dict[str, object]], bool]:
    """Return up to MAX_HISTORY_LOAD_ROWS newest rows, plus a truncation flag.

    Newest first by when_ts; NULL timestamps (parse failures) are treated as
    oldest so they are the first to drop when the cap applies. The ISO string
    is a deterministic tiebreaker.
    """
    if len(shaped) <= MAX_HISTORY_LOAD_ROWS:
        return shaped, False
    kept = sorted(
        shaped,
        key=lambda item: (item.get("when_ts") or 0.0, str(item.get("when_iso") or "")),
        reverse=True,
    )[:MAX_HISTORY_LOAD_ROWS]
    return kept, True


def _unquote_identifier(table_ref: str) -> str:
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


def _timestamp(value: str) -> float | None:
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    return dt_util.as_utc(parsed).timestamp()
