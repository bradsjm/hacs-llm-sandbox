"""Server-side history aggregates for recorder-backed LLM tools."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal

from homeassistant.core import State
from homeassistant.util import dt as dt_util

type AggregateMode = Literal[
    "count_transitions",
    "time_in_state",
    "state_counts",
    "first_seen",
    "last_seen",
    "on_duration",
]
type HistoryRow = State | dict[str, object]
type AggregateSummary = dict[str, object]


@dataclass(frozen=True, slots=True)
class AggregateFilters:
    """Optional aggregate filters supplied by the LLM tool input."""

    from_state: str | None = None
    to_state: str | None = None


type AggregateFn = Callable[[list[HistoryRow], datetime, datetime, AggregateFilters], AggregateSummary]


def count_transitions(
    rows: list[HistoryRow],
    _start: datetime,
    _end: datetime,
    filters: AggregateFilters,
) -> AggregateSummary:
    """Count adjacent state transitions, optionally matching from/to states."""
    transitions = 0
    for previous, current in pairwise(rows):
        previous_state = _row_state(previous)
        current_state = _row_state(current)
        if previous_state == current_state:
            continue
        if filters.from_state is not None and previous_state != filters.from_state:
            continue
        if filters.to_state is not None and current_state != filters.to_state:
            continue
        transitions += 1

    summary: AggregateSummary = {"transitions": transitions}
    if filters.from_state is not None:
        summary["from_state"] = filters.from_state
    if filters.to_state is not None:
        summary["to_state"] = filters.to_state
    return summary


def time_in_state(
    rows: list[HistoryRow],
    start: datetime,
    end: datetime,
    _filters: AggregateFilters,
) -> AggregateSummary:
    """Return seconds spent in each state over the requested window."""
    totals: dict[str, float] = {}
    for state, seconds in _state_periods(rows, start, end):
        totals[state] = totals.get(state, 0.0) + seconds
    return {"time_in_state": totals, "unit": "seconds"}


def state_counts(
    rows: list[HistoryRow],
    _start: datetime,
    _end: datetime,
    _filters: AggregateFilters,
) -> AggregateSummary:
    """Return observed state sample counts, including the start-time state."""
    counts = Counter(_row_state(row) for row in rows)
    return {"state_counts": dict(counts)}


def first_seen(
    rows: list[HistoryRow],
    _start: datetime,
    _end: datetime,
    _filters: AggregateFilters,
) -> AggregateSummary:
    """Return the first state row observed for the query."""
    return {"first_seen": _seen(rows[0]) if rows else None}


def last_seen(
    rows: list[HistoryRow],
    _start: datetime,
    _end: datetime,
    _filters: AggregateFilters,
) -> AggregateSummary:
    """Return the last state row observed for the query."""
    return {"last_seen": _seen(rows[-1]) if rows else None}


def on_duration(
    rows: list[HistoryRow],
    start: datetime,
    end: datetime,
    _filters: AggregateFilters,
) -> AggregateSummary:
    """Return seconds spent in exact state ``on`` over the requested window."""
    seconds = sum(duration for state, duration in _state_periods(rows, start, end) if state == "on")
    return {"on_duration": seconds, "unit": "seconds"}


AGGREGATORS: dict[AggregateMode, AggregateFn] = {
    "count_transitions": count_transitions,
    "time_in_state": time_in_state,
    "state_counts": state_counts,
    "first_seen": first_seen,
    "last_seen": last_seen,
    "on_duration": on_duration,
}


def _state_periods(rows: list[HistoryRow], start: datetime, end: datetime) -> list[tuple[str, float]]:
    """Tile known states over ``[start, end]`` and return ``(state, seconds)`` periods."""
    periods: list[tuple[str, float]] = []
    active_state: str | None = None
    active_at = start

    for row in sorted(rows, key=_row_time):
        row_at = _row_time(row)
        state = _row_state(row)
        if row_at <= start:
            # The recorder start-time state is the value active at the left boundary.
            active_state = state
            active_at = start
            continue
        if row_at >= end:
            if active_state is not None and active_at < end:
                periods.append((active_state, (end - active_at).total_seconds()))
            return periods
        if active_state is not None:
            periods.append((active_state, (row_at - active_at).total_seconds()))
        active_state = state
        active_at = row_at

    if active_state is not None and active_at < end:
        periods.append((active_state, (end - active_at).total_seconds()))
    return periods


def _seen(row: HistoryRow) -> dict[str, str]:
    """Return the compact seen-state shape for first/last aggregate modes."""
    return {"state": _row_state(row), "at": _row_time(row).isoformat()}


def _row_state(row: HistoryRow) -> str:
    """Return a row's state value as a string."""
    if isinstance(row, State):
        return row.state
    return str(row.get("state"))


def _row_time(row: HistoryRow) -> datetime:
    """Return a row's timestamp as a UTC-aware datetime."""
    if isinstance(row, State):
        return dt_util.as_utc(row.last_changed)

    timestamp = row.get("last_changed") or row.get("last_updated")
    if isinstance(timestamp, datetime):
        return dt_util.as_utc(timestamp)
    if isinstance(timestamp, int | float):
        return datetime.fromtimestamp(timestamp, UTC)
    if isinstance(timestamp, str):
        parsed = dt_util.parse_datetime(timestamp)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise ValueError("history row missing timestamp")
