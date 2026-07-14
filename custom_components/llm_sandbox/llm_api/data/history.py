"""History analytics for recorder-backed LLM tools."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
import math
import statistics
from typing import Literal, cast

from homeassistant.core import State
from homeassistant.util import dt as dt_util

from ...snapshot.models import HomeSnapshot, SafeState
from ..errors import RecoverableToolError
from .numeric import finite_float

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
type NumericOp = Literal["count", "min", "max", "mean", "median", "sum", "stdev"]

NUMERIC_OPS: frozenset[str] = frozenset({"count", "min", "max", "mean", "median", "sum", "stdev"})
GROUP_KEYS: frozenset[str] = frozenset({"entity_id", "domain", "area_id", "floor_id", "device_id"})
DEFAULT_ANALYTICS_LIMIT = 500
_SEQUENCE_DEPENDENT_MODES = frozenset({"count_transitions", "time_in_state", "on_duration"})
_TIMESTAMP_ORDERED_MODES = frozenset({"first_seen", "last_seen"})
_BUCKETED_CARRY_FORWARD_MODES = frozenset({"time_in_state", "on_duration"})


@dataclass(frozen=True, slots=True)
class AggregateFilters:
    """Optional legacy aggregate filters supplied by the LLM tool input."""

    from_state: str | None = None
    to_state: str | None = None


@dataclass(frozen=True, slots=True)
class Condition:
    """One declarative row filter."""

    field: str
    op: str
    value: object


@dataclass(frozen=True, slots=True)
class AnalyticsSpec:
    """Declarative analytics request over flat history rows.

    Validation lives entirely in ``analytics_spec_from_data``; runtime
    aggregation reads these typed fields directly without re-checking.
    """

    mode: AggregateMode | None = None
    filters: AggregateFilters = AggregateFilters()
    value_operations: tuple[NumericOp, ...] = ()
    group_by: tuple[str, ...] = ()
    bucket: str | None = None
    where: tuple[Condition, ...] = ()
    order_by: str | None = None
    limit: int | None = None


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
    filters: AggregateFilters,
) -> AggregateSummary:
    """Return the first state row observed for the query."""
    rows = _rows_matching_to_state(rows, filters)
    return {"first_seen": _seen(rows[0]) if rows else None}


def last_seen(
    rows: list[HistoryRow],
    _start: datetime,
    _end: datetime,
    filters: AggregateFilters,
) -> AggregateSummary:
    """Return the last state row observed for the query."""
    rows = _rows_matching_to_state(rows, filters)
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


def analytics_spec_from_data(data: Mapping[str, object]) -> AnalyticsSpec:
    """Build a validated analytics spec from tool/facade arguments.

    All mode/op/group validation happens here once; ``run_analytics`` and its
    helpers read the typed spec fields directly without re-checking.
    """
    mode, filters, value_operations = _parse_aggregate(data)
    group_by = tuple(str(item) for item in _ensure_list(data.get("group_by")))
    unknown_groups = sorted(set(group_by) - GROUP_KEYS)
    if unknown_groups:
        raise RecoverableToolError(
            "analytics_unknown_group_key",
            {"group_key": unknown_groups[0], "valid": ", ".join(sorted(GROUP_KEYS))},
        )
    where = tuple(_condition(item) for item in _ensure_list(data.get("where")))
    limit = cast(str | int | float | None, data.get("limit"))
    return AnalyticsSpec(
        mode=mode,
        filters=filters,
        value_operations=value_operations,
        group_by=group_by,
        bucket=cast(str | None, data.get("bucket")),
        where=where,
        order_by=cast(str | None, data.get("order_by")),
        limit=int(limit) if limit is not None else None,
    )


def _parse_aggregate(
    data: Mapping[str, object],
) -> tuple[AggregateMode | None, AggregateFilters, tuple[NumericOp, ...]]:
    """Parse named aggregate modes and numeric operations."""
    aggregate = data.get("aggregate")
    if aggregate is not None and not isinstance(aggregate, str):
        raise RecoverableToolError("invalid_tool_input", {"error": "aggregate must be a string mode"})

    data_from_state = cast(str | None, data.get("from_state"))
    data_to_state = cast(str | None, data.get("to_state"))
    mode: AggregateMode | None = None
    effective_from_state = data_from_state
    effective_to_state = data_to_state
    operation_names = tuple(str(item) for item in _ensure_list(data.get("value_operations")))
    unknown = sorted(set(operation_names) - NUMERIC_OPS)
    if unknown:
        raise RecoverableToolError("analytics_unknown_op", {"op": unknown[0], "valid": ", ".join(sorted(NUMERIC_OPS))})
    value_operations = cast(tuple[NumericOp, ...], operation_names)

    if aggregate is not None:
        if aggregate not in AGGREGATORS:
            raise RecoverableToolError(
                "analytics_unknown_op", {"op": aggregate, "valid": ", ".join(sorted(AGGREGATORS))}
            )
        if value_operations:
            raise RecoverableToolError(
                "invalid_tool_input", {"error": "aggregate mode and value_operations cannot be combined"}
            )
        mode = aggregate

    if effective_from_state is not None and mode != "count_transitions":
        raise RecoverableToolError("invalid_tool_input", {"error": "from_state is only valid with count_transitions"})
    if effective_to_state is not None and mode not in {"count_transitions", "first_seen", "last_seen"}:
        raise RecoverableToolError(
            "invalid_tool_input",
            {"error": "to_state is only valid with count_transitions, first_seen, or last_seen"},
        )

    return mode, AggregateFilters(effective_from_state, effective_to_state), value_operations


def run_analytics(
    rows: Sequence[HistoryRow],
    spec: AnalyticsSpec,
    window: tuple[datetime, datetime],
    snapshot: HomeSnapshot,
) -> list[dict[str, object]]:
    """Run declarative analytics over history rows and return flat JSON-safe dicts."""
    start, end = window
    bucket_seconds = _bucket_seconds(spec.bucket) if spec.bucket is not None else None
    filtered = [row for row in rows if all(_matches(row, condition, snapshot) for condition in spec.where)]

    # Bucketed state-carry modes attribute state intervals directly to overlapping buckets.
    if bucket_seconds is not None and _carries_state_across_buckets(spec):
        results = _duration_bucket_results(filtered, spec, start, end, bucket_seconds, snapshot)
    # Bucketed transition mode keeps the prior row at each bucket boundary.
    elif bucket_seconds is not None and _carries_transition_across_buckets(spec):
        buckets = _transition_buckets(filtered, spec, start, end, bucket_seconds, snapshot)
        if not _seed_empty_group(buckets, spec, filtered):
            return []
        results = _format_buckets(buckets, spec, start, end, bucket_seconds)
    # Row-assigned buckets handle unbucketed and non-carry aggregate modes.
    else:
        buckets = _row_buckets(filtered, spec, start, bucket_seconds, snapshot)
        if not _seed_empty_group(buckets, spec, filtered):
            return []
        results = _format_buckets(buckets, spec, start, end, bucket_seconds)

    _apply_order(results, spec.order_by)
    return results[: _bounded_limit(spec.limit)]


def _seed_empty_group[T](
    grouped: dict[tuple[object, ...], T],
    spec: AnalyticsSpec,
    empty: T,
) -> bool:
    """Apply the shared empty-dimension policy in place; return True to proceed.

    When a dimension (bucket or group_by) was requested but produced no groups,
    returns False so the caller returns an empty list. Otherwise, when there are
    no groups at all, seeds a single unkeyed group holding ``empty`` so
    unbucketed/un grouped analytics still run over (possibly empty) filtered rows.
    """
    if grouped:
        return True
    if spec.bucket is not None or spec.group_by:
        return False
    grouped[()] = empty
    return True


def flat_history_rows(raw: Mapping[str, Iterable[HistoryRow]], snapshot: HomeSnapshot) -> list[dict[str, object]]:
    """Flatten recorder history streams for facade SQL/history loading."""
    rows: list[dict[str, object]] = []
    for entity_id, entity_rows in raw.items():
        state = snapshot.states.get(entity_id)
        rows.extend(_flat_history_row(entity_id, state, row) for row in entity_rows)
    return sorted(rows, key=lambda item: (str(item["entity_id"]), str(item["when"])))


def _flat_history_row(entity_id: str, state: SafeState | None, row: HistoryRow) -> dict[str, object]:
    """Render one recorder history row for the facade SQL/history table."""
    when = _row_time(row)
    return {
        "entity_id": entity_id,
        "domain": entity_id.split(".", 1)[0],
        "area_id": state.area_id if state is not None else None,
        "floor_id": state.floor_id if state is not None else None,
        "device_id": state.device_id if state is not None else None,
        "when": when.isoformat(),
        "when_ts": when.timestamp(),
        "state": _row_state(row),
        "value": _row_value(row),
    }


def _aggregate_group(rows: list[HistoryRow], spec: AnalyticsSpec, start: datetime, end: datetime) -> dict[str, object]:
    # No mode and no numeric value operations: a plain count over the group.
    if spec.mode is None and not spec.value_operations:
        return {"count": len(rows)}
    if spec.mode is not None:
        mode = spec.mode
        if mode in _SEQUENCE_DEPENDENT_MODES:
            return _aggregate_entity_streams(rows, mode, start, end, spec.filters)
        if mode in _TIMESTAMP_ORDERED_MODES:
            rows = sorted(rows, key=_row_time)
        return AGGREGATORS[mode](rows, start, end, spec.filters)

    output: dict[str, object] = {}
    values: list[float] = []
    skipped = 0
    for row in rows:
        number = finite_float(_field_value(row, "value", None))
        if number is None:
            skipped += 1
            continue
        values.append(number)
    for op in spec.value_operations:
        output[f"value_{op}"] = _numeric_result(values, op)
    if skipped:
        output["value_skipped_non_numeric"] = skipped
    return output


def _bounded_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_ANALYTICS_LIMIT
    return max(1, min(limit, DEFAULT_ANALYTICS_LIMIT))


def _bucket_sort_key(bucket_key: tuple[object, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((1, "") if value is None else (0, str(value)) for value in bucket_key)


def _aggregate_entity_streams(
    rows: list[HistoryRow], mode: AggregateMode, start: datetime, end: datetime, filters: AggregateFilters
) -> dict[str, object]:
    grouped: dict[str, list[HistoryRow]] = {}
    for row in rows:
        grouped.setdefault(str(_field_value(row, "entity_id", None)), []).append(row)
    summaries = [
        AGGREGATORS[mode](sorted(group_rows, key=_row_time), start, end, filters) for group_rows in grouped.values()
    ]
    return _combine_sequence_summaries(mode, summaries)


def _combine_sequence_summaries(mode: AggregateMode, summaries: list[AggregateSummary]) -> dict[str, object]:
    if mode == "count_transitions":
        return {"transitions": sum(cast(int, summary["transitions"]) for summary in summaries)}
    if mode == "time_in_state":
        totals: dict[str, float] = {}
        for summary in summaries:
            for state, seconds in cast(dict[str, float], summary["time_in_state"]).items():
                totals[state] = totals.get(state, 0.0) + seconds
        return {"time_in_state": totals, "unit": "seconds"}
    if mode == "on_duration":
        return {
            "on_duration": math.fsum(cast(float, summary["on_duration"]) for summary in summaries),
            "unit": "seconds",
        }
    return {}


def _carries_state_across_buckets(spec: AnalyticsSpec) -> bool:
    """Return whether the aggregate needs pre-bucket state history."""
    return spec.mode in _BUCKETED_CARRY_FORWARD_MODES


def _carries_transition_across_buckets(spec: AnalyticsSpec) -> bool:
    """Return whether transition buckets need the prior state row."""
    return spec.mode == "count_transitions"


def _transition_buckets(
    rows: list[HistoryRow],
    spec: AnalyticsSpec,
    start: datetime,
    end: datetime,
    bucket_seconds: int,
    snapshot: HomeSnapshot,
) -> dict[tuple[object, ...], list[HistoryRow]]:
    """Build transition buckets with each entity's prior row at bucket boundaries."""
    grouped: dict[tuple[object, ...], list[HistoryRow]] = {}
    for row in rows:
        group_key = tuple(_group_value(row, key, snapshot) for key in spec.group_by)
        grouped.setdefault(group_key, []).append(row)

    buckets: dict[tuple[object, ...], list[HistoryRow]] = {}
    for group_key, group_rows in grouped.items():
        entity_rows: dict[str, list[HistoryRow]] = {}
        for row in group_rows:
            entity_rows.setdefault(str(_field_value(row, "entity_id", None)), []).append(row)
        for stream_rows in entity_rows.values():
            previous: HistoryRow | None = None
            for row in sorted(stream_rows, key=_row_time):
                row_at = _row_time(row)
                if row_at < start or row_at >= end:
                    previous = row
                    continue
                bucket_start = _bucket_start(row_at, start, bucket_seconds)
                bucket_key = (bucket_start.isoformat(), *group_key)
                if previous is not None and _row_time(previous) < bucket_start:
                    buckets.setdefault(bucket_key, []).append(previous)
                buckets.setdefault(bucket_key, []).append(row)
                previous = row
    return buckets


def _duration_bucket_results(
    rows: list[HistoryRow],
    spec: AnalyticsSpec,
    start: datetime,
    end: datetime,
    bucket_seconds: int,
    snapshot: HomeSnapshot,
) -> list[dict[str, object]]:
    """Attribute state durations to bucket overlaps in one pass per entity stream."""
    grouped = _group_duration_streams(rows, spec, snapshot)
    if not _seed_empty_group(grouped, spec, {}):
        return []

    bucket_state: dict[tuple[object, ...], dict[str, float]] = {}
    for group_key, entity_rows in grouped.items():
        # Initialize each requested group bucket so empty duration buckets remain visible.
        bucket_cursor = start
        while bucket_cursor < end:
            bucket_state.setdefault((bucket_cursor.isoformat(), *group_key), {})
            bucket_cursor += timedelta(seconds=bucket_seconds)
        for stream_rows in entity_rows.values():
            for state, period_start, period_end in _state_intervals(stream_rows, start, end):
                cursor = _bucket_start(period_start, start, bucket_seconds)
                while cursor < end and cursor < period_end:
                    bucket_end = min(cursor + timedelta(seconds=bucket_seconds), end)
                    if period_start < bucket_end and period_end > cursor:
                        seconds = (min(period_end, bucket_end) - max(period_start, cursor)).total_seconds()
                        totals = bucket_state.setdefault((cursor.isoformat(), *group_key), {})
                        totals[state] = totals.get(state, 0.0) + seconds
                    cursor += timedelta(seconds=bucket_seconds)

    return _format_duration_bucket_results(bucket_state, spec)


def _group_duration_streams(
    rows: list[HistoryRow], spec: AnalyticsSpec, snapshot: HomeSnapshot
) -> dict[tuple[object, ...], dict[str, list[HistoryRow]]]:
    """Group duration rows by requested dimensions, then by independent entity stream."""
    grouped: dict[tuple[object, ...], dict[str, list[HistoryRow]]] = {}
    for row in rows:
        group_key = tuple(_group_value(row, key, snapshot) for key in spec.group_by)
        entity_id = str(_field_value(row, "entity_id", None))
        grouped.setdefault(group_key, {}).setdefault(entity_id, []).append(row)
    return grouped


def _format_duration_bucket_results(
    bucket_state: dict[tuple[object, ...], dict[str, float]], spec: AnalyticsSpec
) -> list[dict[str, object]]:
    """Format duration bucket state totals in deterministic bucket/group order."""
    results: list[dict[str, object]] = []
    for bucket_key, state_totals in sorted(bucket_state.items(), key=lambda item: _bucket_sort_key(item[0])):
        output: dict[str, object] = {"bucket": bucket_key[0]}
        for index, group_field in enumerate(spec.group_by):
            output[group_field] = bucket_key[index + 1]
        if spec.mode == "on_duration":
            output.update({"on_duration": state_totals.get("on", 0.0), "unit": "seconds"})
        else:
            output.update({"time_in_state": state_totals, "unit": "seconds"})
        results.append(output)
    return results


def _row_buckets(
    rows: list[HistoryRow],
    spec: AnalyticsSpec,
    start: datetime,
    bucket_seconds: int | None,
    snapshot: HomeSnapshot,
) -> dict[tuple[object, ...], list[HistoryRow]]:
    """Build buckets by assigning each row to its own time/group key."""
    buckets: dict[tuple[object, ...], list[HistoryRow]] = {}
    for row in rows:
        key: list[object] = []
        if bucket_seconds is not None:
            key.append(_bucket_start(_row_time(row), start, bucket_seconds).isoformat())
        key.extend(_group_value(row, group_key, snapshot) for group_key in spec.group_by)
        buckets.setdefault(tuple(key), []).append(row)
    return buckets


def _format_buckets(
    buckets: dict[tuple[object, ...], list[HistoryRow]],
    spec: AnalyticsSpec,
    start: datetime,
    end: datetime,
    bucket_seconds: int | None,
) -> list[dict[str, object]]:
    """Format row buckets into aggregate result dictionaries."""
    results: list[dict[str, object]] = []
    for bucket_key, group_rows in sorted(buckets.items(), key=lambda item: _bucket_sort_key(item[0])):
        offset = 0
        output: dict[str, object] = {}
        group_start = start
        group_end = end
        if bucket_seconds is not None:
            output["bucket"] = bucket_key[0]
            group_start = datetime.fromisoformat(str(bucket_key[0]))
            group_end = min(group_start + timedelta(seconds=bucket_seconds), end)
            offset = 1
        for index, group_key in enumerate(spec.group_by):
            output[group_key] = bucket_key[offset + index]
        output.update(_aggregate_group(group_rows, spec, group_start, group_end))
        results.append(output)
    return results


def _numeric_result(values: list[float], op: str) -> int | float | None:
    if op == "count":
        return len(values)
    if not values:
        return None
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    if op == "mean":
        return statistics.fmean(values)
    if op == "median":
        return statistics.median(values)
    if op == "sum":
        return math.fsum(values)
    if op == "stdev":
        return statistics.stdev(values) if len(values) > 1 else 0.0
    return None


def _order_rank(value: object) -> tuple[int, float | str]:
    number = finite_float(value)
    if number is not None:
        return (0, number)
    return (1, str(value))


def _apply_order(results: list[dict[str, object]], order_by: str | None) -> None:
    if order_by is None:
        return
    reverse = order_by.startswith("-")
    field = order_by[1:] if reverse else order_by
    non_null = [row for row in results if row.get(field) is not None]
    nulls = [row for row in results if row.get(field) is None]
    non_null.sort(key=lambda row: _order_rank(row.get(field)), reverse=reverse)
    results[:] = non_null + nulls


def _condition(value: object) -> Condition:
    if not isinstance(value, Mapping):
        raise RecoverableToolError("invalid_tool_input", {"error": "where entries must be objects"})
    field = value.get("field")
    if not isinstance(field, str) or not field:
        raise RecoverableToolError("invalid_tool_input", {"error": "where entries must include a non-empty field"})
    return Condition(field=field, op=str(value.get("op", "eq")), value=value.get("value"))


def _matches(row: HistoryRow, condition: Condition, snapshot: HomeSnapshot) -> bool:
    left = _field_value(row, condition.field, snapshot)
    right = condition.value
    if condition.op in ("eq", "=="):
        return left == right
    if condition.op in ("ne", "!="):
        return left != right
    if condition.op == "in":
        return isinstance(right, Sequence) and not isinstance(right, str) and left in right
    left_number = finite_float(left)
    right_number = finite_float(right)
    if left_number is None or right_number is None:
        return False
    if condition.op == "gt":
        return left_number > right_number
    if condition.op == "gte":
        return left_number >= right_number
    if condition.op == "lt":
        return left_number < right_number
    if condition.op == "lte":
        return left_number <= right_number
    raise RecoverableToolError("analytics_unknown_op", {"op": condition.op, "valid": "eq, ne, in, gt, gte, lt, lte"})


def _field_value(row: HistoryRow, field: str, snapshot: HomeSnapshot | None) -> object:
    if field == "value":
        return _row_value(row)
    if field == "state":
        return _row_state(row)
    if field == "entity_id":
        if isinstance(row, State):
            return row.entity_id
        return row.get("entity_id")
    if field == "domain":
        entity_id = str(_field_value(row, "entity_id", snapshot) or "")
        return entity_id.split(".", 1)[0] if "." in entity_id else None
    if field in GROUP_KEYS and snapshot is not None:
        row_entity_id = cast(str | None, _field_value(row, "entity_id", snapshot))
        state = snapshot.states.get(row_entity_id or "")
        return getattr(state, field) if state is not None else None
    if isinstance(row, State):
        return row.attributes.get(field)
    return row.get(field)


def _group_value(row: HistoryRow, group_key: str, snapshot: HomeSnapshot) -> object:
    return _field_value(row, group_key, snapshot)


def _bucket_seconds(value: str) -> int:
    unit = value[-1:]
    amount = value[:-1]
    if not amount.isdigit() or unit not in {"m", "h", "d"}:
        raise RecoverableToolError("analytics_bad_bucket", {"bucket": value, "examples": "15m, 1h, 1d"})
    seconds = int(amount) * {"m": 60, "h": 3600, "d": 86400}[unit]
    if seconds <= 0:
        raise RecoverableToolError("analytics_bad_bucket", {"bucket": value, "examples": "15m, 1h, 1d"})
    return seconds


def _bucket_start(value: datetime, start: datetime, seconds: int) -> datetime:
    elapsed = max(0, int((value - start).total_seconds()))
    return start + timedelta(seconds=(elapsed // seconds) * seconds)


def _state_intervals(rows: list[HistoryRow], start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    intervals: list[tuple[str, datetime, datetime]] = []
    active_state: str | None = None
    active_at = start
    for row in sorted(rows, key=_row_time):
        row_at = _row_time(row)
        state = _row_state(row)
        if row_at <= start:
            active_state = state
            active_at = start
            continue
        if row_at >= end:
            if active_state is not None and active_at < end:
                intervals.append((active_state, active_at, end))
            return intervals
        if active_state is not None:
            intervals.append((active_state, active_at, row_at))
        active_state = state
        active_at = row_at
    if active_state is not None and active_at < end:
        intervals.append((active_state, active_at, end))
    return intervals


def _state_periods(rows: list[HistoryRow], start: datetime, end: datetime) -> list[tuple[str, float]]:
    return [
        (state, (period_end - period_start).total_seconds())
        for state, period_start, period_end in _state_intervals(rows, start, end)
    ]


def _seen(row: HistoryRow) -> dict[str, str]:
    return {"state": _row_state(row), "at": _row_time(row).isoformat()}


def _rows_matching_to_state(rows: list[HistoryRow], filters: AggregateFilters) -> list[HistoryRow]:
    if filters.to_state is None:
        return rows
    return [row for row in rows if _row_state(row) == filters.to_state]


def _row_state(row: HistoryRow) -> str:
    if isinstance(row, State):
        return row.state
    return str(row.get("state"))


def _row_value(row: HistoryRow) -> float | None:
    return finite_float(_row_state(row))


def _row_time(row: HistoryRow) -> datetime:
    if isinstance(row, State):
        return dt_util.as_utc(row.last_changed)
    timestamp = row.get("when") or row.get("last_changed") or row.get("last_updated")
    if isinstance(timestamp, datetime):
        return dt_util.as_utc(timestamp)
    if isinstance(timestamp, int | float):
        return datetime.fromtimestamp(timestamp, UTC)
    if isinstance(timestamp, str):
        parsed = dt_util.parse_datetime(timestamp)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise ValueError("history row missing timestamp")


def _ensure_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]
