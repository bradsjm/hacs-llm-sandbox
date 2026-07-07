"""Tool runner for the dev-only eval harness."""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import cast

from custom_components.llm_sandbox.const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    DEFAULT_LOGBOOK_WINDOW_HOURS,
    DEFAULT_STATISTICS_WINDOW_HOURS,
    MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
    MAX_HISTORY_ATTRIBUTES,
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_RECORDER_LOOKBACK_HOURS,
    MAX_STATISTICS_LOOKBACK_HOURS,
    MAX_STATISTICS_ROWS,
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.executor import async_execute_home_code
from custom_components.llm_sandbox.llm_api.executor_support import ExecutionState
from custom_components.llm_sandbox.llm_api.facade_views import build_llm_context
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from custom_components.llm_sandbox.llm_api.runtime import RuntimeContext
from custom_components.llm_sandbox.llm_api.errors import RecoverableToolError
from custom_components.llm_sandbox.llm_api.tools._analytics import (
    AGGREGATORS,
    AggregateFilters,
    AggregateMode,
    HistoryRow,
    analytics_spec_from_data,
    flat_history_rows,
    run_analytics,
)
from custom_components.llm_sandbox.llm_api.tools._cursor import (
    _LOGBOOK_CURSOR_KEY,
    INVALID_CURSOR,
    Cursor,
    decode_cursor,
    encode_cursor,
    paginate_stream,
)
from custom_components.llm_sandbox.llm_api.tools.recorder import (
    STATISTIC_VALUE_TYPES,
    recorder_error_envelope,
    resolve_entity_ids,
)
from custom_components.llm_sandbox.runtime import SandboxSettings
from custom_components.llm_sandbox.snapshot import finalize_snapshot
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot, SnapshotScope
from homeassistant.util import dt as dt_util

from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.schema import EvalCase, ToolCall, ToolOutcome

EVAL_SCOPE: SnapshotScope = SnapshotScope(
    assistant="conversation",
    restrict_to_assist_exposed=False,
    exclude_hidden=True,
    excluded_entity_categories=frozenset({"config", "diagnostic"}),
)

_DEFAULT_STATISTIC_VALUE_PRIORITY = ("state", "mean", "sum", "min", "max")


def apply_scope(
    snapshot: HomeSnapshot,
    scope: SnapshotScope,
    *,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Return a new snapshot with entities failing the offline scope checks removed.

    Mirrors production ``_passes_visibility`` for the offline-applicable fields only
    (``exclude_hidden`` + ``excluded_entity_categories``). Assist-exposure filtering
    needs live HA and stays a ``build_snapshot`` concern; the eval scope disables it.
    Collection pruning, state enrichment, and index rebuilding are delegated to the
    production snapshot finalizer.
    """
    visible: set[str] = set()
    for entity_id in snapshot.states:
        entry = snapshot.entities.get(entity_id)
        # Branch boundary: state-only entities skip registry-characteristic visibility checks.
        if entry is None:
            visible.add(entity_id)
            continue
        # Branch boundary: hidden registry entities are excluded when the eval scope asks for it.
        if scope.exclude_hidden and entry.hidden_by is not None:
            continue
        # Branch boundary: config/diagnostic registry entities are excluded by the eval scope.
        if entry.entity_category in scope.excluded_entity_categories:
            continue
        visible.add(entity_id)

    return finalize_snapshot(snapshot, visible=visible, anchor_device_id=anchor_device_id)


class RecordingInvoker:
    """Non-live service invoker: records validated ProposedAction dicts, returns None.

    This is the ONLY live seam in the executor path. It never touches Home Assistant.
    """

    def __init__(self) -> None:
        """Initialize the in-memory action recording list."""
        self.calls: list[dict[str, object]] = []

    async def __call__(self, action: dict[str, object]) -> object:
        """Record one already-validated action without dispatching to live Home Assistant."""
        # Safety constraint: copy the proposed action and never call hass.services or any live callback.
        self.calls.append(dict(action))
        return None


def _for_scoring(action: Mapping[str, object]) -> dict[str, object]:
    """Normalize a recorded action so the frozen scorer can read domain/service separately.

    Production compact records fold domain into service (``domain.service``); the
    invoker's ProposedAction already carries separate domain/service. This split
    is eval-only and never mutates the model-facing result.
    """
    normalized = dict(action)
    if "domain" not in normalized:
        service = normalized.get("service")
        if isinstance(service, str) and "." in service:
            domain, _, svc = service.partition(".")
            normalized["domain"] = domain
            normalized["service"] = svc
    return normalized


async def run_tool(
    call: ToolCall,
    case: EvalCase,
    snapshot: HomeSnapshot,
    prompt_profile: PromptProfile,
    *,
    invoker: RecordingInvoker,
) -> ToolOutcome:
    """Run a selected eval tool against the caller-provided fresh snapshot."""
    tool_name = call.tool_name
    tool_args = _tool_args(call.tool_args)

    # Branch boundary: execute_home_code uses the real production executor with a non-live invoker.
    if tool_name == TOOL_EXECUTE_HOME_CODE:
        return await _run_execute(str(tool_args.get("code", "")), case, snapshot, prompt_profile, invoker)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_HISTORY:
        return _run_history(tool_args, case, snapshot)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_STATISTICS:
        return _run_statistics(tool_args, case, snapshot)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_LOGBOOK:
        return _run_logbook(tool_args, case, snapshot)

    # Branch boundary: unsupported model-selected tool.
    return ToolOutcome(ok=False, tool_name=tool_name, result=None, recorded_actions=(), error="unknown_tool")


async def _run_execute(
    code: str,
    case: EvalCase,
    snapshot: HomeSnapshot,
    prompt_profile: PromptProfile,
    invoker: RecordingInvoker,
) -> ToolOutcome:
    """Run execute_home_code through the production executor against a frozen snapshot."""
    area_id: str | None = None
    area_name: str | None = None
    floor_id: str | None = None
    floor_name: str | None = None

    # Branch boundary: derive request location only from the frozen snapshot device registry.
    if case.llm_context.device_id in snapshot.devices:
        device = snapshot.devices[case.llm_context.device_id]
        area_id = device.area_id
        # Branch boundary: an unassigned device has no area/floor context.
        if area_id is not None and area_id in snapshot.areas:
            area = snapshot.areas[area_id]
            area_name = area.name
            floor_id = area.floor_id
            # Branch boundary: an area may not be assigned to a floor.
            if floor_id is not None and floor_id in snapshot.floors:
                floor_name = snapshot.floors[floor_id].name

    ctx = build_llm_context(
        case.llm_context.platform,
        None,
        None,
        None,
        case.llm_context.language,
        None,
        case.llm_context.device_id,
        area_id,
        area_name,
        floor_id,
        floor_name,
    )
    settings = SandboxSettings(
        execution_timeout_seconds=10,
        helper_call_budget=20,
        scope=EVAL_SCOPE,
        actions_enabled=case.actions_enabled,
        action_domains=case.action_domains,
        prompt_profile=prompt_profile,
    )
    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=20),
        settings=settings,
        invoke=invoker,
        fetch_history=lambda entity_ids, start, end: _eval_fetch_history(case, snapshot, entity_ids, start, end),
        fetch_statistics=lambda statistic_ids, start, end: _eval_fetch_statistics(
            case, statistic_ids, start, end
        ),
        run_blocking=_eval_run_blocking,
        deadline=math.inf,
    )

    try:
        # Safety constraint: the executor receives only the frozen snapshot and RecordingInvoker live seam.
        result = await async_execute_home_code(code=code, snapshot=snapshot, llm_context=ctx, runtime=runtime)
    except Exception as err:  # noqa: BLE001
        return ToolOutcome(
            ok=False,
            tool_name=TOOL_EXECUTE_HOME_CODE,
            result=None,
            recorded_actions=tuple(_for_scoring(action) for action in invoker.calls),
            error=f"{type(err).__name__}: {err}",
        )

    result_dict = cast(dict[str, object], result)
    actions = _dict_list(result_dict.get("actions", []))
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_EXECUTE_HOME_CODE,
        result=result_dict,
        recorded_actions=tuple(_for_scoring(action) for action in actions),
        error=None,
    )


async def _eval_run_blocking(fn: Callable[[], object]) -> object:
    """Run eval-only blocking seams synchronously without live Home Assistant."""
    return fn()


async def _eval_fetch_history(
    case: EvalCase,
    snapshot: HomeSnapshot,
    entity_ids: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Return fixture history as production flat rows for hass.history/query in evals."""
    history = _recorder_section(get_home(case.home), "history")
    scoped = {
        entity_id: _windowed_rows(history.get(entity_id, []), start, end, _history_timestamp)
        for entity_id in entity_ids
    }
    return flat_history_rows(scoped, snapshot)


async def _eval_fetch_statistics(
    case: EvalCase,
    statistic_ids: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Return fixture statistics as flat rows for read-only eval SQL."""
    statistics = _recorder_section(get_home(case.home), "statistics")
    rows: list[dict[str, object]] = []
    for statistic_id in statistic_ids:
        for row in _windowed_rows(statistics.get(statistic_id, []), start, end, _statistics_timestamp):
            rows.append(
                {
                    "statistic_id": statistic_id,
                    "entity_id": statistic_id,
                    "when": _statistics_timestamp(row).isoformat(),
                    "mean": row.get("mean"),
                    "min": row.get("min"),
                    "max": row.get("max"),
                    "state": row.get("state"),
                    "sum": row.get("sum"),
                }
            )
    return rows


def _run_history(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed history rows in the production response envelope."""
    try:
        entity_ids = resolve_entity_ids(snapshot, tool_args, "entity_ids")
        aggregate = cast(AggregateMode | None, tool_args.get("aggregate"))
        analytics_requested = any(
            key in tool_args for key in ("aggregate", "group_by", "bucket", "where", "order_by", "limit")
        )
        legacy_aggregate = isinstance(aggregate, str) and not any(
            key in tool_args for key in ("group_by", "bucket", "where", "order_by", "limit")
        )
        if analytics_requested:
            if tool_args.get("attributes") is not None:
                raise RecoverableToolError(
                    "invalid_tool_input", {"error": "analytics cannot be combined with attributes"}
                )
            if tool_args.get("cursor") is not None:
                raise RecoverableToolError("invalid_tool_input", {"error": "analytics cannot be combined with cursor"})
            start, end, _cursor = _resolve_eval_window(
                tool_args,
                snapshot,
                default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                max_hours=MAX_HISTORY_AGGREGATE_LOOKBACK_HOURS,
            )
        else:
            start, end, cursor = _resolve_eval_window(
                tool_args,
                snapshot,
                default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
                max_hours=MAX_RECORDER_LOOKBACK_HOURS,
            )
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_HISTORY, err, snapshot)

    fixture = get_home(case.home)
    history = _recorder_section(fixture, "history")
    if analytics_requested and not legacy_aggregate:
        scoped = {
            entity_id: _windowed_rows(history.get(entity_id, []), start, end, _history_timestamp)
            for entity_id in entity_ids
        }
        try:
            spec = analytics_spec_from_data(tool_args)
            rows = run_analytics(cast(list[HistoryRow], flat_history_rows(scoped, snapshot)), spec, (start, end), snapshot)
        except RecoverableToolError as err:
            return _recoverable_recorder_error(TOOL_GET_HISTORY, err, snapshot)
        return ToolOutcome(
            ok=True,
            tool_name=TOOL_GET_HISTORY,
            result={"window": _window(start, end), "rows": rows},
            recorded_actions=(),
            error=None,
        )
    if legacy_aggregate:
        aggregate_mode = cast(AggregateMode, aggregate)
        filters = AggregateFilters(
            from_state=cast(str | None, tool_args.get("from_state")),
            to_state=cast(str | None, tool_args.get("to_state")),
        )
        aggregator = AGGREGATORS[aggregate_mode]
        summary = {
            entity_id: aggregator(
                cast(list[HistoryRow], _windowed_rows(history.get(entity_id, []), start, end, _history_timestamp)),
                start,
                end,
                filters,
            )
            for entity_id in entity_ids
        }
        result: dict[str, object] = {"window": _window(start, end), "mode": aggregate_mode, "summary": summary}
        return ToolOutcome(ok=True, tool_name=TOOL_GET_HISTORY, result=result, recorded_actions=(), error=None)

    requested_attributes = _requested_attributes(tool_args.get("attributes"))
    entities: dict[str, dict[str, object]] = {}
    next_cutoffs: dict[str, str] = {}
    for entity_id in entity_ids:
        rows = _windowed_rows(history.get(entity_id, []), start, end, _history_timestamp)
        shaped = [_history_row(row, requested_attributes) for row in rows]
        page, next_cutoff = paginate_stream(
            shaped,
            ts_of=lambda item: str(item[0][0]),
            budget=MAX_HISTORY_STATES,
            cutoff_iso=cursor.cutoffs.get(entity_id),
        )
        entities[entity_id] = _history_entity_payload(page)
        # State mutation point: carry exhausted streams so follow-up pages do not re-emit them.
        next_cutoffs[entity_id] = next_cutoff if next_cutoff is not None else ""
    if next_cutoffs and all(cutoff == "" for cutoff in next_cutoffs.values()):
        next_cutoffs = {}
    result = {"window": _window(start, end), "entities": entities}
    if next_cutoffs:
        result["next_cursor"] = encode_cursor(Cursor(start=start, end=end, cutoffs=next_cutoffs))
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_HISTORY,
        result=result,
        recorded_actions=(),
        error=None,
    )


def _run_statistics(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed statistics rows in the production response envelope."""
    try:
        statistic_ids = resolve_entity_ids(snapshot, tool_args, "statistic_ids")
        start, end, cursor = _resolve_eval_window(
            tool_args,
            snapshot,
            default_hours=DEFAULT_STATISTICS_WINDOW_HOURS,
            max_hours=MAX_STATISTICS_LOOKBACK_HOURS,
        )
        if tool_args.get("cursor") is not None:
            if cursor.period not in (None, "5minute", "hour", "day"):
                raise RecoverableToolError(INVALID_CURSOR, {})
            period = cursor.period or "hour"
            requested_types = cursor.statistic_types
            if requested_types is not None and (
                not requested_types or set(requested_types) - set(STATISTIC_VALUE_TYPES)
            ):
                raise RecoverableToolError(INVALID_CURSOR, {})
        else:
            period = str(tool_args.get("period", "hour"))
            requested_types = tuple(cast(list[str] | None, tool_args.get("types")) or ()) or None
            cursor = Cursor(start=start, end=end, cutoffs={}, period=period, statistic_types=requested_types)
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_STATISTICS, err, snapshot)

    fixture = get_home(case.home)
    statistics = _recorder_section(fixture, "statistics")
    rows: dict[str, dict[str, object]] = {}
    next_cutoffs: dict[str, str] = {}
    for statistic_id in statistic_ids:
        windowed = _windowed_rows(statistics.get(statistic_id, []), start, end, _statistics_timestamp)
        shaped = [_statistics_row(row, requested_types) for row in windowed]
        page, next_cutoff = paginate_stream(
            shaped,
            ts_of=lambda row: str(row[0]),
            budget=MAX_STATISTICS_ROWS,
            cutoff_iso=cursor.cutoffs.get(statistic_id),
        )
        rows[statistic_id] = _statistics_payload(page)
        # State mutation point: mark exhausted statistic streams to avoid duplicate continuation pages.
        next_cutoffs[statistic_id] = next_cutoff if next_cutoff is not None else ""
    if next_cutoffs and all(cutoff == "" for cutoff in next_cutoffs.values()):
        next_cutoffs = {}
    result: dict[str, object] = {
        "window": _window(start, end),
        "period": period,
        "statistics": rows,
    }
    if next_cutoffs:
        result["next_cursor"] = encode_cursor(
            Cursor(start=start, end=end, cutoffs=next_cutoffs, period=period, statistic_types=requested_types)
        )
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_STATISTICS,
        result=result,
        recorded_actions=(),
        error=None,
    )


def _run_logbook(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed logbook rows in the production response envelope."""
    try:
        entity_ids = resolve_entity_ids(snapshot, tool_args, "entity_ids")
        start, end, cursor = _resolve_eval_window(
            tool_args,
            snapshot,
            default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
        )
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_LOGBOOK, err, snapshot)

    fixture = get_home(case.home)
    logbook = _recorder_section(fixture, "logbook")
    entries = sorted(
        (
            _logbook_entry(entity_id, row)
            for entity_id in entity_ids
            for row in _windowed_rows(logbook.get(entity_id, []), start, end, _logbook_timestamp)
        ),
        key=_logbook_entry_timestamp,
    )
    page, next_cutoff = paginate_stream(
        entries,
        ts_of=_logbook_entry_timestamp,
        budget=MAX_LOGBOOK_ENTRIES,
        cutoff_iso=cursor.cutoffs.get(_LOGBOOK_CURSOR_KEY),
    )
    result: dict[str, object] = {"window": _window(start, end), "entries": page}
    if next_cutoff is not None:
        result["next_cursor"] = encode_cursor(Cursor(start=start, end=end, cutoffs={_LOGBOOK_CURSOR_KEY: next_cutoff}))
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_LOGBOOK,
        result=result,
        recorded_actions=(),
        error=None,
    )


def _tool_args(value: object) -> dict[str, object]:
    """Coerce model tool args to the narrow dict shape consumed by runners."""
    # Branch boundary: malformed tool_args are treated as empty per the tool-runner contract.
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, object], value)


def _recoverable_recorder_error(tool_name: str, err: RecoverableToolError, snapshot: HomeSnapshot) -> ToolOutcome:
    """Map production recoverable recorder errors to eval response envelopes."""
    return ToolOutcome(
        ok=True,
        tool_name=tool_name,
        result=cast(dict[str, object], recorder_error_envelope(err.key, err.placeholders, snapshot)),
        recorded_actions=(),
        error=None,
    )


def tool_result_message(tool_call_id: str, result: dict[str, object] | None) -> dict[str, object]:
    """Build the provider tool-result message, bounded for replay."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)[:8000]}


def _recorder_section(fixture: ModuleType, section: str) -> dict[str, list[dict[str, object]]]:
    """Return one typed canned recorder section from a fixture module."""
    recorder = cast(Callable[[], dict[str, object]], fixture.recorder)
    data = recorder()
    return cast(dict[str, list[dict[str, object]]], data[section])


def _resolve_eval_window(
    tool_args: Mapping[str, object],
    snapshot: HomeSnapshot,
    *,
    default_hours: int,
    max_hours: int,
) -> tuple[datetime, datetime, Cursor]:
    """Resolve the eval recorder query window using production cursor precedence."""
    if (cursor_in := tool_args.get("cursor")) is not None:
        # Branch boundary: cursors carry the original validated window for stable pagination.
        cursor = decode_cursor(cursor_in)
        return cursor.start, cursor.end, cursor

    end = (
        _parse_datetime(tool_args.get("end"))
        if tool_args.get("end") is not None
        else _parse_datetime(snapshot.created_at)
    )
    if tool_args.get("start") is not None:
        start = _parse_datetime(tool_args.get("start"))
    elif tool_args.get("hours") is not None:
        start = end - timedelta(hours=float(cast(float | int | str, tool_args["hours"])))
    else:
        start = end - timedelta(hours=default_hours)
    if start > end:
        raise RecoverableToolError("invalid_tool_input", {"error": "start after end"})
    if end - start > timedelta(hours=max_hours):
        raise RecoverableToolError("time_window_too_large", {"max_hours": str(max_hours)})
    return start, end, Cursor(start=start, end=end, cutoffs={})


def _parse_datetime(value: object) -> datetime:
    """Return a UTC-aware datetime or raise the production invalid-input key."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise RecoverableToolError("invalid_tool_input", {"error": "expected an ISO datetime"})


def _window(start: datetime, end: datetime) -> dict[str, str]:
    """Build the production recorder window envelope."""
    return {"start": start.isoformat(), "end": end.isoformat()}


def _windowed_rows(
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
    timestamp: Callable[[Mapping[str, object]], datetime],
) -> list[dict[str, object]]:
    """Keep fixture rows whose timestamp falls inside the computed inclusive window."""
    return [row for row in rows if start <= timestamp(row) <= end]


def _history_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the UTC timestamp of a fixture history row."""
    return _parse_datetime(row.get("last_changed") or row.get("last_updated"))


def _statistics_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the UTC timestamp of a fixture statistics row."""
    value = row.get("start") or row.get("end") or row.get("last_reset")
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC)
    return _parse_datetime(value)


def _logbook_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the UTC timestamp of a fixture logbook row."""
    return _parse_datetime(row.get("when"))


def _logbook_entry_timestamp(row: Mapping[str, object]) -> str:
    """Return one logbook entry's ISO timestamp for cursor pagination."""
    return str(row["when"])


def _requested_attributes(value: object) -> list[str] | None:
    """Normalize and bound history attribute opt-in names."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value][:MAX_HISTORY_ATTRIBUTES]
    if isinstance(value, list | tuple):
        return [str(item) for item in value][:MAX_HISTORY_ATTRIBUTES]
    return [str(value)][:MAX_HISTORY_ATTRIBUTES]


def _history_entity_payload(rows: list[tuple[list[object], str | None]]) -> dict[str, object]:
    """Build the de-duplicated production history payload for one entity."""
    entity: dict[str, object] = {"rows": [row for row, _unit in rows]}
    unit = next((unit for _row, unit in rows if unit), None)
    if unit is not None:
        entity["unit"] = unit
    return entity


def _history_row(
    row: Mapping[str, object], requested_attributes: list[str] | None = None
) -> tuple[list[object], str | None]:
    """Convert a fixture history row to ``([time, state, attrs?], unit)``."""
    timestamp = row.get("last_changed") or row.get("last_updated")
    attributes = row.get("attributes")
    unit = None
    if isinstance(attributes, Mapping):
        unit = attributes.get("unit_of_measurement") or attributes.get("unit")
    shaped: list[object] = [str(timestamp), row.get("state")]
    if requested_attributes is not None:
        present: dict[str, object] = {}
        if isinstance(attributes, Mapping):
            present = {name: attributes[name] for name in requested_attributes if name in attributes}
        shaped.append(present)
    return shaped, str(unit) if unit is not None else None


def _statistics_payload(rows: list[list[object]]) -> dict[str, object]:
    """Build the production statistics payload for one statistic id."""
    fields = sorted({str(key) for row in rows if isinstance(row[1], Mapping) for key in row[1]})
    return {"rows": rows, "fields": fields}


def _statistics_row(row: Mapping[str, object], requested_types: tuple[str, ...] | None) -> list[object]:
    """Convert a fixture statistics row to the production ``[time, {field: value}]`` array."""
    timestamp = row.get("start") or row.get("end") or row.get("last_reset")
    value_keys = requested_types or _DEFAULT_STATISTIC_VALUE_PRIORITY
    values: dict[str, object] = {}
    for key in value_keys:
        if key not in STATISTIC_VALUE_TYPES:
            continue
        value = row.get(key)
        if value is None:
            continue
        values[key] = value
        if requested_types is None:
            break
    return [str(timestamp), values]


def _logbook_entry(entity_id: str, row: Mapping[str, object]) -> dict[str, object]:
    """Build one flat logbook entry with the scoped entity id retained."""
    entry = dict(row)
    entry["entity_id"] = entity_id
    return entry


def _dict_list(value: object) -> list[dict[str, object]]:
    """Coerce a JSON-like list of dicts into the ToolOutcome action tuple shape."""
    # Branch boundary: unexpected executor action payloads are ignored rather than re-shaped unsafely.
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]
