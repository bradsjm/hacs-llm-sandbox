"""Fixture-backed runtime assembly for production-core eval execution."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from types import ModuleType
from typing import cast

from custom_components.llm_sandbox.const import DEFAULT_SERVICE_CALL_LIMIT
from custom_components.llm_sandbox.llm_api.data.energy import EnergyPeriod, _floor_bucket, sanitize_energy_preferences
from custom_components.llm_sandbox.llm_api.data.history import HistoryRow, flat_history_rows
from custom_components.llm_sandbox.llm_api.data.numeric import finite_float
from custom_components.llm_sandbox.llm_api.errors import RecoverableToolError
from custom_components.llm_sandbox.llm_api.executor_support import ExecutionState, json_safe
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from custom_components.llm_sandbox.llm_api.resolution_memory import ResolutionMemory
from custom_components.llm_sandbox.llm_api.sandbox_context import RuntimeContext
from custom_components.llm_sandbox.llm_api.tools.automation import (
    AutomationRecord,
    AutomationSource,
    GetAutomationTool,
)
from custom_components.llm_sandbox.llm_api.tools.code import ExecuteHomeCodeTool
from custom_components.llm_sandbox.llm_api.tools.energy import (
    EnergyQuerySource,
    GetEnergyTool,
    _sanitize_validation,
    _validation_private_locators,
    _validation_public_locators,
    run_energy_query,
)
from custom_components.llm_sandbox.llm_api.tools.recorder import (
    GetHistoryTool,
    GetLogbookTool,
    GetStatisticsTool,
    RecorderSource,
)
from custom_components.llm_sandbox.runtime import SandboxSettings
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from llm_sandbox_evals.schema import EvalCase, PromptCandidate
from llm_sandbox_evals.tools import EVAL_SCOPE, RecordingInvoker

type ToolBoundaryCallback = Callable[[str, bool], None]


@dataclass(frozen=True, slots=True)
class EvalRuntime:
    """Per-case frozen assembly of everything the agent + tools need."""

    case: EvalCase
    candidate: PromptCandidate
    snapshot: HomeSnapshot
    settings: SandboxSettings
    recorder_source: RecorderSource
    invoker: RecordingInvoker
    energy_source: EnergyQuerySource | None
    runtime_context_factory: Callable[[], RuntimeContext]
    code_tool: ExecuteHomeCodeTool
    recorder_tools: tuple[GetHistoryTool | GetStatisticsTool | GetLogbookTool, ...]
    automation_tool: GetAutomationTool
    energy_tool: GetEnergyTool
    automation_source: AutomationSource
    entry_id: str
    on_tool_boundary: ToolBoundaryCallback | None = None


def build_eval_runtime(
    case: EvalCase,
    candidate: PromptCandidate,
    profile: PromptProfile,
    snapshot: HomeSnapshot,
    fixture: ModuleType,
    *,
    on_tool_boundary: ToolBoundaryCallback | None = None,
) -> EvalRuntime:
    """Build one production-core runtime over a fresh scoped fixture snapshot."""
    invoker = RecordingInvoker()
    settings = SandboxSettings(
        execution_timeout_seconds=10,
        service_call_limit=DEFAULT_SERVICE_CALL_LIMIT,
        scope=EVAL_SCOPE,
        actions_enabled=True,
        action_domains=frozenset(),
        prompt_profile=profile,
    )
    entry_id = "eval"
    return EvalRuntime(
        case=case,
        candidate=candidate,
        snapshot=snapshot,
        settings=settings,
        recorder_source=build_fixture_recorder_source(snapshot, fixture),
        energy_source=build_fixture_energy_source(snapshot, fixture),
        automation_tool=GetAutomationTool(entry_id),
        automation_source=build_fixture_automation_source(snapshot, fixture),
        energy_tool=GetEnergyTool(entry_id),
        invoker=invoker,
        runtime_context_factory=_eval_runtime_context_factory(snapshot, settings, invoker, fixture),
        code_tool=ExecuteHomeCodeTool(entry_id),
        recorder_tools=(GetHistoryTool(entry_id), GetStatisticsTool(entry_id), GetLogbookTool(entry_id)),
        entry_id=entry_id,
        on_tool_boundary=on_tool_boundary,
    )


def build_fixture_recorder_source(snapshot: HomeSnapshot, fixture: ModuleType) -> RecorderSource:
    """Build the production RecorderSource over deterministic fixture recorder rows."""
    recorder = _recorder_data(fixture)
    history = _recorder_section(recorder, "history")
    statistics = _recorder_section(recorder, "statistics")
    logbook = _recorder_section(recorder, "logbook") if isinstance(recorder.get("logbook"), dict) else {}

    async def fetch_history(entity_ids: list[str], start: datetime, end: datetime) -> dict[str, list[HistoryRow]]:
        return {
            entity_id: cast(
                list[HistoryRow],
                _windowed_history_rows(history.get(entity_id, []), start, end, _history_timestamp),
            )
            for entity_id in entity_ids
        }

    async def fetch_statistics(
        statistic_ids: list[str],
        start: datetime,
        end: datetime,
        period: str,
        types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        _ = (period, types)
        return {
            statistic_id: _windowed_statistics_rows(
                statistics.get(statistic_id, []), start, end, _statistics_timestamp
            )
            for statistic_id in statistic_ids
        }

    async def fetch_logbook(entity_ids: list[str], start: datetime, end: datetime) -> list[dict[str, object]]:
        return await _eval_fetch_logbook(fixture, entity_ids, start, end)

    return RecorderSource(
        now=_parse_datetime(snapshot.created_at),
        logbook_available=bool(logbook),
        run_in_executor=_eval_run_blocking,
        fetch_history=fetch_history,
        fetch_statistics=fetch_statistics,
        fetch_logbook=fetch_logbook,
    )


def build_fixture_energy_source(snapshot: HomeSnapshot, fixture: ModuleType) -> EnergyQuerySource | None:
    """Build the production Energy core over copied deterministic fixture data."""
    if not hasattr(fixture, "energy"):
        return None
    raw = cast(Callable[[], dict[str, object]], fixture.energy)()
    expected = {"preferences", "cost_sensors", "metadata", "forecast", "validation"}
    if set(raw) != expected:
        raise ValueError(f"energy fixture keys must be exactly {sorted(expected)}")
    preferences = cast(Mapping[str, object], raw["preferences"])
    cost_sensors = cast(Mapping[str, str], raw["cost_sensors"])
    validation_price_locators: dict[int, dict[str, dict[str, str]]] = {}
    catalog, forecast_entries = sanitize_energy_preferences(
        snapshot,
        preferences,
        cost_sensors,
        validation_price_locators=validation_price_locators,
    )
    validation_data = cast(Mapping[str, object], raw["validation"])
    metadata = cast(dict[str, dict[str, object]], raw["metadata"])
    forecast_data = cast(dict[str, dict[str, object]], raw["forecast"])
    recorder_statistics = _recorder_section(_recorder_data(fixture), "statistics")

    async def fetch_metadata(statistic_ids: set[str]) -> dict[str, dict[str, object]]:
        return {statistic_id: dict(metadata[statistic_id]) for statistic_id in statistic_ids if statistic_id in metadata}

    async def fetch_statistics(
        statistic_ids: set[str],
        start: datetime,
        end: datetime,
        period: EnergyPeriod,
        units: dict[str, str] | None,
        types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {}
        for statistic_id in statistic_ids:
            rows = _windowed_statistics_rows(
                recorder_statistics.get(statistic_id, []), start, end, _statistics_timestamp
            )
            # Branch boundary: daily fixture rows already have production-shaped buckets.
            if period == "day":
                selected = rows
            else:
                selected = _aggregate_fixture_statistics(
                    rows, period, snapshot.config.time_zone
                )
            result[statistic_id] = []
            for row in selected:
                copied = {
                    key: value
                    for key, value in row.items()
                    if key == "start" or key == "last_reset" or key in types
                }
                if (
                    units is not None
                    and units.get("power") == "kW"
                    and metadata.get(statistic_id, {}).get("unit_class") == "power"
                    and (mean := copied.get("mean")) is not None
                ):
                    copied["mean"] = float(cast(float, mean)) / 1000
                result[statistic_id].append(copied)
        return result

    async def fetch_forecasts(source_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for source_id in source_ids:
            points: dict[str, float] = {}
            for config_entry_id in forecast_entries.get(source_id, ()):
                forecast = forecast_data.get(config_entry_id)
                wh_hours = forecast.get("wh_hours") if forecast is not None else None
                if not isinstance(wh_hours, Mapping):
                    continue
                for timestamp, raw_value in wh_hours.items():
                    if isinstance(timestamp, str) and (value := finite_float(raw_value)) is not None:
                        points[timestamp] = points.get(timestamp, 0.0) + value
            if points:
                result[source_id] = {"points": [[timestamp, value] for timestamp, value in sorted(points.items())]}
        return result

    public = _validation_public_locators(catalog)
    private = _validation_private_locators(catalog, forecast_entries)

    async def fetch_validation() -> tuple[dict[str, object], ...]:
        return _sanitize_validation(
            validation_data,
            public,
            private,
            validation_price_locators,
        )

    return EnergyQuerySource(
        now=_parse_datetime(snapshot.created_at),
        catalog=catalog,
        time_zone=snapshot.config.time_zone,
        visible_entity_ids=frozenset(snapshot.states),
        fetch_metadata=fetch_metadata,
        fetch_statistics=fetch_statistics,
        fetch_forecasts=fetch_forecasts,
        fetch_validation=fetch_validation,
    )


def build_fixture_automation_source(snapshot: HomeSnapshot, fixture: ModuleType) -> AutomationSource:
    """Build the production automation source over copied fixture values."""
    if not hasattr(fixture, "automation"):

        async def unavailable(
            _entity_ids: list[str], _start: datetime, _end: datetime
        ) -> Mapping[str, list[dict[str, object]]]:
            raise RecoverableToolError("automation_unavailable", {})

        return AutomationSource(_parse_datetime(snapshot.created_at), False, True, (), unavailable)

    data = cast(Callable[[], dict[str, object]], fixture.automation)()
    raw_records = cast(list[dict[str, object]], data["records"])
    records = tuple(
        AutomationRecord(
            entity_id=str(raw_record["entity_id"]),
            summary=cast(
                dict[str, object],
                json_safe({key: value for key, value in raw_record.items() if key not in {"content", "search_terms"}}),
            ),
            search_terms=tuple(str(term) for term in cast(tuple[object, ...], raw_record["search_terms"])),
            content=cast(dict[str, object] | None, json_safe(raw_record.get("content")))
            if isinstance(raw_record.get("content"), Mapping)
            else None,
        )
        for raw_record in raw_records
    )
    raw_runs = cast(dict[str, list[dict[str, object]]], data.get("runs", {}))

    async def fetch_runs(
        entity_ids: list[str], start: datetime, end: datetime
    ) -> Mapping[str, list[dict[str, object]]]:
        return {
            entity_id: [dict(row) for row in raw_runs.get(entity_id, ()) if start <= _logbook_timestamp(row) <= end]
            for entity_id in entity_ids
        }

    return AutomationSource(_parse_datetime(snapshot.created_at), True, True, records, fetch_runs)


def _eval_runtime_context_factory(
    snapshot: HomeSnapshot,
    settings: SandboxSettings,
    invoker: RecordingInvoker,
    fixture: ModuleType,
) -> Callable[[], RuntimeContext]:
    """Return a factory because execute_home_code state is per tool invocation."""

    energy_source = build_fixture_energy_source(snapshot, fixture)

    async def fetch_energy(data: dict[str, object]) -> JsonObjectType:
        if energy_source is None:
            raise RecoverableToolError("energy_not_configured", {})
        return await run_energy_query(data, energy_source)

    def factory() -> RuntimeContext:
        # State mutation point: each execute_home_code call gets fresh service/action accounting.
        state = ExecutionState(service_call_limit=settings.service_call_limit)
        fixture_now = _parse_datetime(snapshot.created_at)
        return RuntimeContext(
            state=state,
            settings=settings,
            invoke=invoker,
            fetch_history=lambda entity_ids, start, end: _eval_fetch_history(
                fixture, snapshot, entity_ids, start, end
            ),
            fetch_statistics=lambda statistic_ids, start, end: _eval_fetch_statistics(
                fixture, statistic_ids, start, end
            ),
            # The facade receives only this fresh fixture seam, never recorder data
            # cached from a prior execute_home_code invocation.
            fetch_logbook=lambda entity_ids, start, end: _eval_fetch_logbook(fixture, entity_ids, start, end),
            run_blocking=_eval_run_blocking,
            _utcnow=lambda: fixture_now,
            deadline=time.monotonic() + settings.execution_timeout_seconds,
            fetch_energy=fetch_energy,
            memory=ResolutionMemory(),
        )

    return factory


async def _eval_run_blocking(fn: Callable[[], object]) -> object:
    """Run eval-only blocking seams off the loop like Home Assistant's executor job."""
    return await asyncio.to_thread(fn)


async def _eval_fetch_history(
    fixture: ModuleType,
    snapshot: HomeSnapshot,
    entity_ids: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Return fixture history as production flat rows for hass.history/query in evals."""
    history = _recorder_section(_recorder_data(fixture), "history")
    scoped = {
        entity_id: _windowed_history_rows(history.get(entity_id, []), start, end, _history_timestamp)
        for entity_id in entity_ids
    }
    return flat_history_rows(scoped, snapshot)


async def _eval_fetch_statistics(
    fixture: ModuleType,
    statistic_ids: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Return fixture statistics as flat rows for read-only eval SQL."""
    statistics = _recorder_section(_recorder_data(fixture), "statistics")
    rows: list[dict[str, object]] = []
    for statistic_id in statistic_ids:
        rows.extend(
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
            for row in _windowed_statistics_rows(statistics.get(statistic_id, []), start, end, _statistics_timestamp)
        )
    return rows


async def _eval_fetch_logbook(
    fixture: ModuleType,
    entity_ids: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    """Return chronological, entity-scoped fixture logbook entries for Monty."""
    logbook = _recorder_section(_recorder_data(fixture), "logbook")
    entries = [
        _logbook_entry(entity_id, row)
        for entity_id in entity_ids
        for row in _windowed_logbook_rows(logbook.get(entity_id, []), start, end, _logbook_timestamp)
    ]
    return sorted(entries, key=_logbook_timestamp)


def _recorder_data(fixture: ModuleType) -> dict[str, object]:
    recorder = cast(Callable[[], dict[str, object]], fixture.recorder)
    return recorder()


def _recorder_section(recorder: Mapping[str, object], section: str) -> dict[str, list[dict[str, object]]]:
    """Return one typed canned recorder section from a fixture recorder mapping."""
    return cast(dict[str, list[dict[str, object]]], recorder[section])


def _aggregate_fixture_statistics(
    rows: list[dict[str, object]], period: EnergyPeriod, time_zone: str
) -> list[dict[str, object]]:
    """Aggregate deterministic daily rows on production calendar boundaries."""
    buckets: dict[datetime, list[dict[str, object]]] = {}
    for row in rows:
        bucket_start = _floor_bucket(_statistics_timestamp(row), period, time_zone)
        # State mutation point: retain source order within each calendar bucket.
        buckets.setdefault(bucket_start, []).append(row)

    aggregated: list[dict[str, object]] = []
    for bucket_start, bucket_rows in sorted(buckets.items()):
        result: dict[str, object] = {"start": bucket_start.isoformat()}
        for field in ("change", "mean", "min", "max", "state", "sum"):
            values = [
                value
                for row in bucket_rows
                if (value := finite_float(row.get(field))) is not None
            ]
            # Branch boundary: recorder omits statistic fields absent from a bucket.
            if not values:
                continue
            if field == "change":
                result[field] = sum(values)
            elif field == "mean":
                result[field] = sum(values) / len(values)
            elif field == "min":
                result[field] = min(values)
            elif field == "max":
                result[field] = max(values)
            else:
                result[field] = values[-1]
        for row in reversed(bucket_rows):
            # Branch boundary: retain an explicit null reset just like recorder output.
            if "last_reset" in row:
                result["last_reset"] = row["last_reset"]
                break
        # State mutation point: emit one completed production-shaped calendar bucket.
        aggregated.append(result)
    return aggregated


def _windowed_statistics_rows(
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
    timestamp: Callable[[Mapping[str, object]], datetime],
) -> list[dict[str, object]]:
    """Mirror recorder statistics selection with a half-open time window."""
    return [row for row in rows if start <= timestamp(row) < end]


def _windowed_logbook_rows(
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
    timestamp: Callable[[Mapping[str, object]], datetime],
) -> list[dict[str, object]]:
    """Mirror logbook event selection with an open time window."""
    return [row for row in rows if start < timestamp(row) < end]


def _windowed_history_rows(
    rows: list[dict[str, object]],
    start: datetime,
    end: datetime,
    timestamp: Callable[[Mapping[str, object]], datetime],
) -> list[dict[str, object]]:
    """Mirror significant-state history with a retained start-time baseline."""
    in_window = [row for row in rows if start < timestamp(row) < end]
    before = [row for row in rows if timestamp(row) < start]
    # Branch boundary: no earlier state exists to establish the window's starting value.
    if not before:
        return in_window
    baseline = max(before, key=timestamp)
    return [baseline, *in_window]


def _history_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the recorder selection timestamp of a fixture history row."""
    return _parse_datetime(row.get("last_updated") or row.get("last_changed"))


def _statistics_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the UTC timestamp of a fixture statistics row."""
    value = row.get("start") or row.get("end") or row.get("last_reset")
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC)
    return _parse_datetime(value)


def _logbook_timestamp(row: Mapping[str, object]) -> datetime:
    """Return the UTC timestamp of a fixture logbook row."""
    return _parse_datetime(row.get("when"))


def _logbook_entry(entity_id: str, row: Mapping[str, object]) -> dict[str, object]:
    """Build one flat logbook entry with the scoped entity id retained."""
    entry = dict(row)
    entry["entity_id"] = entity_id
    return entry


def _parse_datetime(value: object) -> datetime:
    """Return a UTC-aware datetime for known-good fixture/schema values."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise ValueError("expected an ISO datetime")
