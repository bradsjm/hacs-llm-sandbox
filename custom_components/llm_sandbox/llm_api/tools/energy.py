"""First-class Home Assistant Energy query tool and hass-free query seam."""

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import functools
import time
from typing import Literal, cast, final, override

from homeassistant.components.energy import data as energy_data
from homeassistant.components.energy import validate as energy_validate
from homeassistant.components.energy import websocket_api as energy_websocket
from homeassistant.components.energy.const import DOMAIN as ENERGY_DOMAIN
from homeassistant.components.recorder import statistics as recorder_statistics
from homeassistant.core import HomeAssistant, valid_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm, selector
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
import voluptuous as vol

from ...const import (
    MAX_ENERGY_FORECAST_SOURCES,
    MAX_ENERGY_SOURCE_RECORDS,
    MAX_ENERGY_STATISTIC_IDS,
    MAX_RECORDER_PAGE_BYTES,
    TOOL_GET_ENERGY,
)
from ...snapshot import build_recorder_snapshot
from ...snapshot.models import HomeSnapshot
from ..data.energy import (
    EnergyInclude,
    EnergyPeriod,
    EnergyRole,
    EnergySourceType,
    EnergyValidationPriceLocators,
    EnergyWindow,
    SafeEnergyCatalog,
    SafeEnergyDeviceRecord,
    SafeEnergyOmission,
    SafeEnergySourceRecord,
    align_energy_window,
    choose_energy_period,
    comparison_energy_window,
    comparison_query_period,
    compute_energy_result,
    energy_bucket_count,
    enforce_energy_point_bounds,
    fit_energy_result,
    resolve_requested_energy_window,
    sanitize_energy_preferences,
)
from ..data.numeric import finite_float
from ..errors import RecoverableToolError, tool_error_envelope, tool_error_from_exception
from ..executor_support.state import ExecutionState
from ..prompts import build_get_energy_description
from ._recorder_runtime import _await_deadline, _run_query, recorder_available
from ._support import _bounded_list, _omit_empty_optional_args, _require_loaded_entry_error, _require_sandbox_runtime

type EnergyMetadataFetcher = Callable[[set[str]], Awaitable[dict[str, dict[str, object]]]]
type EnergyStatisticsFetcher = Callable[
    [set[str], datetime, datetime, EnergyPeriod, dict[str, str] | None, set[str]],
    Awaitable[Mapping[str, list[dict[str, object]]]],
]
type EnergyForecastFetcher = Callable[[tuple[str, ...]], Awaitable[dict[str, dict[str, object]]]]
type EnergyValidationFetcher = Callable[[], Awaitable[tuple[dict[str, object], ...]]]

_SOURCE_TYPES: tuple[EnergySourceType, ...] = (
    "grid",
    "solar",
    "battery",
    "gas",
    "water",
    "device",
    "device_water",
)
_INCLUDES: tuple[EnergyInclude, ...] = ("summary", "series", "current", "forecast", "carbon", "validation")
_PERIODS: tuple[str, ...] = ("auto", "5minute", "hour", "day", "week", "month", "year")
_NULL_KEYS = frozenset(
    {"hours", "start", "end", "period", "source_types", "device_statistic_ids", "include", "compare"}
)
_EMPTY_STRINGS = frozenset({"start", "end", "compare"})
_EMPTY_LISTS = frozenset({"source_types", "device_statistic_ids", "include"})


@dataclass(frozen=True, slots=True)
class EnergyQuerySource:
    """Host-owned safe Energy catalog and async copied-data fetchers."""

    now: datetime
    catalog: SafeEnergyCatalog
    time_zone: str
    visible_entity_ids: frozenset[str]
    fetch_metadata: EnergyMetadataFetcher
    fetch_statistics: EnergyStatisticsFetcher
    fetch_forecasts: EnergyForecastFetcher
    fetch_validation: EnergyValidationFetcher


def _iso_datetime(value: object) -> datetime:
    """Validate an ISO datetime and normalize it to UTC."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str) and (parsed := dt_util.parse_datetime(value)) is not None:
        return dt_util.as_utc(parsed)
    raise vol.Invalid("expected an ISO datetime")


def _canonical_values(value: object, allowed: tuple[str, ...], field: str) -> list[str]:
    """Validate and canonicalize one unique list in declared order."""
    if not isinstance(value, list):
        raise vol.Invalid(f"{field} must be a list")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise vol.Invalid(f"{field} contains an unsupported value")
    return [item for item in allowed if item in value]


def _canonical_source_types(value: object) -> list[str]:
    return _canonical_values(value, cast(tuple[str, ...], _SOURCE_TYPES), "source_types")


def _canonical_include(value: object) -> list[str]:
    values = _canonical_values(value, cast(tuple[str, ...], _INCLUDES), "include")
    if "series" in values and "summary" not in values:
        # A series is a projection of the same totals, never a separate section.
        values.insert(0, "summary")
    return values


ENERGY_QUERY_SCHEMA: vol.Schema = vol.Schema(
    {
        vol.Optional("hours"): vol.All(vol.Coerce(float), vol.Range(min=0)),
        vol.Optional("start"): vol.All(selector.DateTimeSelector(), _iso_datetime),
        vol.Optional("end"): vol.All(selector.DateTimeSelector(), _iso_datetime),
        vol.Optional("period", default="auto"): vol.In(_PERIODS),
        vol.Optional("source_types"): vol.All(
            cv.ensure_list,
            selector.SelectSelector(selector.SelectSelectorConfig(options=list(_SOURCE_TYPES), multiple=True)),
            _canonical_source_types,
            _bounded_list("source_types", min_items=1, max_items=len(_SOURCE_TYPES)),
        ),
        vol.Optional("device_statistic_ids"): vol.All(
            cv.ensure_list,
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor", multiple=True)),
            [str],
            _bounded_list("device_statistic_ids", min_items=1, max_items=20),
        ),
        vol.Optional("include", default=["summary", "current"]): vol.All(
            cv.ensure_list,
            selector.SelectSelector(selector.SelectSelectorConfig(options=list(_INCLUDES), multiple=True)),
            _canonical_include,
            _bounded_list("include", min_items=1, max_items=len(_INCLUDES)),
        ),
        vol.Optional("compare"): vol.In(("previous", "year_over_year")),
    }
)


def normalize_energy_args(args: Mapping[str, object]) -> dict[str, object]:
    """Normalize optional Energy values before shared schema validation."""
    return _omit_empty_optional_args(
        args,
        null_keys=_NULL_KEYS,
        empty_string_keys=_EMPTY_STRINGS,
        empty_list_keys=_EMPTY_LISTS,
    )


def validate_energy_args(args: Mapping[str, object]) -> dict[str, object]:
    """Apply the shared direct/eval/Monty Energy schema."""
    try:
        data = cast(dict[str, object], ENERGY_QUERY_SCHEMA(normalize_energy_args(args)))
        if "hours" in data and "start" in data:
            raise vol.Invalid("hours cannot be combined with start")
        if "compare" in data and "summary" not in cast(list[str], data["include"]):
            raise vol.Invalid("compare requires include=summary")
        return data
    except Exception as err:
        mapped = tool_error_from_exception(err)
        if mapped is None:
            raise
        raise RecoverableToolError(*mapped) from err


async def async_copy_energy_catalog(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    *,
    validation_price_locators: EnergyValidationPriceLocators | None = None,
) -> tuple[SafeEnergyCatalog, dict[str, tuple[str, ...]]] | None:
    """Copy and sanitize the configured Energy dashboard on the event loop."""
    if not isinstance(hass.data.get(ENERGY_DOMAIN), Mapping):
        return None
    manager = await energy_data.async_get_manager(hass)
    preferences = manager.data
    if preferences is None or preferences == manager.default_preferences():
        return None
    domain_data = cast(Mapping[str, object], hass.data[ENERGY_DOMAIN])
    raw_cost_sensors = domain_data.get("cost_sensors")
    cost_sensors = (
        {str(key): str(value) for key, value in raw_cost_sensors.items()}
        if isinstance(raw_cost_sensors, Mapping)
        else {}
    )
    # Detach the sanitizer from manager-owned mutable preference trees.
    copied_preferences = cast(dict[str, object], deepcopy(preferences))
    return sanitize_energy_preferences(
        snapshot,
        copied_preferences,
        cost_sensors,
        validation_price_locators=validation_price_locators,
    )


def energy_source_counts(catalog: SafeEnergyCatalog) -> dict[str, int]:
    """Return visible Energy source counts in stable source-type order."""
    counts: Counter[str] = Counter(source.source_type for source in catalog.sources)
    counts.update(device.source_type for device in catalog.devices)
    return {source_type: counts[source_type] for source_type in _SOURCE_TYPES if counts[source_type]}


async def async_build_energy_source(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    deadline: float,
    state: ExecutionState | None = None,
) -> EnergyQuerySource:
    """Build one host-only Energy source from a fresh visible snapshot."""
    if not isinstance(hass.data.get(ENERGY_DOMAIN), Mapping):
        raise RecoverableToolError("energy_unavailable", {})
    if not recorder_available(hass):
        raise RecoverableToolError("recorder_unavailable", {})
    validation_price_locators: EnergyValidationPriceLocators = {}
    copied = await async_copy_energy_catalog(hass, snapshot, validation_price_locators=validation_price_locators)
    if copied is None:
        raise RecoverableToolError("energy_not_configured", {})
    catalog, forecast_config_entries = copied

    async def fetch_metadata(statistic_ids: set[str]) -> dict[str, dict[str, object]]:
        rows = await _await_deadline(recorder_statistics.async_list_statistic_ids(hass, statistic_ids), deadline)
        return {
            cast(str, row["statistic_id"]): {
                key: deepcopy(row[key])
                for key in (
                    "display_unit_of_measurement",
                    "has_mean",
                    "has_sum",
                    "mean_type",
                    "name",
                    "source",
                    "unit_class",
                    "unit_of_measurement",
                )
                if key in row
            }
            for row in rows
            if isinstance(row.get("statistic_id"), str)
        }

    async def fetch_statistics(
        statistic_ids: set[str],
        start: datetime,
        end: datetime,
        period: EnergyPeriod,
        units: dict[str, str] | None,
        types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        query_end = end - timedelta(microseconds=1) if period in {"day", "week", "month", "year"} else end
        rows = await _run_query(
            hass,
            deadline,
            functools.partial(
                recorder_statistics.statistics_during_period,
                hass=hass,
                start_time=start,
                end_time=query_end,
                statistic_ids=statistic_ids,
                period=period,
                units=units,
                types=cast(set, types),
            ),
            sync=state is None or state.live_write_dispatched,
        )
        return copy_statistics_rows(cast(Mapping[str, list[Mapping[str, object]]], rows), types)

    async def fetch_forecasts(source_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        platforms = await _await_deadline(energy_websocket.async_get_energy_platforms(hass), deadline)
        result: dict[str, dict[str, object]] = {}
        for source_id in source_ids:
            points: dict[str, float] = {}
            for config_entry_id in forecast_config_entries.get(source_id, ()):
                config_entry = hass.config_entries.async_get_entry(config_entry_id)
                if config_entry is None or (platform := platforms.get(config_entry.domain)) is None:
                    continue
                forecast = await _await_deadline(platform(hass, config_entry_id), deadline)
                if not isinstance(forecast, Mapping) or not isinstance(forecast.get("wh_hours"), Mapping):
                    continue
                for timestamp, raw_value in cast(Mapping[object, object], forecast["wh_hours"]).items():
                    value = finite_float(raw_value)
                    if isinstance(timestamp, str) and value is not None:
                        points[timestamp] = points.get(timestamp, 0.0) + value
            if points:
                result[source_id] = {"points": [[timestamp, value] for timestamp, value in sorted(points.items())]}
        return result

    validation_public = _validation_public_locators(catalog)
    validation_private = _validation_private_locators(catalog, forecast_config_entries)

    async def fetch_validation() -> tuple[dict[str, object], ...]:
        validation = await _await_deadline(energy_validate.async_validate(hass), deadline)
        return _sanitize_validation(
            cast(dict[str, object], validation.as_dict()),
            validation_public,
            validation_private,
            validation_price_locators,
        )

    return EnergyQuerySource(
        now=dt_util.utcnow(),
        catalog=catalog,
        time_zone=snapshot.config.time_zone,
        visible_entity_ids=frozenset(snapshot.states),
        fetch_metadata=fetch_metadata,
        fetch_statistics=fetch_statistics,
        fetch_forecasts=fetch_forecasts,
        fetch_validation=fetch_validation,
    )


def copy_statistics_rows(
    rows: Mapping[str, list[Mapping[str, object]]], requested_types: set[str]
) -> dict[str, list[dict[str, object]]]:
    """Copy only requested finite statistic columns with UTC ISO timestamps."""
    copied: dict[str, list[dict[str, object]]] = {}
    for statistic_id, statistic_rows in rows.items():
        selected_rows: list[dict[str, object]] = []
        for row in statistic_rows:
            start = _copy_timestamp(row.get("start"))
            if start is None:
                continue
            selected: dict[str, object] = {"start": start}
            for field in requested_types:
                if (value := finite_float(row.get(field))) is not None:
                    selected[field] = value
            if "last_reset" in requested_types and (last_reset := _copy_timestamp(row.get("last_reset"))) is not None:
                selected["last_reset"] = last_reset
            selected_rows.append(selected)
        copied[str(statistic_id)] = selected_rows
    return copied


def _copy_timestamp(value: object) -> str | None:
    """Copy a datetime, POSIX timestamp, or ISO string as UTC ISO text."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value).isoformat()
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC).isoformat()
    if isinstance(value, str) and (parsed := dt_util.parse_datetime(value)) is not None:
        return dt_util.as_utc(parsed).isoformat()
    return None


def _validation_public_locators(catalog: SafeEnergyCatalog) -> dict[str, dict[str, str]]:
    """Index public cumulative IDs to their already-exposed locators."""
    locators = {
        measure.statistic_id: {"role": measure.role, "statistic_id": measure.statistic_id}
        for source in catalog.sources
        for measure in source.measures
    }
    locators.update(
        {
            device.statistic_id: {
                "role": "device_consumption" if device.source_type == "device" else "device_water_consumption",
                "statistic_id": device.statistic_id,
            }
            for device in catalog.devices
        }
    )
    return locators


def _validation_private_locators(
    catalog: SafeEnergyCatalog,
    forecast_entries: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, str]]:
    """Index private configured IDs to safe source/device locators."""
    locators: dict[str, dict[str, str]] = {}
    for source in catalog.sources:
        if source.rate_statistic_id is not None:
            locators[source.rate_statistic_id] = {"role": "rate", "source_id": source.source_id}
        if source.state_of_charge_statistic_id is not None:
            locators[source.state_of_charge_statistic_id] = {
                "role": "state_of_charge",
                "source_id": source.source_id,
            }
    for device in catalog.devices:
        if device.rate_statistic_id is not None:
            locators[device.rate_statistic_id] = {"role": "rate", "device_statistic_id": device.statistic_id}
    for source_id, config_entries in forecast_entries.items():
        for config_entry_id in config_entries:
            locators[config_entry_id] = {"role": "forecast", "source_id": source_id}
    return locators


def _sanitize_validation(
    raw: Mapping[str, object],
    public: Mapping[str, dict[str, str]],
    private: Mapping[str, dict[str, str]],
    price_locators: Mapping[int, Mapping[str, dict[str, str]]],
) -> tuple[dict[str, object], ...]:
    """Map HA validation details to safe public locators and discard raw IDs."""
    results: list[dict[str, object]] = []
    for section_name in ("energy_sources", "device_consumption", "device_consumption_water"):
        raw_groups = raw.get(section_name)
        if not isinstance(raw_groups, list):
            continue
        for group_index, issues in enumerate(raw_groups):
            if not isinstance(issues, list):
                continue
            group_prices = price_locators.get(group_index, {}) if section_name == "energy_sources" else {}
            for issue in issues:
                if not isinstance(issue, Mapping) or not isinstance(issue.get("type"), str):
                    continue
                seen: set[tuple[tuple[str, str], ...]] = set()
                affected: list[dict[str, str]] = []
                raw_affected = issue.get("affected_entities")
                if isinstance(raw_affected, list | tuple | set):
                    for item in raw_affected:
                        raw_id = item[0] if isinstance(item, list | tuple) and item else item
                        if (
                            isinstance(raw_id, str)
                            and (locator := public.get(raw_id) or private.get(raw_id) or group_prices.get(raw_id))
                            is not None
                        ):
                            copied_locator = dict(locator)
                            locator_key = tuple(sorted(copied_locator.items()))
                            if locator_key not in seen:
                                seen.add(locator_key)
                                affected.append(copied_locator)
                affected.sort(key=lambda locator: tuple(sorted(locator.items())))
                if affected:
                    results.append({"type": issue["type"], "affected": affected})
    return tuple(results)


def _filter_validation_scope(
    validation: tuple[dict[str, object], ...],
    selected_sources: tuple[SafeEnergySourceRecord, ...],
    selected_devices: tuple[SafeEnergyDeviceRecord, ...],
) -> tuple[dict[str, object], ...]:
    """Keep validation locators within the current public response scope."""
    public_statistic_ids = {measure.statistic_id for source in selected_sources for measure in source.measures} | {
        device.statistic_id for device in selected_devices
    }
    source_ids = {source.source_id for source in selected_sources}
    device_statistic_ids = {device.statistic_id for device in selected_devices}
    filtered: list[dict[str, object]] = []
    for entry in validation:
        issue_type = entry.get("type")
        affected = entry.get("affected")
        if not isinstance(issue_type, str) or not isinstance(affected, list):
            continue
        scoped_affected: list[dict[str, str]] = []
        for locator in affected:
            if not isinstance(locator, Mapping) or not isinstance(locator.get("role"), str):
                continue
            role = locator["role"]
            if isinstance(statistic_id := locator.get("statistic_id"), str) and statistic_id in public_statistic_ids:
                scoped_affected.append({"role": role, "statistic_id": statistic_id})
            elif isinstance(source_id := locator.get("source_id"), str) and source_id in source_ids:
                scoped_affected.append({"role": role, "source_id": source_id})
            elif (
                isinstance(device_statistic_id := locator.get("device_statistic_id"), str)
                and device_statistic_id in device_statistic_ids
            ):
                scoped_affected.append({"role": role, "device_statistic_id": device_statistic_id})
        if scoped_affected:
            filtered.append({"type": issue_type, "affected": scoped_affected})
    return tuple(filtered)


async def run_energy_query(data: dict[str, object], source: EnergyQuerySource) -> JsonObjectType:
    """Run the raising, hass-free Energy query core."""
    catalog = source.catalog
    if not catalog.sources and not catalog.devices:
        raise RecoverableToolError("no_visible_energy_sources", {})
    selected_source_types = set(cast(list[EnergySourceType], data.get("source_types", list(_SOURCE_TYPES))))
    include = tuple(cast(list[EnergyInclude], data["include"]))
    device_filter = _validate_device_filter(data, source, selected_source_types)
    selected_sources = tuple(item for item in catalog.sources if item.source_type in selected_source_types)
    selected_devices = tuple(
        item
        for item in catalog.devices
        if item.source_type in selected_source_types and (device_filter is None or item.statistic_id in device_filter)
    )
    retained_count = len(selected_sources) + len(selected_devices)
    if retained_count > MAX_ENERGY_SOURCE_RECORDS:
        raise RecoverableToolError(
            "energy_source_limit_exceeded",
            {"source_count": str(retained_count), "max_sources": str(MAX_ENERGY_SOURCE_RECORDS)},
        )
    if "validation" in include and catalog.configured_record_count > MAX_ENERGY_SOURCE_RECORDS:
        raise RecoverableToolError(
            "energy_source_limit_exceeded",
            {"source_count": str(catalog.configured_record_count), "max_sources": str(MAX_ENERGY_SOURCE_RECORDS)},
        )
    selected_solar = tuple(source.source_id for source in selected_sources if source.source_type == "solar")
    if "forecast" in include and len(selected_solar) > MAX_ENERGY_FORECAST_SOURCES:
        raise RecoverableToolError(
            "energy_source_limit_exceeded",
            {"source_count": str(len(selected_solar)), "max_sources": str(MAX_ENERGY_FORECAST_SOURCES)},
        )

    co2_statistic_id = catalog.co2_statistic_id
    carbon_grid_import_ids = {
        measure.statistic_id
        for source in selected_sources
        if source.source_type == "grid"
        for measure in source.measures
        if measure.role == "grid_import"
    }
    carbon_pair_available = "carbon" in include and co2_statistic_id is not None and bool(carbon_grid_import_ids)
    cumulative_roles, cumulative_ids, rate_ids = _query_statistic_ids(
        catalog,
        selected_sources,
        selected_devices,
        include,
        cast(str | None, data.get("compare")),
        carbon_pair_available=carbon_pair_available,
    )
    query_ids = cumulative_ids | rate_ids
    if carbon_pair_available and co2_statistic_id is not None:
        query_ids.add(co2_statistic_id)
    if len(query_ids) > MAX_ENERGY_STATISTIC_IDS:
        raise RecoverableToolError(
            "energy_query_too_large",
            {"statistic_count": str(len(query_ids)), "bucket_count": "1", "max_points": str(MAX_ENERGY_STATISTIC_IDS)},
        )

    requested_start, requested_end = resolve_requested_energy_window(
        source.now,
        cast(datetime | None, data.get("start")),
        cast(datetime | None, data.get("end")),
        cast(float | None, data.get("hours")),
    )
    recorder_needed = bool(query_ids)
    _, series_count = _point_estimates(
        selected_sources,
        selected_devices,
        include,
        carbon_pair_available=carbon_pair_available,
    )
    statistic_count = max(1, len(query_ids))
    raw_period = cast(str, data["period"])
    comparison_mode = cast(Literal["previous", "year_over_year"] | None, data.get("compare"))
    if raw_period == "auto":
        period, window = choose_energy_period(
            requested_start,
            requested_end,
            source.time_zone,
            statistic_count=statistic_count,
            returned_series_count=series_count,
            recorder_needed=recorder_needed,
            series_requested="series" in include,
            rate_selected=bool(rate_ids),
            comparison_mode=comparison_mode,
        )
    elif recorder_needed:
        period = cast(EnergyPeriod, raw_period)
        window = align_energy_window(requested_start, requested_end, period, source.time_zone)
    else:
        period = cast(EnergyPeriod, raw_period)
        window = EnergyWindow(requested_start, requested_end, requested_start, requested_end)

    comparison_window = (
        comparison_energy_window(window, period, comparison_mode, source.time_zone)
        if comparison_mode is not None
        else None
    )
    comparison_period = comparison_query_period(period, comparison_mode)

    if raw_period != "auto" and recorder_needed:
        bucket_count = max(
            energy_bucket_count(window.start, window.end, period, source.time_zone),
            energy_bucket_count(
                comparison_window.start,
                comparison_window.end,
                comparison_period,
                source.time_zone,
            )
            if comparison_window is not None
            else 0,
        )
        enforce_energy_point_bounds(
            bucket_count,
            statistic_count,
            series_count,
            "series" in include,
        )
    metadata = await source.fetch_metadata(query_ids) if query_ids else {}
    supported_cumulative = {
        statistic_id for statistic_id in cumulative_ids if metadata.get(statistic_id, {}).get("has_sum")
    }
    supported_rates = {statistic_id for statistic_id in rate_ids if metadata.get(statistic_id, {}).get("has_mean")}
    omissions = Counter(
        (omission.role, omission.reason) for omission in catalog.omissions for _ in range(omission.count)
    )
    for statistic_id in cumulative_ids - supported_cumulative:
        omissions[(cumulative_roles.get(statistic_id, "source"), "metadata_unavailable")] += 1
    omissions[("rate", "metadata_unavailable")] += len(rate_ids - supported_rates)
    effective_catalog = replace(
        catalog,
        omissions=tuple(
            SafeEnergyOmission(role, reason, count) for (role, reason), count in sorted(omissions.items()) if count
        ),
    )

    carbon_statistics_available = (
        carbon_pair_available
        and co2_statistic_id is not None
        and bool(carbon_grid_import_ids.intersection(supported_cumulative))
        and bool(metadata.get(co2_statistic_id, {}).get("has_mean"))
    )
    statistic_ids = supported_cumulative | supported_rates
    if carbon_statistics_available and co2_statistic_id is not None:
        statistic_ids.add(co2_statistic_id)
    statistics = (
        await source.fetch_statistics(
            statistic_ids,
            window.start,
            window.end,
            period,
            {"energy": "kWh", "power": "kW"},
            {"change", "mean"},
        )
        if statistic_ids
        else {}
    )
    comparison_statistics = (
        await source.fetch_statistics(
            supported_cumulative,
            comparison_window.start,
            comparison_window.end,
            comparison_period,
            {"energy": "kWh", "power": "kW"},
            {"change"},
        )
        if comparison_window is not None and supported_cumulative
        else None
    )
    forecasts = await source.fetch_forecasts(selected_solar) if "forecast" in include else None
    validation = (
        _filter_validation_scope(await source.fetch_validation(), selected_sources, selected_devices)
        if "validation" in include
        else ()
    )
    result = compute_energy_result(
        effective_catalog,
        statistics,
        metadata,
        window=window,
        period=period,
        include=include,
        selected_source_types=selected_source_types,
        selected_device_ids=device_filter,
        comparison_statistics=comparison_statistics,
        comparison_window=comparison_window,
        forecasts=forecasts,
        carbon_statistics=statistics,
        validation=validation,
    )
    if comparison_mode is not None and isinstance(result.get("comparison"), dict):
        cast(dict[str, object], result["comparison"])["mode"] = comparison_mode
    if "forecast" in include and forecasts is not None:
        missing_forecasts = len(selected_solar) - len(forecasts)
        if missing_forecasts:
            cast(list[dict[str, object]], result["omissions"]).append(
                {"role": "forecast", "reason": "metadata_unavailable", "count": missing_forecasts}
            )
    return fit_energy_result(result, limit=MAX_RECORDER_PAGE_BYTES)


def _validate_device_filter(
    data: Mapping[str, object],
    source: EnergyQuerySource,
    selected_source_types: set[EnergySourceType],
) -> set[str] | None:
    """Validate selected tracked devices against visibility and configuration."""
    raw_ids = cast(list[str] | None, data.get("device_statistic_ids"))
    if raw_ids is None:
        return None
    if not selected_source_types.intersection({"device", "device_water"}):
        raise RecoverableToolError(
            "invalid_tool_input", {"error": "device_statistic_ids require a device source type"}
        )
    configured = {device.statistic_id for device in source.catalog.devices}
    selected: set[str] = set()
    for statistic_id in raw_ids:
        if not valid_entity_id(statistic_id):
            raise RecoverableToolError("invalid_tool_input", {"error": "invalid device statistic entity ID"})
        if statistic_id not in source.visible_entity_ids:
            raise RecoverableToolError("entity_not_visible", {"entity_id": statistic_id})
        if statistic_id not in configured:
            raise RecoverableToolError(
                "invalid_tool_input", {"error": "device statistic is not configured for Energy"}
            )
        selected.add(statistic_id)
    return selected


def _query_statistic_ids(
    catalog: SafeEnergyCatalog,
    sources: tuple[SafeEnergySourceRecord, ...],
    devices: tuple[SafeEnergyDeviceRecord, ...],
    include: tuple[EnergyInclude, ...],
    compare: str | None,
    *,
    carbon_pair_available: bool,
) -> tuple[dict[str, EnergyRole], set[str], set[str]]:
    """Return cumulative role map plus cumulative and mean query IDs."""
    needs_full_cumulative = bool({"summary", "series"}.intersection(include) or compare)
    roles: dict[str, EnergyRole] = {}
    if needs_full_cumulative:
        roles.update((measure.statistic_id, measure.role) for source in sources for measure in source.measures)
        roles.update(
            (
                device.statistic_id,
                "device_consumption" if device.source_type == "device" else "device_water_consumption",
            )
            for device in devices
        )
        selected_parents = {device.statistic_id for device in devices}
        roles.update(
            (
                child.statistic_id,
                "device_consumption" if child.source_type == "device" else "device_water_consumption",
            )
            for child in catalog.devices
            if child.included_in_stat in selected_parents
        )
    elif carbon_pair_available:
        roles.update(
            (measure.statistic_id, measure.role)
            for source in sources
            if source.source_type == "grid"
            for measure in source.measures
            if measure.role == "grid_import"
        )
    rates = (
        {
            statistic_id
            for statistic_id in (
                *(source.rate_statistic_id for source in sources),
                *(device.rate_statistic_id for device in devices),
            )
            if statistic_id is not None
        }
        if "series" in include
        else set()
    )
    return roles, set(roles), rates


def _point_estimates(
    sources: tuple[SafeEnergySourceRecord, ...],
    devices: tuple[SafeEnergyDeviceRecord, ...],
    include: tuple[EnergyInclude, ...],
    *,
    carbon_pair_available: bool,
) -> tuple[int, int]:
    """Conservatively count every returned or derived statistic series."""
    measures = sum(len(source.measures) for source in sources)
    rates = sum(source.rate_statistic_id is not None for source in sources) + sum(
        device.rate_statistic_id is not None for device in devices
    )
    device_values = len(devices) * 2
    electricity = 13 if any(source.source_type in {"grid", "solar", "battery"} for source in sources) else 0
    unit_group_summaries = sum(
        measure.role in {"gas_consumption", "water_consumption", "cost", "compensation"}
        for source in sources
        for measure in source.measures
    )
    statistic_count = max(1, measures + rates + device_values + electricity)
    returned_series = (
        measures
        + rates
        + device_values
        + electricity
        + unit_group_summaries
        + ("carbon" in include and carbon_pair_available)
        if "series" in include
        else 0
    )
    return statistic_count, returned_series


@final
class GetEnergyTool(llm.Tool):
    """Return visibility-filtered Home Assistant Energy dashboard data."""

    name = TOOL_GET_ENERGY
    description = build_get_energy_description()
    parameters = ENERGY_QUERY_SCHEMA

    def __init__(self, entry_id: str) -> None:
        """Initialize the tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Validate, snapshot, build the safe source, and run the Energy query."""
        try:
            data = validate_energy_args(tool_input.tool_args)
        except RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)
        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            return tool_error_envelope(*setup_error)
        runtime = _require_sandbox_runtime(hass, self.entry_id)
        snapshot = build_recorder_snapshot(
            hass,
            scope=runtime.settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        deadline = time.monotonic() + runtime.settings.execution_timeout_seconds
        try:
            source = await async_build_energy_source(hass, snapshot, deadline)
        except RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)
        except Exception as err:  # noqa: BLE001 - source races use the stable setup/query envelope
            mapped = tool_error_from_exception(err)
            return tool_error_envelope(*(mapped or ("query_failed", {"error": type(err).__name__})))
        return await self.run_query(data, source)

    @staticmethod
    async def run_query(data: dict[str, object], source: EnergyQuerySource) -> JsonObjectType:
        """Envelope the shared raising Energy core for direct/eval callers."""
        try:
            return await run_energy_query(data, source)
        except RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)
        except Exception as err:  # noqa: BLE001 - direct query failures use a stable envelope
            mapped = tool_error_from_exception(err)
            return tool_error_envelope(*(mapped or ("query_failed", {"error": type(err).__name__})))
