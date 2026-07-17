"""Safe Home Assistant Energy records and deterministic result shaping."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from typing import Literal, cast
from zoneinfo import ZoneInfo

from homeassistant.const import UnitOfPower
from homeassistant.core import valid_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.json import JsonObjectType
from homeassistant.util.unit_conversion import PowerConverter, VolumeFlowRateConverter

from ...snapshot.models import HomeSnapshot, SafeState
from .numeric import finite_float

type EnergyPeriod = Literal["5minute", "hour", "day", "week", "month", "year"]
type EnergyInclude = Literal["summary", "series", "current", "forecast", "carbon", "validation"]
type EnergySourceType = Literal["grid", "solar", "battery", "gas", "water", "device", "device_water"]
type EnergyRole = Literal[
    "grid_import",
    "grid_export",
    "solar_generation",
    "battery_charge",
    "battery_discharge",
    "gas_consumption",
    "water_consumption",
    "device_consumption",
    "device_water_consumption",
    "cost",
    "compensation",
]
type EnergyOmissionRole = (
    EnergyRole
    | Literal[
        "rate",
        "state_of_charge",
        "current_price",
        "forecast",
        "carbon_signal",
        "validation",
        "source",
    ]
)
type EnergyOmissionReason = Literal["not_visible", "external_statistic", "metadata_unavailable", "ambiguous"]


@dataclass(frozen=True, slots=True)
class SafeEnergyCurrentPrice:
    """Sanitized current Energy price copied from configuration or a state."""

    value: float
    unit: str
    source: Literal["entity", "fixed"]


@dataclass(frozen=True, slots=True)
class SafeEnergyMeasureRef:
    """Visible cumulative statistic and its semantic Energy role."""

    role: EnergyRole
    statistic_id: str
    current_price: SafeEnergyCurrentPrice | None = None


@dataclass(frozen=True, slots=True)
class SafeEnergySourceRecord:
    """Visible, frozen Energy dashboard source."""

    source_id: str
    source_type: EnergySourceType
    name: str
    measures: tuple[SafeEnergyMeasureRef, ...]
    rate_statistic_id: str | None = None
    current_rate_value: float | None = None
    current_rate_unit: str | None = None
    state_of_charge_value: float | None = None
    state_of_charge_statistic_id: str | None = None


@dataclass(frozen=True, slots=True)
class SafeEnergyDeviceRecord:
    """Visible, frozen tracked-device Energy record."""

    source_type: Literal["device", "device_water"]
    name: str
    statistic_id: str
    included_in_stat: str | None = None
    rate_statistic_id: str | None = None
    current_rate_value: float | None = None
    current_rate_unit: str | None = None


@dataclass(frozen=True, slots=True)
class SafeEnergyOmission:
    """Count-only description of rejected Energy configuration."""

    role: EnergyOmissionRole
    reason: EnergyOmissionReason
    count: int


@dataclass(frozen=True, slots=True)
class SafeEnergyCatalog:
    """Complete sanitized Energy catalog for one fresh Home Assistant snapshot."""

    sources: tuple[SafeEnergySourceRecord, ...]
    devices: tuple[SafeEnergyDeviceRecord, ...]
    omissions: tuple[SafeEnergyOmission, ...]
    co2_statistic_id: str | None
    configured_record_count: int


_SOURCE_ORDER: tuple[EnergySourceType, ...] = (
    "grid",
    "solar",
    "battery",
    "gas",
    "water",
    "device",
    "device_water",
)
_SOURCE_MEASURES: dict[str, tuple[tuple[str, EnergyRole], ...]] = {
    "grid": (
        ("stat_energy_from", "grid_import"),
        ("stat_energy_to", "grid_export"),
        ("stat_cost", "cost"),
        ("stat_compensation", "compensation"),
    ),
    "solar": (("stat_energy_from", "solar_generation"),),
    "battery": (
        ("stat_energy_to", "battery_charge"),
        ("stat_energy_from", "battery_discharge"),
    ),
    "gas": (("stat_energy_from", "gas_consumption"), ("stat_cost", "cost")),
    "water": (("stat_energy_from", "water_consumption"), ("stat_cost", "cost")),
}
_PRIMARY_ROLE: dict[str, EnergyRole] = {
    "grid": "grid_import",
    "solar": "solar_generation",
    "battery": "battery_discharge",
    "gas": "gas_consumption",
    "water": "water_consumption",
}
_ELECTRICITY_CUMULATIVE_ROLES: frozenset[EnergyRole] = frozenset(
    {
        "grid_import",
        "grid_export",
        "solar_generation",
        "battery_charge",
        "battery_discharge",
    }
)

_PRICE_FIELDS: dict[str, tuple[tuple[EnergyRole, str, str], ...]] = {
    "grid": (
        ("grid_import", "entity_energy_price", "number_energy_price"),
        ("grid_export", "entity_energy_price_export", "number_energy_price_export"),
    ),
    "gas": (("gas_consumption", "entity_energy_price", "number_energy_price"),),
    "water": (("water_consumption", "entity_energy_price", "number_energy_price"),),
}


# Host-only correlation map for raw Energy validation price entities. The
# group index matches Home Assistant's energy_sources preference list.
type EnergyValidationPriceLocators = dict[int, dict[str, dict[str, str]]]


def sanitize_energy_preferences(
    snapshot: HomeSnapshot,
    preferences: Mapping[str, object],
    cost_sensors: Mapping[str, str],
    *,
    validation_price_locators: EnergyValidationPriceLocators | None = None,
) -> tuple[SafeEnergyCatalog, dict[str, tuple[str, ...]]]:
    """Copy raw Energy preferences into a visibility-filtered safe catalog."""
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]] = Counter()
    sources: list[SafeEnergySourceRecord] = []
    devices: list[SafeEnergyDeviceRecord] = []
    forecast_entries: dict[str, tuple[str, ...]] = {}
    raw_sources = _mapping_sequence(preferences.get("energy_sources"))
    raw_devices = _mapping_sequence(preferences.get("device_consumption"))
    raw_water_devices = _mapping_sequence(preferences.get("device_consumption_water"))
    configured_record_count = len(raw_sources) + len(raw_devices) + len(raw_water_devices)

    source_indexes: Counter[str] = Counter()
    for validation_index, raw_source in enumerate(raw_sources):
        raw_type = raw_source.get("type")
        if not isinstance(raw_type, str) or raw_type not in _SOURCE_MEASURES:
            # Unknown/new HA source types remain count-only until explicitly modeled.
            omissions[("source", "metadata_unavailable")] += 1
            continue
        source_type = cast(Literal["grid", "solar", "battery", "gas", "water"], raw_type)
        source_index = source_indexes[source_type]
        source_indexes[source_type] += 1
        source_id = f"{source_type}:{source_index}"
        prices, visible_price_entity_ids = _sanitize_prices(snapshot, raw_source, source_type, omissions)
        measures = _sanitize_source_measures(snapshot, raw_source, source_type, prices, cost_sensors, omissions)
        rate_id, rate_value, rate_unit = _sanitize_current_state_ref(
            snapshot, raw_source.get("stat_rate"), "rate", omissions
        )
        soc_id, soc_value, _soc_unit = _sanitize_current_state_ref(
            snapshot, raw_source.get("stat_soc"), "state_of_charge", omissions
        )
        if not measures and rate_id is None and soc_id is None:
            # A configured source with no visible data is omitted rather than
            # exposing its configured name, prices, or forecast association.
            continue
        primary_id = _primary_measure_id(measures, source_type)
        configured_name = raw_source.get("name")
        name = _safe_name(configured_name, snapshot.states.get(primary_id or ""), primary_id, source_type)
        sources.append(
            SafeEnergySourceRecord(
                source_id=source_id,
                source_type=source_type,
                name=name,
                measures=measures,
                rate_statistic_id=rate_id,
                current_rate_value=rate_value,
                current_rate_unit=rate_unit,
                state_of_charge_value=soc_value if soc_id is not None else None,
                state_of_charge_statistic_id=soc_id,
            )
        )
        if validation_price_locators is not None and visible_price_entity_ids:
            validation_price_locators[validation_index] = {
                entity_id: {"role": "current_price", "source_id": source_id} for entity_id in visible_price_entity_ids
            }
        if source_type == "solar":
            # Forecast config-entry IDs stay in this private host-only mapping.
            entries = raw_source.get("config_entry_solar_forecast")
            if isinstance(entries, list | tuple):
                forecast_entries[source_id] = tuple(entry for entry in entries if isinstance(entry, str))

    devices.extend(
        device
        for raw_device in raw_devices
        if (device := _sanitize_device(snapshot, raw_device, "device", omissions)) is not None
    )
    devices.extend(
        device
        for raw_device in raw_water_devices
        if (device := _sanitize_device(snapshot, raw_device, "device_water", omissions)) is not None
    )

    co2_statistic_id = _sanitize_co2_statistic(snapshot, omissions)
    catalog = SafeEnergyCatalog(
        sources=tuple(sources),
        devices=tuple(devices),
        omissions=tuple(
            SafeEnergyOmission(role=role, reason=reason, count=count)
            for (role, reason), count in sorted(omissions.items())
        ),
        co2_statistic_id=co2_statistic_id,
        configured_record_count=configured_record_count,
    )
    return catalog, forecast_entries


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    """Return only mapping records from a raw preference collection."""
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _sanitize_source_measures(
    snapshot: HomeSnapshot,
    source: Mapping[str, object],
    source_type: Literal["grid", "solar", "battery", "gas", "water"],
    prices: Mapping[EnergyRole, SafeEnergyCurrentPrice],
    cost_sensors: Mapping[str, str],
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> tuple[SafeEnergyMeasureRef, ...]:
    """Return visible cumulative measures for one configured source."""
    measures: list[SafeEnergyMeasureRef] = []
    for key, role in _SOURCE_MEASURES[source_type]:
        raw_id = source.get(key)
        if raw_id is None and role in ("cost", "compensation"):
            # HA-generated cost/compensation sensors are copied only through the host
            # mapping. Cost is keyed by the import flow (stat_energy_from);
            # compensation is keyed by the export flow (stat_energy_to).
            flow_key = "stat_energy_to" if role == "compensation" else "stat_energy_from"
            flow_id = source.get(flow_key)
            if isinstance(flow_id, str):
                raw_id = cost_sensors.get(flow_id)
        statistic_id = _visible_statistic_id(snapshot, raw_id, role, omissions)
        if statistic_id is not None:
            measures.append(SafeEnergyMeasureRef(role, statistic_id, prices.get(role)))
    return tuple(measures)


def _sanitize_prices(
    snapshot: HomeSnapshot,
    source: Mapping[str, object],
    source_type: str,
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> tuple[dict[EnergyRole, SafeEnergyCurrentPrice], frozenset[str]]:
    """Copy prices and retain only visible raw entity IDs for host validation."""
    result: dict[EnergyRole, SafeEnergyCurrentPrice] = {}
    visible_entity_ids: set[str] = set()
    for role, entity_key, number_key in _PRICE_FIELDS.get(source_type, ()):
        entity_id = source.get(entity_key)
        if entity_id is not None:
            if not isinstance(entity_id, str) or not valid_entity_id(entity_id):
                omissions[("current_price", "external_statistic")] += 1
            elif (state := snapshot.states.get(entity_id)) is None:
                omissions[("current_price", "not_visible")] += 1
            else:
                visible_entity_ids.add(entity_id)
                if (value := finite_float(state.state)) is None:
                    omissions[("current_price", "metadata_unavailable")] += 1
                else:
                    result[role] = SafeEnergyCurrentPrice(
                        value=value,
                        unit=_state_unit(state) or snapshot.config.currency,
                        source="entity",
                    )
            continue
        fixed = finite_float(source.get(number_key))
        if fixed is not None:
            primary_key = "stat_energy_to" if role == "grid_export" else "stat_energy_from"
            primary_state = snapshot.states.get(cast(str, source.get(primary_key)))
            # Grid energy is always kWh; gas/water use their visible consumption unit,
            # falling back to the configured volume unit when state metadata is absent.
            if source_type == "grid":
                denominator: str = "kWh"
            elif primary_state is not None and (state_unit := _state_unit(primary_state)) is not None:
                denominator = state_unit
            else:
                denominator = snapshot.config.units.volume_unit
            result[role] = SafeEnergyCurrentPrice(
                value=fixed,
                unit=f"{snapshot.config.currency}/{denominator}",
                source="fixed",
            )
    return result, frozenset(visible_entity_ids)


def _sanitize_device(
    snapshot: HomeSnapshot,
    raw_device: Mapping[str, object],
    source_type: Literal["device", "device_water"],
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> SafeEnergyDeviceRecord | None:
    """Copy one visible tracked-device record and visible parent link."""
    role: EnergyRole = "device_consumption" if source_type == "device" else "device_water_consumption"
    statistic_id = _visible_statistic_id(snapshot, raw_device.get("stat_consumption"), role, omissions)
    if statistic_id is None:
        return None
    parent = raw_device.get("included_in_stat")
    included_in_stat = _visible_statistic_id(snapshot, parent, role, omissions) if parent is not None else None
    rate_id, rate_value, rate_unit = _sanitize_current_state_ref(
        snapshot, raw_device.get("stat_rate"), "rate", omissions
    )
    return SafeEnergyDeviceRecord(
        source_type=source_type,
        name=_safe_name(raw_device.get("name"), snapshot.states.get(statistic_id), statistic_id, source_type),
        statistic_id=statistic_id,
        included_in_stat=included_in_stat,
        rate_statistic_id=rate_id,
        current_rate_value=rate_value,
        current_rate_unit=rate_unit,
    )


def _visible_statistic_id(
    snapshot: HomeSnapshot,
    raw_id: object,
    role: EnergyOmissionRole,
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> str | None:
    """Return an entity-backed visible statistic ID or record a count-only omission."""
    if raw_id is None:
        return None
    if not isinstance(raw_id, str) or not valid_entity_id(raw_id):
        omissions[(role, "external_statistic")] += 1
        return None
    if raw_id not in snapshot.states:
        omissions[(role, "not_visible")] += 1
        return None
    return raw_id


def _sanitize_current_state_ref(
    snapshot: HomeSnapshot,
    raw_id: object,
    role: Literal["rate", "state_of_charge"],
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> tuple[str | None, float | None, str | None]:
    """Copy a visible state-backed ID and its current finite numeric value."""
    statistic_id = _visible_statistic_id(snapshot, raw_id, role, omissions)
    if statistic_id is None:
        return None, None, None
    state = snapshot.states[statistic_id]
    return statistic_id, finite_float(state.state), _state_unit(state)


def _sanitize_co2_statistic(
    snapshot: HomeSnapshot,
    omissions: Counter[tuple[EnergyOmissionRole, EnergyOmissionReason]],
) -> str | None:
    """Select one visible CO2 Signal percentage state deterministically."""
    candidates = sorted(
        state.entity_id
        for state in snapshot.states.values()
        if state.platform == "co2signal" and state.attributes.get("unit_of_measurement") == "%"
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        omissions[("carbon_signal", "ambiguous")] += 1
    return None


def _primary_measure_id(measures: tuple[SafeEnergyMeasureRef, ...], source_type: str) -> str | None:
    """Return the preferred visible measure ID used for display naming."""
    role = _PRIMARY_ROLE[source_type]
    return next((measure.statistic_id for measure in measures if measure.role == role), None)


def _safe_name(configured: object, state: SafeState | None, statistic_id: str | None, fallback: str) -> str:
    """Choose a safe source name in deterministic preference order."""
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    if state is not None and state.name:
        return state.name
    return statistic_id or fallback.replace("_", " ").title()


def _state_unit(state: SafeState) -> str | None:
    """Return a copied state unit when present."""
    unit = state.attributes.get("unit_of_measurement")
    return unit if isinstance(unit, str) and unit else None


@dataclass(frozen=True, slots=True)
class EnergyWindow:
    """Requested and effective UTC window for one Energy query."""

    start: datetime
    end: datetime
    requested_start: datetime
    requested_end: datetime

    def as_dict(self) -> dict[str, object]:
        """Return the public window shape."""
        result: dict[str, object] = {"start": self.start.isoformat(), "end": self.end.isoformat()}
        if self.start != self.requested_start or self.end != self.requested_end:
            result["requested"] = {
                "start": self.requested_start.isoformat(),
                "end": self.requested_end.isoformat(),
            }
        return result


def resolve_requested_energy_window(
    now: datetime,
    start: datetime | None,
    end: datetime | None,
    hours: float | None,
) -> tuple[datetime, datetime]:
    """Resolve an unbounded Energy request window in UTC."""
    from ..errors import RecoverableToolError

    if start is not None and hours is not None:
        raise RecoverableToolError("invalid_tool_input", {"error": "hours cannot be combined with start"})
    requested_end = _as_utc(end or now)
    if start is not None:
        requested_start = _as_utc(start)
    elif hours is not None:
        requested_start = requested_end - timedelta(hours=hours)
    else:
        requested_start = requested_end - timedelta(hours=24)
    if requested_start > requested_end:
        raise RecoverableToolError("invalid_tool_input", {"error": "start after end"})
    return requested_start, requested_end


def align_energy_window(
    start: datetime,
    end: datetime,
    period: EnergyPeriod,
    time_zone: str,
) -> EnergyWindow:
    """Align a requested window to complete Energy statistic buckets."""
    requested_start = _as_utc(start)
    requested_end = _as_utc(end)
    effective_start = _floor_bucket(requested_start, period, time_zone)
    if requested_start == requested_end:
        # A point request still selects the complete bucket containing that point.
        effective_end = _move_bucket(effective_start, period, 1, time_zone)
    else:
        final_bucket = _floor_bucket(requested_end - timedelta(microseconds=1), period, time_zone)
        effective_end = _move_bucket(final_bucket, period, 1, time_zone)
    return EnergyWindow(effective_start, effective_end, requested_start, requested_end)


def energy_bucket_count(start: datetime, end: datetime, period: EnergyPeriod, time_zone: str) -> int:
    """Count aligned period buckets without relying on elapsed UTC duration."""
    if end <= start:
        return 0
    if period == "5minute":
        return int((end - start) / timedelta(minutes=5))
    if period == "hour":
        return int((end - start) / timedelta(hours=1))
    count = 0
    cursor = start
    while cursor < end:
        # Calendar movement is intentionally local so DST/month length cannot
        # make estimates disagree with Home Assistant's bucket boundaries.
        cursor = _move_bucket(cursor, period, 1, time_zone)
        count += 1
    return count


def comparison_query_period(
    period: EnergyPeriod,
    mode: Literal["previous", "year_over_year"] | None,
) -> EnergyPeriod:
    """Return the recorder period used for a comparison window."""
    if mode == "year_over_year" and period == "week":
        return "day"
    return period


def comparison_energy_window(
    window: EnergyWindow,
    period: EnergyPeriod,
    mode: Literal["previous", "year_over_year"],
    time_zone: str,
) -> EnergyWindow:
    """Return the period-shifted comparison window."""
    if mode == "year_over_year":
        start = _move_local_year(window.start, -1, time_zone)
        end = _move_local_year(window.end, -1, time_zone)
    else:
        bucket_count = energy_bucket_count(window.start, window.end, period, time_zone)
        start = _move_bucket(window.start, period, -bucket_count, time_zone)
        end = window.start
    return EnergyWindow(start, end, start, end)


def choose_energy_period(
    requested_start: datetime,
    requested_end: datetime,
    time_zone: str,
    *,
    statistic_count: int,
    returned_series_count: int,
    recorder_needed: bool,
    series_requested: bool,
    rate_selected: bool,
    comparison_mode: Literal["previous", "year_over_year"] | None = None,
) -> tuple[EnergyPeriod, EnergyWindow]:
    """Choose the first period satisfying both Energy point budgets."""
    from ...const import MAX_ENERGY_QUERY_POINTS
    from ..errors import RecoverableToolError

    if not recorder_needed:
        return "hour", EnergyWindow(requested_start, requested_end, requested_start, requested_end)
    candidates: tuple[EnergyPeriod, ...] = (
        ("5minute", "hour", "day", "week", "month", "year")
        if series_requested and rate_selected
        else ("hour", "day", "week", "month", "year")
    )
    last_bucket_count = 0
    last_error: RecoverableToolError | None = None
    for candidate in candidates:
        window = align_energy_window(requested_start, requested_end, candidate, time_zone)
        comparison_window = (
            comparison_energy_window(window, candidate, comparison_mode, time_zone)
            if comparison_mode is not None
            else None
        )
        bucket_count = max(
            energy_bucket_count(window.start, window.end, candidate, time_zone),
            energy_bucket_count(
                comparison_window.start,
                comparison_window.end,
                comparison_query_period(candidate, comparison_mode),
                time_zone,
            )
            if comparison_window is not None
            else 0,
        )
        last_bucket_count = bucket_count
        try:
            enforce_energy_point_bounds(
                bucket_count,
                statistic_count,
                returned_series_count,
                series_requested,
            )
        except RecoverableToolError as err:
            last_error = err
            continue
        return candidate, window
    # Year is the coarsest candidate. Its actual failing budget is the one
    # callers need to correct, including the returned-series ceiling.
    if last_error is not None:
        raise last_error
    raise RecoverableToolError(
        "energy_query_too_large",
        {
            "statistic_count": str(statistic_count),
            "bucket_count": str(last_bucket_count),
            "max_points": str(MAX_ENERGY_QUERY_POINTS),
        },
    )


def enforce_energy_point_bounds(
    bucket_count: int,
    statistic_count: int,
    returned_series_count: int,
    series_requested: bool,
) -> None:
    """Raise the stable point-budget error for the first violated bound."""
    from ...const import MAX_ENERGY_QUERY_POINTS, MAX_ENERGY_SERIES_POINTS
    from ..errors import RecoverableToolError

    if statistic_count * bucket_count > MAX_ENERGY_QUERY_POINTS:
        raise RecoverableToolError(
            "energy_query_too_large",
            {
                "statistic_count": str(statistic_count),
                "bucket_count": str(bucket_count),
                "max_points": str(MAX_ENERGY_QUERY_POINTS),
            },
        )
    if series_requested and returned_series_count * bucket_count > MAX_ENERGY_SERIES_POINTS:
        raise RecoverableToolError(
            "energy_query_too_large",
            {
                "statistic_count": str(returned_series_count),
                "bucket_count": str(bucket_count),
                "max_points": str(MAX_ENERGY_SERIES_POINTS),
            },
        )


def _floor_bucket(value: datetime, period: EnergyPeriod, time_zone: str) -> datetime:
    """Floor a UTC datetime to a Home Assistant statistic boundary."""
    value = _as_utc(value)
    if period == "5minute":
        return value.replace(minute=value.minute - value.minute % 5, second=0, microsecond=0)
    if period == "hour":
        return value.replace(minute=0, second=0, microsecond=0)
    local = value.astimezone(ZoneInfo(time_zone))
    if period == "day":
        floored = local.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        floored = (local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        floored = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        floored = local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return floored.astimezone(UTC)


def _move_bucket(value: datetime, period: EnergyPeriod, count: int, time_zone: str) -> datetime:
    """Move an aligned bucket boundary by a signed bucket count."""
    if period == "5minute":
        return value + timedelta(minutes=5 * count)
    if period == "hour":
        return value + timedelta(hours=count)
    local = value.astimezone(ZoneInfo(time_zone))
    if period == "day":
        moved = local + timedelta(days=count)
    elif period == "week":
        moved = local + timedelta(weeks=count)
    elif period == "month":
        moved = _move_local_month(local, count)
    else:
        moved = local.replace(year=local.year + count)
    return moved.astimezone(UTC)


def _move_local_month(value: datetime, count: int) -> datetime:
    """Move a local calendar boundary by signed months."""
    month_index = value.year * 12 + value.month - 1 + count
    year, zero_based_month = divmod(month_index, 12)
    return value.replace(year=year, month=zero_based_month + 1)


def _move_local_year(value: datetime, count: int, time_zone: str) -> datetime:
    """Move a UTC boundary by local years, clamping February 29."""
    local = value.astimezone(ZoneInfo(time_zone))
    target_year = local.year + count
    try:
        moved = local.replace(year=target_year)
    except ValueError:
        moved = local.replace(year=target_year, day=28)
    return moved.astimezone(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def compute_energy_result(
    catalog: SafeEnergyCatalog,
    statistics: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    window: EnergyWindow,
    period: EnergyPeriod,
    include: tuple[EnergyInclude, ...],
    selected_source_types: set[EnergySourceType],
    selected_device_ids: set[str] | None,
    comparison_statistics: Mapping[str, list[dict[str, object]]] | None = None,
    comparison_window: EnergyWindow | None = None,
    forecasts: Mapping[str, Mapping[str, object]] | None = None,
    carbon_statistics: Mapping[str, list[dict[str, object]]] | None = None,
    validation: tuple[dict[str, object], ...] = (),
) -> JsonObjectType:
    """Build the exact bounded public Energy result from copied host data."""
    include_set: set[str] = set(include)
    selected_sources = tuple(source for source in catalog.sources if source.source_type in selected_source_types)
    selected_devices = tuple(
        device
        for device in catalog.devices
        if device.source_type in selected_source_types
        and (selected_device_ids is None or device.statistic_id in selected_device_ids)
    )
    source_outputs, source_bucket_values, unavailable_electricity_roles = _source_results(
        selected_sources, statistics, metadata, include_set
    )
    device_outputs, _device_bucket_values = _device_results(
        selected_devices, catalog.devices, statistics, metadata, include_set
    )
    filtered = selected_source_types != set(_SOURCE_ORDER) or selected_device_ids is not None
    incomplete_roles = {
        "grid_import",
        "grid_export",
        "solar_generation",
        "battery_charge",
        "battery_discharge",
        "gas_consumption",
        "water_consumption",
        "device_consumption",
        "device_water_consumption",
        "cost",
        "compensation",
        "source",
    }
    payload: dict[str, object] = {
        "window": window.as_dict(),
        "period": period,
        "scope": {
            "source_types": [source_type for source_type in _SOURCE_ORDER if source_type in selected_source_types],
            "device_statistic_ids": [device.statistic_id for device in selected_devices],
            "complete_dashboard": not filtered
            and not any(omission.role in incomplete_roles for omission in catalog.omissions),
        },
        "sources": source_outputs,
        "devices": device_outputs,
        "omissions": [
            {"role": omission.role, "reason": omission.reason, "count": omission.count}
            for omission in catalog.omissions
        ],
    }
    if "summary" in include_set:
        payload["summary"] = _energy_summary(
            source_bucket_values,
            unavailable_electricity_roles=unavailable_electricity_roles,
            include_series="series" in include_set,
        )
    if comparison_statistics is not None and comparison_window is not None:
        (
            comparison_sources,
            comparison_source_values,
            comparison_unavailable_electricity_roles,
        ) = _source_results(selected_sources, comparison_statistics, metadata, {"summary"})
        comparison_devices, _comparison_device_values = _device_results(
            selected_devices, catalog.devices, comparison_statistics, metadata, {"summary"}
        )
        payload["comparison"] = {
            "window": comparison_window.as_dict(),
            "summary": _energy_summary(
                comparison_source_values,
                unavailable_electricity_roles=comparison_unavailable_electricity_roles,
                include_series=False,
            ),
            "source_totals": [
                {
                    "source_id": source["source_id"],
                    "source_type": source["source_type"],
                    "measures": [
                        {key: value for key, value in measure.items() if key in {"role", "value", "unit"}}
                        for measure in cast(list[dict[str, object]], source.get("measures", []))
                    ],
                }
                for source in comparison_sources
            ],
            "device_totals": [
                {
                    key: value
                    for key, value in device.items()
                    if key
                    in {
                        "source_type",
                        "statistic_id",
                        "inclusive_value",
                        "exclusive_value",
                        "unit",
                    }
                }
                for device in comparison_devices
            ],
        }
    if "forecast" in include_set:
        payload["forecast"] = _forecast_results(selected_sources, forecasts or {})
    if "carbon" in include_set:
        payload["carbon"] = _carbon_result(
            catalog,
            selected_sources,
            source_bucket_values,
            metadata,
            carbon_statistics or {},
            include_series="series" in include_set,
        )
    if "validation" in include_set:
        payload["validation"] = list(validation)
    return cast(JsonObjectType, payload)


def _source_results(
    sources: tuple[SafeEnergySourceRecord, ...],
    statistics: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, Mapping[str, object]],
    include: set[str],
) -> tuple[
    list[dict[str, object]],
    dict[EnergyRole, dict[str, dict[str, float]]],
    set[EnergyRole],
]:
    """Build source rows and role-indexed per-bucket cumulative values."""
    outputs: list[dict[str, object]] = []
    unavailable_electricity_roles: set[EnergyRole] = set()
    electricity_roles = {
        "grid_import",
        "grid_export",
        "solar_generation",
        "battery_charge",
        "battery_discharge",
    }
    all_values: dict[EnergyRole, dict[str, dict[str, float]]] = {}
    for source in sources:
        output: dict[str, object] = {
            "source_id": source.source_id,
            "source_type": source.source_type,
            "name": source.name,
        }
        measures: list[dict[str, object]] = []
        for measure in source.measures:
            unit = _statistic_unit(metadata.get(measure.statistic_id))
            supported = bool(metadata.get(measure.statistic_id, {}).get("has_sum"))
            rows = statistics.get(measure.statistic_id, [])
            is_supported_electricity_measure = measure.role not in electricity_roles or unit == "kWh"
            values = (
                _row_values(rows, "change")
                if supported and unit is not None and is_supported_electricity_measure
                else {}
            )
            has_cumulative_result = (
                supported and unit is not None and is_supported_electricity_measure and (bool(values) or not rows)
            )
            if measure.role in electricity_roles and (not has_cumulative_result or unit != "kWh"):
                unavailable_electricity_roles.add(measure.role)
            if has_cumulative_result and unit is not None:
                unit_values = all_values.setdefault(measure.role, {}).setdefault(unit, {})
                for start, value in values.items():
                    unit_values[start] = unit_values.get(start, 0.0) + value
            measure_output: dict[str, object] = {
                "role": measure.role,
                "statistic_id": measure.statistic_id,
            }
            if has_cumulative_result and unit is not None and ("summary" in include or "series" in include):
                measure_output["value"] = sum(values.values())
                measure_output["unit"] = unit
                if "series" in include:
                    measure_output["series"] = [[start, value] for start, value in sorted(values.items())]
            if "current" in include and measure.current_price is not None:
                measure_output["current_price"] = {
                    "value": measure.current_price.value,
                    "unit": measure.current_price.unit,
                    "source": measure.current_price.source,
                }
            if len(measure_output) > 2:
                measures.append(measure_output)
        if measures:
            output["measures"] = measures
        if "current" in include and source.current_rate_value is not None and source.current_rate_unit is not None:
            current_rate = _normalize_rate_value(source.current_rate_value, source.current_rate_unit)
            if current_rate is not None:
                value, rate_unit = current_rate
                output["current_rate"] = {"value": value, "unit": rate_unit}
        if "series" in include and source.rate_statistic_id is not None:
            rate = _rate_series(source.rate_statistic_id, statistics, metadata)
            if rate is not None:
                output["rate_series"] = rate
        if "current" in include and source.state_of_charge_value is not None:
            output["state_of_charge"] = {"value": source.state_of_charge_value, "unit": "%"}
        outputs.append(output)
    return outputs, all_values, unavailable_electricity_roles


def _device_results(
    selected: tuple[SafeEnergyDeviceRecord, ...],
    all_devices: tuple[SafeEnergyDeviceRecord, ...],
    statistics: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, Mapping[str, object]],
    include: set[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, tuple[float, float, str]]]]:
    """Build tracked-device inclusive/exclusive totals and series."""
    outputs: list[dict[str, object]] = []
    bucket_values: dict[str, dict[str, tuple[float, float, str]]] = {}
    children: dict[str, list[str]] = {}
    for device in all_devices:
        if device.included_in_stat is not None:
            children.setdefault(device.included_in_stat, []).append(device.statistic_id)
    for device in selected:
        unit = _statistic_unit(metadata.get(device.statistic_id))
        supported = bool(metadata.get(device.statistic_id, {}).get("has_sum"))
        rows = statistics.get(device.statistic_id, [])
        inclusive = _row_values(rows, "change") if supported else {}
        child_values = {
            child_id: _row_values(statistics.get(child_id, []), "change")
            for child_id in children.get(device.statistic_id, [])
            if _statistic_unit(metadata.get(child_id)) == unit and bool(metadata.get(child_id, {}).get("has_sum"))
        }
        values: dict[str, tuple[float, float, str]] = {}
        if unit is not None and supported:
            for start, inclusive_value in inclusive.items():
                exclusive = max(0.0, inclusive_value - sum(rows.get(start, 0.0) for rows in child_values.values()))
                values[start] = (inclusive_value, exclusive, unit)
        bucket_values[device.statistic_id] = values
        output: dict[str, object] = {
            "source_type": device.source_type,
            "name": device.name,
            "statistic_id": device.statistic_id,
        }
        if device.included_in_stat is not None:
            output["included_in_stat"] = device.included_in_stat
        has_cumulative_result = unit is not None and supported and (bool(values) or not rows)
        if has_cumulative_result and unit is not None and ("summary" in include or "series" in include):
            output["inclusive_value"] = sum(value[0] for value in values.values())
            output["exclusive_value"] = sum(value[1] for value in values.values())
            output["unit"] = unit
            if "series" in include:
                output["series"] = [
                    {"start": start, "inclusive_value": value[0], "exclusive_value": value[1]}
                    for start, value in sorted(values.items())
                ]
        if "current" in include and device.current_rate_value is not None and device.current_rate_unit is not None:
            current_rate = _normalize_rate_value(device.current_rate_value, device.current_rate_unit)
            if current_rate is not None:
                value, rate_unit = current_rate
                output["current_rate"] = {"value": value, "unit": rate_unit}
        if "series" in include and device.rate_statistic_id is not None:
            rate = _rate_series(device.rate_statistic_id, statistics, metadata)
            if rate is not None:
                output["rate_series"] = rate
        outputs.append(output)
    return outputs, bucket_values


def _energy_summary(
    values: Mapping[EnergyRole, Mapping[str, Mapping[str, float]]],
    *,
    unavailable_electricity_roles: set[EnergyRole],
    include_series: bool,
) -> dict[str, object]:
    """Aggregate source measures without crossing unit boundaries."""
    summary: dict[str, object] = {}
    electricity_roles: tuple[EnergyRole, ...] = (
        "grid_import",
        "grid_export",
        "solar_generation",
        "battery_charge",
        "battery_discharge",
    )
    available_electricity_roles = tuple(
        role for role in electricity_roles if role not in unavailable_electricity_roles
    )
    electricity_buckets: dict[str, dict[EnergyRole, float]] = {}
    for role in available_electricity_roles:
        for start, value in values.get(role, {}).get("kWh", {}).items():
            electricity_buckets.setdefault(start, {})[role] = value
    known_electricity_roles = tuple(role for role in available_electricity_roles if "kWh" in values.get(role, {}))
    if electricity_buckets or known_electricity_roles:
        quantity_roles: tuple[str, ...] = known_electricity_roles
        if electricity_buckets:
            quantity_roles = available_electricity_roles
        if not unavailable_electricity_roles and (
            electricity_buckets or len(known_electricity_roles) == len(electricity_roles)
        ):
            quantity_roles = (
                *electricity_roles,
                "home_consumption",
                "grid_to_battery",
                "battery_to_grid",
                "solar_to_battery",
                "solar_to_grid",
                "used_solar",
                "used_grid",
                "used_battery",
            )
        quantity_series: dict[str, list[list[object]]] = {role: [] for role in quantity_roles}
        for start, bucket in sorted(electricity_buckets.items()):
            for quantity_role in quantity_roles:
                if quantity_role in electricity_roles:
                    quantity_series[quantity_role].append([start, max(bucket.get(quantity_role, 0.0), 0.0)])
            if unavailable_electricity_roles:
                continue
            flows = _compute_electricity_flows(
                bucket.get("grid_import", 0.0),
                bucket.get("grid_export", 0.0),
                bucket.get("solar_generation", 0.0),
                bucket.get("battery_charge", 0.0),
                bucket.get("battery_discharge", 0.0),
            )
            for flow_role, value in flows.items():
                if flow_role != "used_total":
                    quantity_series[flow_role].append([start, value])
            quantity_series["home_consumption"].append([start, flows["used_total"]])
        summary["electricity"] = {
            role: {
                "value": sum(cast(float, point[1]) for point in points),
                "unit": "kWh",
                **({"series": points} if include_series else {}),
            }
            for role, points in quantity_series.items()
        }
    section_roles: tuple[tuple[str, tuple[EnergyRole, ...]], ...] = (
        ("gas", ("gas_consumption",)),
        ("water", ("water_consumption",)),
        ("cost", ("cost",)),
        ("compensation", ("compensation",)),
    )
    for section, roles in section_roles:
        groups: dict[str, dict[str, float]] = {}
        for summary_role in roles:
            for unit, points in values.get(summary_role, {}).items():
                group = groups.setdefault(unit, {})
                for start, value in points.items():
                    group[start] = group.get(start, 0.0) + value
        if groups:
            summary[section] = [
                {
                    "value": sum(points.values()),
                    "unit": unit,
                    **(
                        {"series": [[start, value] for start, value in sorted(points.items())]}
                        if include_series
                        else {}
                    ),
                }
                for unit, points in sorted(groups.items())
            ]
    return summary


def _compute_electricity_flows(
    grid_import: float,
    grid_export: float,
    solar_generation: float,
    battery_charge: float,
    battery_discharge: float,
) -> dict[str, float]:
    """Mirror HA frontend's deterministic per-bucket Energy allocation."""
    to_grid = max(grid_export, 0.0)
    to_battery = max(battery_charge, 0.0)
    solar = max(solar_generation, 0.0)
    from_grid = max(grid_import, 0.0)
    from_battery = max(battery_discharge, 0.0)
    used_total = from_grid + solar + from_battery - to_grid - to_battery
    remaining = max(used_total, 0.0)
    grid_to_battery = max(0.0, min(to_battery, from_grid - remaining))
    to_battery -= grid_to_battery
    from_grid -= grid_to_battery
    solar_to_battery = min(solar, to_battery)
    solar -= solar_to_battery
    to_battery -= solar_to_battery
    solar_to_grid = min(solar, to_grid)
    solar -= solar_to_grid
    to_grid -= solar_to_grid
    battery_to_grid = min(from_battery, to_grid)
    from_battery -= battery_to_grid
    second_grid_to_battery = min(from_grid, to_battery)
    grid_to_battery += second_grid_to_battery
    from_grid -= second_grid_to_battery
    used_solar = min(remaining, solar)
    remaining -= used_solar
    used_battery = min(from_battery, remaining)
    remaining -= used_battery
    used_grid = min(remaining, from_grid)
    return {
        "grid_to_battery": grid_to_battery,
        "battery_to_grid": battery_to_grid,
        "solar_to_battery": solar_to_battery,
        "solar_to_grid": solar_to_grid,
        "used_solar": used_solar,
        "used_grid": used_grid,
        "used_battery": used_battery,
        "used_total": used_total,
    }


def _forecast_results(
    sources: tuple[SafeEnergySourceRecord, ...],
    forecasts: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build forecast output keyed only by safe source IDs."""
    results: list[dict[str, object]] = []
    for source in sources:
        if source.source_type != "solar" or source.source_id not in forecasts:
            continue
        forecast = forecasts[source.source_id]
        raw_points = forecast.get("points")
        points = (
            [
                [point[0], value]
                for point in cast(list[list[object]], raw_points)
                if len(point) == 2 and isinstance(point[0], str) and (value := finite_float(point[1])) is not None
            ]
            if isinstance(raw_points, list)
            else []
        )
        results.append(
            {
                "source_id": source.source_id,
                "name": source.name,
                "unit": "Wh",
                "total": sum(cast(float, point[1]) for point in points),
                "points": points,
            }
        )
    return results


def _carbon_result(
    catalog: SafeEnergyCatalog,
    sources: tuple[SafeEnergySourceRecord, ...],
    source_values: Mapping[EnergyRole, Mapping[str, Mapping[str, float]]],
    metadata: Mapping[str, Mapping[str, object]],
    statistics: Mapping[str, list[dict[str, object]]],
    *,
    include_series: bool,
) -> dict[str, object]:
    """Compute visible grid fossil consumption using HA's percentage formula."""
    if catalog.co2_statistic_id is None:
        reason = (
            "ambiguous"
            if any(
                omission.role == "carbon_signal" and omission.reason == "ambiguous" for omission in catalog.omissions
            )
            else "not_configured"
        )
        return {"available": False, "reason": reason}
    if not any(measure.role == "grid_import" for source in sources for measure in source.measures):
        return {"available": False, "reason": "no_visible_grid_import"}
    co2_metadata = metadata.get(catalog.co2_statistic_id)
    if co2_metadata is None or not co2_metadata.get("has_mean") or co2_metadata.get("unit_of_measurement") != "%":
        return {"available": False, "reason": "metadata_unavailable"}
    co2 = _row_values(statistics.get(catalog.co2_statistic_id, []), "mean")
    if not co2:
        return {"available": False, "reason": "no_data"}
    points: list[list[object]] = []
    assumed = 0
    for start, grid_value in sorted(source_values.get("grid_import", {}).get("kWh", {}).items()):
        if start in co2:
            percentage = co2[start]
        else:
            percentage = 100.0
            assumed += 1
        points.append([start, grid_value * percentage / 100])
    if not points:
        return {"available": False, "reason": "no_data"}
    return {
        "available": True,
        "value": sum(cast(float, point[1]) for point in points),
        "unit": "kWh",
        **({"series": points} if include_series else {}),
        **({"assumed_full_fossil_points": assumed} if assumed else {}),
    }


def _row_values(rows: list[dict[str, object]], field: Literal["change", "mean"]) -> dict[str, float]:
    """Index finite copied statistic values by ISO bucket start."""
    values: dict[str, float] = {}
    for row in rows:
        start = row.get("start")
        value = finite_float(row.get(field))
        if isinstance(start, str) and value is not None:
            values[start] = value
    return values


def _statistic_unit(metadata: Mapping[str, object] | None) -> str | None:
    """Return the normalized public unit for one statistic."""
    if metadata is None:
        return None
    if metadata.get("unit_class") == "energy":
        return "kWh"
    if metadata.get("unit_class") == "power":
        return "kW"
    unit = metadata.get("unit_of_measurement")
    return unit if isinstance(unit, str) and unit else None


def _rate_series(
    statistic_id: str,
    statistics: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    """Return a supported normalized mean series."""
    statistic_metadata = metadata.get(statistic_id)
    if statistic_metadata is None or not statistic_metadata.get("has_mean"):
        return None
    unit = _statistic_unit(statistic_metadata)
    if unit is None:
        return None
    values = _row_values(statistics.get(statistic_id, []), "mean")
    if statistic_metadata.get("unit_class") == "power":
        return {"unit": UnitOfPower.KILO_WATT, "points": [[start, value] for start, value in sorted(values.items())]}
    if unit in VolumeFlowRateConverter.VALID_UNITS:
        return {"unit": unit, "points": [[start, value] for start, value in sorted(values.items())]}
    if unit not in PowerConverter.VALID_UNITS:
        return None
    points: list[list[object]] = []
    for start, value in sorted(values.items()):
        try:
            normalized = finite_float(PowerConverter.convert(value, unit, UnitOfPower.KILO_WATT))
        except HomeAssistantError, OverflowError, TypeError, ValueError:
            return None
        if normalized is None:
            return None
        points.append([start, normalized])
    return {"unit": UnitOfPower.KILO_WATT, "points": points}


def _normalize_rate_value(value: float, unit: str) -> tuple[float, str] | None:
    """Normalize supported power rates to kW and retain native volume-flow units."""
    if unit in VolumeFlowRateConverter.VALID_UNITS:
        return value, unit
    if unit not in PowerConverter.VALID_UNITS:
        return None
    try:
        normalized = finite_float(PowerConverter.convert(value, unit, UnitOfPower.KILO_WATT))
    except HomeAssistantError, OverflowError, TypeError, ValueError:
        return None
    if normalized is None:
        return None
    return normalized, UnitOfPower.KILO_WATT


def fit_energy_result(payload: JsonObjectType, *, limit: int) -> JsonObjectType:
    """Fit an Energy result by deterministic point removal while preserving totals."""
    from ...const import MAX_ENERGY_FORECAST_POINTS, MAX_ENERGY_SERIES_POINTS
    from ..errors import RecoverableToolError

    historical = _historical_point_lists(cast(dict[str, object], payload))
    forecasts = _forecast_point_lists(cast(dict[str, object], payload))
    omitted_series = _limit_historical_points(historical, MAX_ENERGY_SERIES_POINTS)
    omitted_forecast = _limit_forecast_points(forecasts, MAX_ENERGY_FORECAST_POINTS)
    if omitted_series or omitted_forecast:
        cast(dict[str, object], payload)["overflow"] = {
            "truncated": True,
            "limit": limit,
            "omitted_series_points": omitted_series,
            "omitted_forecast_points": omitted_forecast,
        }
    while _compact_json_size(payload) > limit:
        oldest = _oldest_point_list(historical)
        if oldest is not None:
            # Series are chronological, so removing index zero drops the oldest.
            oldest.pop(0)
            omitted_series += 1
        else:
            farthest = _farthest_forecast_list(forecasts)
            if farthest is None:
                raise RecoverableToolError("energy_result_too_large", {})
            farthest.pop()
            omitted_forecast += 1
        cast(dict[str, object], payload)["overflow"] = {
            "truncated": True,
            "limit": limit,
            "omitted_series_points": omitted_series,
            "omitted_forecast_points": omitted_forecast,
        }
    return payload


def _historical_point_lists(payload: dict[str, object]) -> list[list[object]]:
    """Return every historical point list, excluding future forecasts."""
    result: list[list[object]] = []

    def visit(value: object, *, in_forecast: bool = False, parent_key: str | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, in_forecast=in_forecast or key == "forecast", parent_key=key)
        elif isinstance(value, list):
            if (
                not in_forecast
                and parent_key in {"series", "points"}
                and all(_point_timestamp(point) is not None for point in value)
            ):
                result.append(value)
                return
            for child in value:
                visit(child, in_forecast=in_forecast, parent_key=parent_key)

    visit(payload)
    return result


def _forecast_point_lists(payload: dict[str, object]) -> list[list[object]]:
    """Return forecast point lists in stable source order."""
    forecasts = payload.get("forecast")
    if not isinstance(forecasts, list):
        return []
    return [
        cast(list[object], forecast["points"])
        for forecast in forecasts
        if isinstance(forecast, dict) and isinstance(forecast.get("points"), list)
    ]


def _limit_historical_points(point_lists: list[list[object]], limit: int) -> int:
    """Drop globally oldest historical points until the aggregate limit is met."""
    omitted = 0
    remaining = sum(len(points) for points in point_lists)
    while remaining > limit:
        oldest = _oldest_point_list(point_lists)
        if oldest is None:
            break
        oldest.pop(0)
        omitted += 1
        remaining -= 1
    return omitted


def _limit_forecast_points(point_lists: list[list[object]], limit: int) -> int:
    """Keep the globally earliest forecast points with source order as the tie-break."""
    indexed = sorted(
        (
            (_point_timestamp(point), source_index, point_index)
            for source_index, points in enumerate(point_lists)
            for point_index, point in enumerate(points)
            if _point_timestamp(point) is not None
        ),
        key=lambda item: (cast(str, item[0]), item[1]),
    )
    keep = {(source_index, point_index) for _timestamp, source_index, point_index in indexed[:limit]}
    omitted = 0
    for source_index, points in enumerate(point_lists):
        retained = [point for point_index, point in enumerate(points) if (source_index, point_index) in keep]
        omitted += len(points) - len(retained)
        points[:] = retained
    return omitted


def _oldest_point_list(point_lists: list[list[object]]) -> list[object] | None:
    """Return the list containing the globally oldest historical point."""
    candidates = [
        (cast(str, _point_timestamp(points[0])), index)
        for index, points in enumerate(point_lists)
        if points and _point_timestamp(points[0]) is not None
    ]
    if not candidates:
        return None
    return point_lists[min(candidates)[1]]


def _farthest_forecast_list(point_lists: list[list[object]]) -> list[object] | None:
    """Return the list containing the globally farthest-future forecast point."""
    candidates = [
        (cast(str, _point_timestamp(points[-1])), -index)
        for index, points in enumerate(point_lists)
        if points and _point_timestamp(points[-1]) is not None
    ]
    if not candidates:
        return None
    _timestamp, negative_index = max(candidates)
    return point_lists[-negative_index]


def _point_timestamp(point: object) -> str | None:
    """Return the timestamp from a supported historical/forecast point shape."""
    if isinstance(point, list | tuple) and point and isinstance(point[0], str):
        return point[0]
    if isinstance(point, dict) and isinstance(point.get("start"), str):
        return cast(str, point["start"])
    return None


def _compact_json_size(value: object) -> int:
    """Return compact UTF-8 serialized size."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
