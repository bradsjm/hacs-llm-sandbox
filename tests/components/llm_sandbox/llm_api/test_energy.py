"""Focused public-contract tests for Home Assistant Energy queries."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

from custom_components.llm_sandbox.const import TOOL_GET_ENERGY
from custom_components.llm_sandbox.llm_api.data.energy import (
    SafeEnergyCatalog,
    SafeEnergyCurrentPrice,
    SafeEnergyDeviceRecord,
    SafeEnergyMeasureRef,
    SafeEnergySourceRecord,
    fit_energy_result,
    sanitize_energy_preferences,
)
from custom_components.llm_sandbox.llm_api.errors import RecoverableToolError
from custom_components.llm_sandbox.llm_api.tools.energy import (
    EnergyQuerySource,
    GetEnergyTool,
    _sanitize_validation,
    _validation_private_locators,
    _validation_public_locators,
    run_energy_query,
    validate_energy_args,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot, SafeState
from homeassistant.const import UnitOfPower, UnitOfVolumeFlowRate
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.unit_conversion import PowerConverter
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .tools.test_analytics import _snapshot

_START = datetime(2026, 6, 22, tzinfo=UTC)


def _state(entity_id: str, value: str, unit: str = "kWh", *, platform: str | None = None) -> SafeState:
    """Create one visible state using the project snapshot shape."""
    base = _snapshot().states["sensor.temp"]
    domain, object_id = entity_id.split(".", 1)
    return replace(
        base,
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=object_id.replace("_", " ").title(),
        state=value,
        attributes={"unit_of_measurement": unit},
        platform=platform,
    )


def _snapshot_with_states(*states: SafeState) -> HomeSnapshot:
    """Return a UTC snapshot exposing exactly the supplied state identifiers."""
    base = _snapshot()
    return replace(base, states={state.entity_id: state for state in states})


def _rows(statistic_id: str, values: list[float], field: str = "change") -> list[dict[str, object]]:
    """Create daily copied recorder rows."""
    return [
        {"start": (_START + timedelta(days=index)).isoformat(), field: value} for index, value in enumerate(values)
    ]


def _metadata(*ids: str, rate_ids: tuple[str, ...] = ()) -> dict[str, dict[str, object]]:
    """Return supported recorder metadata with public units."""
    return {
        **{
            statistic_id: {"has_sum": True, "unit_class": "energy", "unit_of_measurement": "kWh"}
            for statistic_id in ids
        },
        **{
            statistic_id: {"has_mean": True, "unit_class": "power", "unit_of_measurement": "W"}
            for statistic_id in rate_ids
        },
    }


def _catalog(*, prices: bool = False, devices: bool = True) -> SafeEnergyCatalog:
    """Build a compact safe catalog for query behavior tests."""
    grid_price = SafeEnergyCurrentPrice(0.31, "USD/kWh", "entity") if prices else None
    sources = (
        SafeEnergySourceRecord(
            "grid:0",
            "grid",
            "Grid",
            (
                SafeEnergyMeasureRef("grid_import", "sensor.grid_import", grid_price),
                SafeEnergyMeasureRef(
                    "grid_export",
                    "sensor.grid_export",
                    SafeEnergyCurrentPrice(0.08, "USD/kWh", "fixed") if prices else None,
                ),
                SafeEnergyMeasureRef("cost", "sensor.grid_cost"),
                SafeEnergyMeasureRef("compensation", "sensor.grid_compensation"),
            ),
            "sensor.grid_rate",
            1500.0,
            "W",
        ),
        SafeEnergySourceRecord(
            "solar:0", "solar", "Solar", (SafeEnergyMeasureRef("solar_generation", "sensor.solar"),)
        ),
        SafeEnergySourceRecord(
            "battery:0",
            "battery",
            "Battery",
            (
                SafeEnergyMeasureRef("battery_charge", "sensor.battery_charge"),
                SafeEnergyMeasureRef("battery_discharge", "sensor.battery_discharge"),
            ),
            state_of_charge_value=63.0,
        ),
    )
    tracked = (
        (
            SafeEnergyDeviceRecord("device", "Parent", "sensor.parent"),
            SafeEnergyDeviceRecord("device", "Child one", "sensor.child_one", "sensor.parent"),
            SafeEnergyDeviceRecord("device", "Child two", "sensor.child_two", "sensor.parent"),
        )
        if devices
        else ()
    )
    return SafeEnergyCatalog(sources, tracked, (), None, len(sources) + len(tracked))


def _source(
    catalog: SafeEnergyCatalog,
    rows: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, dict[str, object]],
    *,
    now: datetime = _START + timedelta(days=7),
    calls: list[tuple[datetime, datetime, str]] | None = None,
    validation: tuple[dict[str, object], ...] = (),
) -> EnergyQuerySource:
    """Make a copied-data source whose fetchers cannot expose host state."""

    async def fetch_metadata(ids: set[str]) -> dict[str, dict[str, object]]:
        return {statistic_id: dict(metadata[statistic_id]) for statistic_id in ids if statistic_id in metadata}

    async def fetch_statistics(
        ids: set[str],
        start: datetime,
        end: datetime,
        period: str,
        units: dict[str, str] | None,
        types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        del types
        if calls is not None:
            calls.append((start, end, period))
        result: dict[str, list[dict[str, object]]] = {}
        for statistic_id in ids:
            copied = [
                dict(row)
                for row in rows.get(statistic_id, ())
                if start <= datetime.fromisoformat(cast(str, row["start"])) < end
            ]
            if (
                units == {"energy": "kWh", "power": "kW"}
                and metadata.get(statistic_id, {}).get("unit_class") == "power"
            ):
                copied = [{**row, "mean": cast(float, row["mean"]) / 1000} if "mean" in row else row for row in copied]
            result[statistic_id] = copied
        return result

    async def fetch_forecasts(source_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        return {source_id: {"unit": "Wh", "points": []} for source_id in source_ids}

    async def fetch_validation() -> tuple[dict[str, object], ...]:
        return validation

    visible_ids = frozenset(
        {measure.statistic_id for source in catalog.sources for measure in source.measures}
        | {device.statistic_id for device in catalog.devices}
    )
    return EnergyQuerySource(
        now, catalog, "UTC", visible_ids, fetch_metadata, fetch_statistics, fetch_forecasts, fetch_validation
    )


def _recording_source(
    catalog: SafeEnergyCatalog,
    rows: Mapping[str, list[dict[str, object]]],
    metadata: Mapping[str, dict[str, object]],
    *,
    now: datetime = _START + timedelta(days=7),
) -> tuple[EnergyQuerySource, list[frozenset[str]], list[tuple[frozenset[str], str]]]:
    """Wrap copied fetchers to make their public host-boundary inputs observable."""
    source = _source(catalog, rows, metadata, now=now)
    metadata_calls: list[frozenset[str]] = []
    statistics_calls: list[tuple[frozenset[str], str]] = []

    async def fetch_metadata(ids: set[str]) -> dict[str, dict[str, object]]:
        metadata_calls.append(frozenset(ids))
        return await source.fetch_metadata(ids)

    async def fetch_statistics(
        ids: set[str],
        start: datetime,
        end: datetime,
        period: str,
        units: dict[str, str] | None,
        types: set[str],
    ) -> Mapping[str, list[dict[str, object]]]:
        statistics_calls.append((frozenset(ids), period))
        return await source.fetch_statistics(ids, start, end, period, units, types)

    return (
        replace(source, fetch_metadata=fetch_metadata, fetch_statistics=fetch_statistics),
        metadata_calls,
        statistics_calls,
    )


def _gas_catalog(count: int) -> SafeEnergyCatalog:
    """Create independent cumulative gas sources for cardinality-bound tests."""
    sources = tuple(
        SafeEnergySourceRecord(
            f"gas:{index}",
            "gas",
            f"Gas {index}",
            (SafeEnergyMeasureRef("gas_consumption", f"sensor.gas_{index:02}"),),
        )
        for index in range(count)
    )
    return SafeEnergyCatalog(sources, (), (), None, count)


async def test_sanitizer_keeps_security_omissions_count_only() -> None:
    """Unsafe statistics and forecast configuration never cross the safe boundary."""
    snapshot = _snapshot_with_states(_state("sensor.visible", "12"), _state("sensor.visible_price", "0.2", "USD/kWh"))
    catalog, forecasts = sanitize_energy_preferences(
        snapshot,
        {
            "energy_sources": [
                {
                    "type": "solar",
                    "stat_energy_from": "sensor.visible",
                    "config_entry_solar_forecast": ["secret-entry"],
                },
                {"type": "grid", "stat_energy_from": "external:meter", "entity_energy_price": "sensor.hidden_price"},
            ],
            "device_consumption": [{"stat_consumption": "sensor.hidden_device"}],
        },
        {"external:meter": "sensor.hidden_cost"},
    )

    public = json.dumps(catalog, default=lambda item: item.__dict__ if hasattr(item, "__dict__") else str(item))
    assert forecasts == {"solar:0": ("secret-entry",)}
    assert [source.source_id for source in catalog.sources] == ["solar:0"]
    assert "secret-entry" not in public
    assert "external:meter" not in public
    assert "hidden_price" not in public
    assert "hidden_device" not in public
    assert "hidden_cost" not in public
    assert {(item.role, item.reason, item.count) for item in catalog.omissions} >= {
        ("grid_import", "external_statistic", 1),
        ("current_price", "not_visible", 1),
        ("device_consumption", "not_visible", 1),
    }


@pytest.mark.parametrize(
    ("states", "expected_statistic_id", "expected_omissions", "rejected_ids"),
    [
        pytest.param((), None, set(), (), id="no-candidates"),
        pytest.param(
            (_state("sensor.co2_signal", "42", "%", platform="co2signal"),),
            "sensor.co2_signal",
            set(),
            (),
            id="one-visible-signal",
        ),
        pytest.param(
            (
                _state("sensor.co2_alpha", "42", "%", platform="co2signal"),
                _state("sensor.co2_beta", "43", "%", platform="co2signal"),
            ),
            None,
            {("carbon_signal", "ambiguous", 1)},
            ("sensor.co2_alpha", "sensor.co2_beta"),
            id="multiple-visible-signals",
        ),
        pytest.param(
            (_state("sensor.co2_wrong_platform", "42", "%", platform="other"),),
            None,
            set(),
            ("sensor.co2_wrong_platform",),
            id="wrong-platform",
        ),
        pytest.param(
            (_state("sensor.co2_wrong_unit", "42", "g/kWh", platform="co2signal"),),
            None,
            set(),
            ("sensor.co2_wrong_unit",),
            id="wrong-unit",
        ),
    ],
)
def test_sanitizer_selects_only_one_visible_co2signal_percentage(
    states: tuple[SafeState, ...],
    expected_statistic_id: str | None,
    expected_omissions: set[tuple[str, str, int]],
    rejected_ids: tuple[str, ...],
) -> None:
    """Carbon discovery exposes exactly one qualifying visible statistic."""
    catalog, _ = sanitize_energy_preferences(_snapshot_with_states(*states), {"energy_sources": []}, {})

    serialized = json.dumps(catalog, default=lambda item: item.__dict__ if hasattr(item, "__dict__") else str(item))
    assert catalog.co2_statistic_id == expected_statistic_id
    assert {(item.role, item.reason, item.count) for item in catalog.omissions} == expected_omissions
    assert all(rejected_id not in serialized for rejected_id in rejected_ids)


async def test_seven_bucket_summary_series_and_device_exclusive_allocation() -> None:
    """Daily totals, trends, and tracked-parent subtraction use the same visible buckets."""
    catalog = _catalog()
    values = {
        "sensor.grid_import": _rows("sensor.grid_import", [10.0] * 7),
        "sensor.grid_export": _rows("sensor.grid_export", [3.0] * 7),
        "sensor.solar": _rows("sensor.solar", [8.0] * 7),
        "sensor.battery_charge": _rows("sensor.battery_charge", [1.0] * 7),
        "sensor.battery_discharge": _rows("sensor.battery_discharge", [2.0] * 7),
        "sensor.parent": _rows("sensor.parent", [8.0] * 7),
        "sensor.child_one": _rows("sensor.child_one", [3.0] * 7),
        "sensor.child_two": _rows("sensor.child_two", [2.0] * 7),
    }
    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(days=7)).isoformat(),
                "period": "day",
                "include": ["summary", "series"],
            }
        ),
        _source(catalog, values, _metadata(*values)),
    )

    electricity = cast(dict[str, object], cast(dict[str, object], result["summary"])["electricity"])
    assert electricity["home_consumption"] == {
        "value": 112.0,
        "unit": "kWh",
        "series": [[(_START + timedelta(days=index)).isoformat(), 16.0] for index in range(7)],
    }
    parent = next(
        device
        for device in cast(list[dict[str, object]], result["devices"])
        if device["statistic_id"] == "sensor.parent"
    )
    assert parent["inclusive_value"] == 56.0
    assert parent["exclusive_value"] == 21.0
    assert [point["exclusive_value"] for point in cast(list[dict[str, object]], parent["series"])] == [3.0] * 7
    bucket_starts = [(_START + timedelta(days=index)).isoformat() for index in range(7)]
    expected_electricity_values = {
        "grid_import": 10.0,
        "grid_export": 3.0,
        "solar_generation": 8.0,
        "battery_charge": 1.0,
        "battery_discharge": 2.0,
        "home_consumption": 16.0,
        "grid_to_battery": 0.0,
        "battery_to_grid": 0.0,
        "solar_to_battery": 1.0,
        "solar_to_grid": 3.0,
        "used_solar": 4.0,
        "used_grid": 10.0,
        "used_battery": 2.0,
    }
    expected_electricity = {
        role: {
            "value": value * 7,
            "unit": "kWh",
            "series": [[start, value] for start in bucket_starts],
        }
        for role, value in expected_electricity_values.items()
    }
    assert electricity == expected_electricity

    grid = next(
        source for source in cast(list[dict[str, object]], result["sources"]) if source["source_id"] == "grid:0"
    )
    assert {measure["role"]: measure for measure in cast(list[dict[str, object]], grid["measures"])} == {
        "grid_import": {
            "role": "grid_import",
            "statistic_id": "sensor.grid_import",
            **expected_electricity["grid_import"],
        },
        "grid_export": {
            "role": "grid_export",
            "statistic_id": "sensor.grid_export",
            **expected_electricity["grid_export"],
        },
    }
    solar = next(
        source for source in cast(list[dict[str, object]], result["sources"]) if source["source_id"] == "solar:0"
    )
    assert cast(list[dict[str, object]], solar["measures"]) == [
        {
            "role": "solar_generation",
            "statistic_id": "sensor.solar",
            **expected_electricity["solar_generation"],
        }
    ]
    battery = next(
        source for source in cast(list[dict[str, object]], result["sources"]) if source["source_id"] == "battery:0"
    )
    assert cast(list[dict[str, object]], battery["measures"]) == [
        {
            "role": "battery_charge",
            "statistic_id": "sensor.battery_charge",
            **expected_electricity["battery_charge"],
        },
        {
            "role": "battery_discharge",
            "statistic_id": "sensor.battery_discharge",
            **expected_electricity["battery_discharge"],
        },
    ]


async def test_prices_rates_costs_and_compensation_use_public_units() -> None:
    """Prices omit entity IDs while power and money retain only public values and units."""
    catalog = _catalog(prices=True, devices=False)
    rows = {
        "sensor.grid_import": _rows("sensor.grid_import", [10.0]),
        "sensor.grid_export": _rows("sensor.grid_export", [3.0]),
        "sensor.grid_cost": _rows("sensor.grid_cost", [4.5]),
        "sensor.grid_compensation": _rows("sensor.grid_compensation", [0.7]),
        "sensor.grid_rate": _rows("sensor.grid_rate", [1500.0], "mean"),
    }
    metadata = _metadata(
        "sensor.grid_import",
        "sensor.grid_export",
        "sensor.grid_cost",
        "sensor.grid_compensation",
        rate_ids=("sensor.grid_rate",),
    )
    metadata["sensor.grid_cost"] = {
        "has_sum": True,
        "unit_class": "monetary",
        "unit_of_measurement": "USD",
    }
    metadata["sensor.grid_compensation"] = dict(metadata["sensor.grid_cost"])
    result = await run_energy_query(
        validate_energy_args(
            {
                "hours": 24,
                "period": "hour",
                "source_types": ["grid"],
                "include": ["summary", "series", "current"],
            }
        ),
        _source(catalog, rows, metadata, now=_START + timedelta(days=1)),
    )

    grid = next(
        source for source in cast(list[dict[str, object]], result["sources"]) if source["source_id"] == "grid:0"
    )
    assert grid["current_rate"] == {"value": 1.5, "unit": "kW"}
    assert grid["rate_series"] == {"unit": "kW", "points": [[_START.isoformat(), 1.5]]}
    prices = {
        measure["role"]: measure["current_price"]
        for measure in cast(list[dict[str, object]], grid["measures"])
        if "current_price" in measure
    }
    assert prices == {
        "grid_import": {"value": 0.31, "unit": "USD/kWh", "source": "entity"},
        "grid_export": {"value": 0.08, "unit": "USD/kWh", "source": "fixed"},
    }
    summary = cast(dict[str, object], result["summary"])
    assert summary["cost"] == [{"value": 4.5, "unit": "USD", "series": [[_START.isoformat(), 4.5]]}]
    assert summary["compensation"] == [{"value": 0.7, "unit": "USD", "series": [[_START.isoformat(), 0.7]]}]
    assert "sensor.grid_rate" not in json.dumps(result)


@pytest.mark.parametrize(
    ("source_type", "input_unit", "expected_value", "expected_unit"),
    [
        pytest.param(
            "grid",
            unit,
            PowerConverter.convert(2.5, unit, UnitOfPower.KILO_WATT),
            UnitOfPower.KILO_WATT,
            id=f"power-{unit}",
        )
        for unit in sorted(PowerConverter.VALID_UNITS, key=str)
    ]
    + [
        pytest.param(
            "gas",
            UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR,
            2.5,
            UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR,
            id="gas-cubic-meters-per-hour",
        ),
        pytest.param(
            "water",
            UnitOfVolumeFlowRate.GALLONS_PER_MINUTE,
            2.5,
            UnitOfVolumeFlowRate.GALLONS_PER_MINUTE,
            id="water-gallons-per-minute",
        ),
    ],
)
async def test_current_rate_normalizes_all_power_units_and_preserves_volume_flow_units(
    source_type: Literal["grid", "gas", "water"],
    input_unit: str,
    expected_value: float,
    expected_unit: str,
) -> None:
    """Current power normalizes to kW while native volume-flow rates are unchanged."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                f"{source_type}:0",
                source_type,
                "Rate",
                (),
                current_rate_value=2.5,
                current_rate_unit=input_unit,
            ),
        ),
        (),
        (),
        None,
        1,
    )
    result = await run_energy_query(
        validate_energy_args({"include": ["current"], "source_types": [source_type]}),
        _source(catalog, {}, {}),
    )

    current_rate = cast(dict[str, object], cast(list[dict[str, object]], result["sources"])[0]["current_rate"])
    assert current_rate == {"value": pytest.approx(expected_value), "unit": expected_unit}


@pytest.mark.parametrize(
    ("source_type", "device_source_type", "input_unit", "rate_metadata", "expected_current", "expected_series"),
    [
        pytest.param(
            "grid",
            "device",
            "W",
            {"has_mean": True, "unit_of_measurement": "W"},
            {"value": 0.0025, "unit": "kW"},
            {"unit": "kW", "points": [[_START.isoformat(), 0.0025]]},
            id="power-normalized-to-kw",
        ),
        pytest.param(
            "water",
            "device_water",
            UnitOfVolumeFlowRate.GALLONS_PER_MINUTE,
            {"has_mean": True, "unit_of_measurement": UnitOfVolumeFlowRate.GALLONS_PER_MINUTE},
            {"value": 2.5, "unit": UnitOfVolumeFlowRate.GALLONS_PER_MINUTE},
            {
                "unit": UnitOfVolumeFlowRate.GALLONS_PER_MINUTE,
                "points": [[_START.isoformat(), 2.5]],
            },
            id="volume-flow-retains-native-unit",
        ),
        pytest.param(
            "grid",
            "device",
            "°C",
            {"has_mean": True, "unit_class": "temperature", "unit_of_measurement": "°C"},
            None,
            None,
            id="temperature-omitted",
        ),
        pytest.param(
            "grid",
            "device",
            "kWh",
            {"has_mean": True, "unit_class": "energy", "unit_of_measurement": "kWh"},
            None,
            None,
            id="energy-omitted",
        ),
    ],
)
async def test_current_and_historical_rates_allow_only_power_or_volume_flow_units(
    source_type: Literal["grid", "water"],
    device_source_type: Literal["device", "device_water"],
    input_unit: str,
    rate_metadata: dict[str, object],
    expected_current: dict[str, object] | None,
    expected_series: dict[str, object] | None,
) -> None:
    """Only power and volume-flow rate units have public current or historical outputs."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                f"{source_type}:0",
                source_type,
                "Source rate",
                (),
                "sensor.source_rate",
                2.5,
                input_unit,
            ),
        ),
        (
            SafeEnergyDeviceRecord(
                device_source_type,
                "Device rate",
                "sensor.device_consumption",
                rate_statistic_id="sensor.device_rate",
                current_rate_value=2.5,
                current_rate_unit=input_unit,
            ),
        ),
        (),
        None,
        2,
    )
    result = await run_energy_query(
        validate_energy_args({"hours": 24, "period": "hour", "include": ["current", "series"]}),
        _source(
            catalog,
            {
                "sensor.source_rate": _rows("sensor.source_rate", [2.5], "mean"),
                "sensor.device_rate": _rows("sensor.device_rate", [2.5], "mean"),
            },
            {
                "sensor.source_rate": dict(rate_metadata),
                "sensor.device_rate": dict(rate_metadata),
            },
            now=_START + timedelta(days=1),
        ),
    )

    source = cast(list[dict[str, object]], result["sources"])[0]
    device = cast(list[dict[str, object]], result["devices"])[0]
    assert source.get("current_rate") == expected_current
    assert source.get("rate_series") == expected_series
    assert device.get("current_rate") == expected_current
    assert device.get("rate_series") == expected_series
    assert ("current_rate" in source) is (expected_current is not None)
    assert ("rate_series" in source) is (expected_series is not None)
    assert ("current_rate" in device) is (expected_current is not None)
    assert ("rate_series" in device) is (expected_series is not None)


async def test_empty_supported_non_electric_cumulative_statistics_return_native_zeroes() -> None:
    """Every supported empty cumulative measure and device returns a native-unit zero."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "gas:0",
                "gas",
                "Gas",
                (
                    SafeEnergyMeasureRef("gas_consumption", "sensor.gas"),
                    SafeEnergyMeasureRef("cost", "sensor.gas_cost"),
                ),
            ),
            SafeEnergySourceRecord(
                "water:0",
                "water",
                "Water",
                (
                    SafeEnergyMeasureRef("water_consumption", "sensor.water"),
                    SafeEnergyMeasureRef("compensation", "sensor.water_compensation"),
                ),
            ),
        ),
        (SafeEnergyDeviceRecord("device_water", "Water device", "sensor.device_water"),),
        (),
        None,
        3,
    )
    metadata = {
        "sensor.gas": {"has_sum": True, "unit_of_measurement": "m³"},
        "sensor.gas_cost": {"has_sum": True, "unit_of_measurement": "USD"},
        "sensor.water": {"has_sum": True, "unit_of_measurement": "gal"},
        "sensor.water_compensation": {"has_sum": True, "unit_of_measurement": "EUR"},
        "sensor.device_water": {"has_sum": True, "unit_of_measurement": "L"},
    }
    result = await run_energy_query(
        validate_energy_args({"hours": 24, "period": "hour", "include": ["summary", "series"]}),
        _source(catalog, {}, metadata, now=_START + timedelta(days=1)),
    )

    sources = {cast(str, source["source_id"]): source for source in cast(list[dict[str, object]], result["sources"])}
    assert {
        measure["role"]: {key: measure[key] for key in ("value", "unit", "series")}
        for source in sources.values()
        for measure in cast(list[dict[str, object]], source["measures"])
    } == {
        "gas_consumption": {"value": 0.0, "unit": "m³", "series": []},
        "cost": {"value": 0.0, "unit": "USD", "series": []},
        "water_consumption": {"value": 0.0, "unit": "gal", "series": []},
        "compensation": {"value": 0.0, "unit": "EUR", "series": []},
    }
    assert cast(list[dict[str, object]], result["devices"]) == [
        {
            "source_type": "device_water",
            "name": "Water device",
            "statistic_id": "sensor.device_water",
            "inclusive_value": 0.0,
            "exclusive_value": 0.0,
            "unit": "L",
            "series": [],
        }
    ]
    assert result["summary"] == {
        "gas": [{"value": 0.0, "unit": "m³", "series": []}],
        "water": [{"value": 0.0, "unit": "gal", "series": []}],
        "cost": [{"value": 0.0, "unit": "USD", "series": []}],
        "compensation": [{"value": 0.0, "unit": "EUR", "series": []}],
    }


async def test_nonfinite_cumulative_row_is_omitted_instead_of_becoming_an_empty_window_zero() -> None:
    """A populated unsupported row does not use the no-rows zero-value rule."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "water:0",
                "water",
                "Water",
                (SafeEnergyMeasureRef("water_consumption", "sensor.water"),),
            ),
        ),
        (),
        (),
        None,
        1,
    )
    result = await run_energy_query(
        validate_energy_args({"hours": 24, "period": "hour", "include": ["summary", "series"]}),
        _source(
            catalog,
            {"sensor.water": _rows("sensor.water", [float("nan")])},
            {"sensor.water": {"has_sum": True, "unit_of_measurement": "m³"}},
            now=_START + timedelta(days=1),
        ),
    )

    assert "measures" not in cast(list[dict[str, object]], result["sources"])[0]
    assert "water" not in cast(dict[str, object], result["summary"])


async def test_invalid_configured_solar_rows_do_not_create_electricity_zeroes_or_flows() -> None:
    """Configured invalid solar data makes all derived electricity quantities unavailable."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "grid:0",
                "grid",
                "Grid",
                (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
            ),
            SafeEnergySourceRecord(
                "solar:0",
                "solar",
                "Solar",
                (SafeEnergyMeasureRef("solar_generation", "sensor.solar"),),
            ),
        ),
        (),
        (),
        None,
        2,
    )
    result = await run_energy_query(
        validate_energy_args({"hours": 24, "period": "hour", "include": ["summary", "series"]}),
        _source(
            catalog,
            {
                "sensor.grid_import": _rows("sensor.grid_import", [10.0]),
                "sensor.solar": _rows("sensor.solar", [float("nan")]),
            },
            _metadata("sensor.grid_import", "sensor.solar"),
            now=_START + timedelta(days=1),
        ),
    )

    sources = {cast(str, source["source_id"]): source for source in cast(list[dict[str, object]], result["sources"])}
    assert sources["grid:0"]["measures"] == [
        {
            "role": "grid_import",
            "statistic_id": "sensor.grid_import",
            "value": 10.0,
            "unit": "kWh",
            "series": [[_START.isoformat(), 10.0]],
        }
    ]
    assert "measures" not in sources["solar:0"]

    electricity = cast(dict[str, object], cast(dict[str, object], result["summary"])["electricity"])
    assert electricity["grid_import"] == {
        "value": 10.0,
        "unit": "kWh",
        "series": [[_START.isoformat(), 10.0]],
    }
    assert "solar_generation" not in electricity
    assert all(
        role not in electricity
        for role in (
            "home_consumption",
            "grid_to_battery",
            "battery_to_grid",
            "solar_to_battery",
            "solar_to_grid",
            "used_solar",
            "used_grid",
            "used_battery",
        )
    )
    assert all(
        cast(dict[str, float], electricity[role])["value"] == 0.0
        for role in ("grid_export", "battery_charge", "battery_discharge")
    )


async def test_empty_supported_electricity_window_omits_incomplete_role_flows() -> None:
    """Supported empty measures remain zero, but an unsupported peer role blocks flows."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "grid:0",
                "grid",
                "Grid",
                (
                    SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),
                    SafeEnergyMeasureRef("grid_export", "sensor.grid_export"),
                ),
            ),
            SafeEnergySourceRecord(
                "solar:0",
                "solar",
                "Solar",
                (SafeEnergyMeasureRef("solar_generation", "sensor.solar"),),
            ),
            SafeEnergySourceRecord(
                "battery:0",
                "battery",
                "Battery",
                (
                    SafeEnergyMeasureRef("battery_charge", "sensor.battery_charge"),
                    SafeEnergyMeasureRef("battery_discharge", "sensor.battery_discharge"),
                ),
            ),
            SafeEnergySourceRecord(
                "solar:1",
                "solar",
                "Unsupported solar",
                (SafeEnergyMeasureRef("solar_generation", "sensor.unsupported_solar"),),
            ),
        ),
        (),
        (),
        None,
        4,
    )
    supported_ids = (
        "sensor.grid_import",
        "sensor.grid_export",
        "sensor.solar",
        "sensor.battery_charge",
        "sensor.battery_discharge",
    )
    metadata = _metadata(*supported_ids)
    metadata["sensor.unsupported_solar"] = {
        "has_sum": False,
        "unit_class": "energy",
        "unit_of_measurement": "kWh",
    }
    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(days=1)).isoformat(),
                "period": "hour",
                "source_types": ["grid", "solar", "battery"],
                "include": ["summary", "series"],
            }
        ),
        _source(catalog, {}, metadata),
    )

    expected_measure = {"value": 0.0, "unit": "kWh", "series": []}
    sources = {cast(str, source["source_id"]): source for source in cast(list[dict[str, object]], result["sources"])}
    assert {
        measure["role"]: {key: measure[key] for key in ("value", "unit", "series")}
        for source_id in ("grid:0", "solar:0", "battery:0")
        for measure in cast(list[dict[str, object]], sources[source_id]["measures"])
    } == {
        "grid_import": expected_measure,
        "grid_export": expected_measure,
        "solar_generation": expected_measure,
        "battery_charge": expected_measure,
        "battery_discharge": expected_measure,
    }
    assert "measures" not in sources["solar:1"]
    assert result["omissions"] == [{"role": "solar_generation", "reason": "metadata_unavailable", "count": 1}]

    summary = cast(dict[str, object], result["summary"])
    electricity = cast(dict[str, object], summary["electricity"])
    assert {
        role: electricity[role] for role in ("grid_import", "grid_export", "battery_charge", "battery_discharge")
    } == dict.fromkeys(
        ("grid_import", "grid_export", "battery_charge", "battery_discharge"),
        expected_measure,
    )
    assert "solar_generation" not in electricity
    assert all(
        role not in electricity
        for role in (
            "home_consumption",
            "grid_to_battery",
            "battery_to_grid",
            "solar_to_battery",
            "solar_to_grid",
            "used_solar",
            "used_grid",
            "used_battery",
        )
    )


async def test_empty_supported_electricity_window_returns_zero_flows() -> None:
    """A fully supported empty electricity window retains direct and derived zeros."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "grid:0",
                "grid",
                "Grid",
                (
                    SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),
                    SafeEnergyMeasureRef("grid_export", "sensor.grid_export"),
                ),
            ),
            SafeEnergySourceRecord(
                "solar:0",
                "solar",
                "Solar",
                (SafeEnergyMeasureRef("solar_generation", "sensor.solar"),),
            ),
            SafeEnergySourceRecord(
                "battery:0",
                "battery",
                "Battery",
                (
                    SafeEnergyMeasureRef("battery_charge", "sensor.battery_charge"),
                    SafeEnergyMeasureRef("battery_discharge", "sensor.battery_discharge"),
                ),
            ),
        ),
        (),
        (),
        None,
        3,
    )
    supported_ids = (
        "sensor.grid_import",
        "sensor.grid_export",
        "sensor.solar",
        "sensor.battery_charge",
        "sensor.battery_discharge",
    )

    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(days=1)).isoformat(),
                "period": "hour",
                "source_types": ["grid", "solar", "battery"],
                "include": ["summary", "series"],
            }
        ),
        _source(catalog, {}, _metadata(*supported_ids)),
    )

    expected_measure = {"value": 0.0, "unit": "kWh", "series": []}
    summary = cast(dict[str, object], result["summary"])
    assert summary["electricity"] == dict.fromkeys(
        (
            "grid_import",
            "grid_export",
            "solar_generation",
            "battery_charge",
            "battery_discharge",
            "home_consumption",
            "grid_to_battery",
            "battery_to_grid",
            "solar_to_battery",
            "solar_to_grid",
            "used_solar",
            "used_grid",
            "used_battery",
        ),
        expected_measure,
    )


async def test_period_budget_comparison_and_alignment_contracts() -> None:
    """Auto coarsens safely while explicit oversized requests and boundaries stay deterministic."""
    catalog = _catalog(devices=False)
    rows = {"sensor.grid_import": _rows("sensor.grid_import", [1.0])}
    source = _source(catalog, rows, _metadata("sensor.grid_import"))
    year_start = datetime(2024, 2, 29, tzinfo=UTC)
    auto = await run_energy_query(
        validate_energy_args(
            {
                "start": year_start.isoformat(),
                "end": datetime(2025, 3, 1, tzinfo=UTC).isoformat(),
                "include": ["summary"],
            }
        ),
        source,
    )
    assert auto["period"] in {"day", "week", "month", "year"}
    with pytest.raises(RecoverableToolError, match="energy_query_too_large"):
        await run_energy_query(
            validate_energy_args(
                {
                    "start": _START.isoformat(),
                    "end": (_START + timedelta(days=366)).isoformat(),
                    "period": "hour",
                    "include": ["summary"],
                }
            ),
            source,
        )

    calls: list[tuple[datetime, datetime, str]] = []
    compare = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(days=7)).isoformat(),
                "period": "day",
                "include": ["summary"],
                "compare": "previous",
            }
        ),
        _source(catalog, rows, _metadata("sensor.grid_import"), calls=calls),
    )
    assert cast(dict[str, object], compare["comparison"])["window"] == {
        "start": (_START - timedelta(days=7)).isoformat(),
        "end": _START.isoformat(),
    }
    leap = await run_energy_query(
        validate_energy_args(
            {
                "start": year_start.isoformat(),
                "end": datetime(2024, 3, 1, tzinfo=UTC).isoformat(),
                "period": "day",
                "include": ["summary"],
                "compare": "year_over_year",
            }
        ),
        source,
    )
    assert (
        cast(dict[str, object], cast(dict[str, object], leap["comparison"])["window"])["start"]
        == "2023-02-28T00:00:00+00:00"
    )

    unaligned = await run_energy_query(
        validate_energy_args(
            {
                "start": "2026-06-22T00:17:00+00:00",
                "end": "2026-06-22T02:03:00+00:00",
                "period": "hour",
                "include": ["summary"],
            }
        ),
        source,
    )
    assert unaligned["window"] == {
        "start": "2026-06-22T00:00:00+00:00",
        "end": "2026-06-22T03:00:00+00:00",
        "requested": {"start": "2026-06-22T00:17:00+00:00", "end": "2026-06-22T02:03:00+00:00"},
    }


async def test_year_over_year_auto_uses_larger_comparison_bucket_count() -> None:
    """Auto period selection considers the leap-year comparison window before reads."""
    catalog = _gas_catalog(40)
    statistic_ids = tuple(f"sensor.gas_{index:02}" for index in range(40))
    source, _, statistics_calls = _recording_source(catalog, {}, _metadata(*statistic_ids))

    result = await run_energy_query(
        validate_energy_args(
            {
                "start": "2025-01-01T00:00:00+00:00",
                "end": "2025-10-28T00:00:00+00:00",
                "include": ["summary"],
                "compare": "year_over_year",
            }
        ),
        source,
    )

    assert result["period"] == "month"
    assert [period for _, period in statistics_calls] == ["month", "month"]


async def test_weekly_year_over_year_fetches_exact_shifted_window_as_daily_comparison() -> None:
    """Weekly YoY keeps the primary week but avoids widening its shifted predecessor."""
    catalog = _catalog(devices=False)
    primary_start = datetime(2025, 1, 6, tzinfo=UTC)
    primary_end = datetime(2025, 1, 13, tzinfo=UTC)
    comparison_start = datetime(2024, 1, 6, tzinfo=UTC)
    comparison_end = datetime(2024, 1, 13, tzinfo=UTC)
    calls: list[tuple[datetime, datetime, str]] = []
    rows = {
        "sensor.grid_import": [
            {"start": primary_start.isoformat(), "change": 11.0},
            {"start": "2024-01-01T00:00:00+00:00", "change": 100.0},
            *[
                {"start": (comparison_start + timedelta(days=offset)).isoformat(), "change": 2.0}
                for offset in range(7)
            ],
            {"start": comparison_end.isoformat(), "change": 100.0},
        ]
    }

    result = await run_energy_query(
        validate_energy_args(
            {
                "start": primary_start.isoformat(),
                "end": primary_end.isoformat(),
                "period": "week",
                "include": ["summary"],
                "compare": "year_over_year",
            }
        ),
        _source(catalog, rows, _metadata("sensor.grid_import"), calls=calls),
    )

    assert calls == [
        (primary_start, primary_end, "week"),
        (comparison_start, comparison_end, "day"),
    ]
    comparison = cast(dict[str, object], result["comparison"])
    assert comparison["window"] == {
        "start": comparison_start.isoformat(),
        "end": comparison_end.isoformat(),
    }
    electricity = cast(dict[str, object], cast(dict[str, object], comparison["summary"])["electricity"])
    assert electricity["grid_import"] == {"value": 14.0, "unit": "kWh"}


async def test_carbon_only_large_catalog_fetches_only_visible_grid_import_and_co2() -> None:
    """Carbon-only reads do not inherit irrelevant dashboard statistic cardinality."""
    grid_id = "sensor.grid_import"
    co2_id = "sensor.co2"
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "grid:0",
                "grid",
                "Grid",
                (SafeEnergyMeasureRef("grid_import", grid_id),),
            ),
            *_gas_catalog(40).sources,
        ),
        (),
        (),
        co2_id,
        41,
    )
    source, metadata_calls, statistics_calls = _recording_source(
        catalog,
        {
            grid_id: [{"start": _START.isoformat(), "change": 10.0}],
            co2_id: [{"start": _START.isoformat(), "mean": 25.0}],
        },
        {
            grid_id: _metadata(grid_id)[grid_id],
            co2_id: {"has_mean": True, "unit_of_measurement": "%"},
        },
    )

    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(hours=1)).isoformat(),
                "period": "hour",
                "include": ["carbon"],
            }
        ),
        source,
    )

    fetched_ids = frozenset({grid_id, co2_id})
    assert result["carbon"] == {"available": True, "value": 2.5, "unit": "kWh"}
    assert metadata_calls == [fetched_ids]
    assert statistics_calls == [(fetched_ids, "hour")]


@pytest.mark.parametrize(
    ("catalog", "source_types", "reason"),
    [
        pytest.param(
            SafeEnergyCatalog(
                (
                    SafeEnergySourceRecord(
                        "grid:0",
                        "grid",
                        "Grid",
                        (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
                    ),
                ),
                (),
                (),
                None,
                1,
            ),
            None,
            "not_configured",
            id="no-safe-co2-statistic",
        ),
        pytest.param(
            SafeEnergyCatalog(
                (
                    SafeEnergySourceRecord(
                        "grid:0",
                        "grid",
                        "Grid",
                        (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
                    ),
                    SafeEnergySourceRecord(
                        "solar:0",
                        "solar",
                        "Solar",
                        (SafeEnergyMeasureRef("solar_generation", "sensor.solar_generation"),),
                    ),
                ),
                (),
                (),
                "sensor.co2",
                2,
            ),
            ["solar"],
            "no_visible_grid_import",
            id="no-selected-visible-grid-import",
        ),
    ],
)
async def test_carbon_only_incomplete_pair_skips_long_hourly_query(
    catalog: SafeEnergyCatalog, source_types: list[str] | None, reason: str
) -> None:
    """Carbon-only requests need both safe CO₂ and selected grid import before recorder work."""
    source, metadata_calls, statistics_calls = _recording_source(catalog, {}, {})
    data: dict[str, object] = {
        "hours": 20_000,
        "period": "hour",
        "include": ["carbon"],
    }
    if source_types is not None:
        data["source_types"] = source_types

    result = await run_energy_query(validate_energy_args(data), source)

    assert result["carbon"] == {"available": False, "reason": reason}
    assert metadata_calls == []
    assert statistics_calls == []


async def test_gas_and_cost_group_series_auto_coarsens_before_host_calls() -> None:
    """Auto includes unit-grouped gas and cost summary series in its point budget."""
    gas_id = "sensor.gas"
    cost_id = "sensor.gas_cost"
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "gas:0",
                "gas",
                "Gas",
                (
                    SafeEnergyMeasureRef("gas_consumption", gas_id),
                    SafeEnergyMeasureRef("cost", cost_id),
                ),
            ),
        ),
        (),
        (),
        None,
        1,
    )
    source, _, statistics_calls = _recording_source(
        catalog,
        {},
        {
            gas_id: {"has_sum": True, "unit_of_measurement": "m³"},
            cost_id: {"has_sum": True, "unit_of_measurement": "USD"},
        },
    )

    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(hours=126)).isoformat(),
                "include": ["series"],
            }
        ),
        source,
    )

    assert result["period"] == "day"
    assert [period for _, period in statistics_calls] == ["day"]


@pytest.mark.parametrize(
    ("catalog", "metadata", "args", "expected_placeholders"),
    [
        pytest.param(
            _gas_catalog(40),
            _metadata(*(f"sensor.gas_{index:02}" for index in range(40))),
            {
                "start": "2025-01-01T00:00:00+00:00",
                "end": "2025-10-28T00:00:00+00:00",
                "period": "day",
                "include": ["summary"],
                "compare": "year_over_year",
            },
            {"statistic_count": "40", "bucket_count": "301", "max_points": "12000"},
            id="year-over-year-comparison-window",
        ),
        pytest.param(
            SafeEnergyCatalog(
                (
                    SafeEnergySourceRecord(
                        "gas:0",
                        "gas",
                        "Gas",
                        (
                            SafeEnergyMeasureRef("gas_consumption", "sensor.gas"),
                            SafeEnergyMeasureRef("cost", "sensor.gas_cost"),
                        ),
                    ),
                ),
                (),
                (),
                None,
                1,
            ),
            {
                "sensor.gas": {"has_sum": True, "unit_of_measurement": "m³"},
                "sensor.gas_cost": {"has_sum": True, "unit_of_measurement": "USD"},
            },
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(hours=126)).isoformat(),
                "period": "hour",
                "include": ["series"],
            },
            {"statistic_count": "4", "bucket_count": "126", "max_points": "500"},
            id="gas-and-cost-unit-groups",
        ),
        pytest.param(
            SafeEnergyCatalog(
                (
                    SafeEnergySourceRecord(
                        "grid:0",
                        "grid",
                        "Grid",
                        (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
                    ),
                ),
                (),
                (),
                None,
                1,
            ),
            _metadata("sensor.grid_import"),
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(hours=36)).isoformat(),
                "period": "hour",
                "include": ["series"],
            },
            {"statistic_count": "14", "bucket_count": "36", "max_points": "500"},
            id="grid-import-electricity-derived-series",
        ),
    ],
)
async def test_explicit_period_budget_errors_are_pre_host(
    catalog: SafeEnergyCatalog,
    metadata: Mapping[str, dict[str, object]],
    args: dict[str, object],
    expected_placeholders: dict[str, str],
) -> None:
    """Every explicit point-budget error is truthful and rejects before host work."""
    source, metadata_calls, statistics_calls = _recording_source(catalog, {}, metadata)

    with pytest.raises(RecoverableToolError) as error:
        await run_energy_query(validate_energy_args(args), source)

    assert error.value.key == "energy_query_too_large"
    assert error.value.placeholders == expected_placeholders
    assert metadata_calls == []
    assert statistics_calls == []


@pytest.mark.parametrize(
    ("section", "source_type", "role", "first_unit", "second_unit", "expected"),
    [
        pytest.param(
            "gas",
            "gas",
            "gas_consumption",
            "m³",
            "ft³",
            [{"value": 5.0, "unit": "ft³"}, {"value": 2.0, "unit": "m³"}],
            id="gas",
        ),
        pytest.param(
            "water",
            "water",
            "water_consumption",
            "m³",
            "gal",
            [{"value": 5.0, "unit": "gal"}, {"value": 2.0, "unit": "m³"}],
            id="water",
        ),
        pytest.param(
            "cost",
            "grid",
            "cost",
            "EUR",
            "USD",
            [{"value": 2.0, "unit": "EUR"}, {"value": 5.0, "unit": "USD"}],
            id="cost",
        ),
        pytest.param(
            "compensation",
            "grid",
            "compensation",
            "EUR",
            "USD",
            [{"value": 2.0, "unit": "EUR"}, {"value": 5.0, "unit": "USD"}],
            id="compensation",
        ),
    ],
)
async def test_summary_preserves_distinct_native_unit_groups(
    section: Literal["gas", "water", "cost", "compensation"],
    source_type: Literal["grid", "gas", "water"],
    role: Literal["gas_consumption", "water_consumption", "cost", "compensation"],
    first_unit: str,
    second_unit: str,
    expected: list[dict[str, object]],
) -> None:
    """Every non-electric summary section retains distinct native unit groups."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                f"{source_type}:0",
                source_type,
                "First source",
                (SafeEnergyMeasureRef(role, "sensor.first"),),
            ),
            SafeEnergySourceRecord(
                f"{source_type}:1",
                source_type,
                "Second source",
                (SafeEnergyMeasureRef(role, "sensor.second"),),
            ),
        ),
        (),
        (),
        None,
        2,
    )
    metadata = {
        "sensor.first": {"has_sum": True, "unit_of_measurement": first_unit},
        "sensor.second": {"has_sum": True, "unit_of_measurement": second_unit},
    }
    result = await run_energy_query(
        validate_energy_args({"hours": 24, "period": "hour", "include": ["summary"]}),
        _source(
            catalog,
            {
                "sensor.first": _rows("sensor.first", [2.0]),
                "sensor.second": _rows("sensor.second", [5.0]),
            },
            metadata,
            now=_START + timedelta(days=1),
        ),
    )

    assert cast(dict[str, object], result["summary"])[section] == expected


def test_validation_maps_public_and_soc_locators_in_stable_order() -> None:
    """Public statistics and explicit SOC IDs map to stable safe locators."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "battery:0",
                "battery",
                "Battery",
                (
                    SafeEnergyMeasureRef(
                        "battery_discharge",
                        "sensor.battery_energy",
                        SafeEnergyCurrentPrice(0.2, "USD/kWh", "entity"),
                    ),
                ),
                state_of_charge_statistic_id="sensor.battery_soc",
            ),
        ),
        (),
        (),
        None,
        1,
    )
    public = _validation_public_locators(catalog)
    private = _validation_private_locators(catalog, {})

    assert _sanitize_validation(
        {
            "energy_sources": [
                [],
                [
                    {
                        "type": "invalid_state",
                        "affected_entities": {
                            ("sensor.battery_soc", None),
                            ("sensor.battery_energy", None),
                        },
                    }
                ],
            ]
        },
        public,
        private,
        {},
    ) == (
        {
            "type": "invalid_state",
            "affected": [
                {"role": "battery_discharge", "statistic_id": "sensor.battery_energy"},
                {"role": "state_of_charge", "source_id": "battery:0"},
            ],
        },
    )


@pytest.mark.parametrize(
    ("raw_id", "extra_private", "price_locators", "expected_locator"),
    [
        pytest.param(
            "sensor.battery_price",
            {},
            {0: {"sensor.battery_price": {"role": "current_price", "source_id": "battery:0"}}},
            {"role": "current_price", "source_id": "battery:0"},
            id="price-sidecar",
        ),
        pytest.param(
            "sensor.battery_soc",
            {},
            {},
            {"role": "state_of_charge", "source_id": "battery:0"},
            id="state-of-charge",
        ),
        pytest.param(
            "sensor.battery_rate",
            {},
            {},
            {"role": "rate", "source_id": "battery:0"},
            id="source-rate",
        ),
        pytest.param(
            "sensor.device_rate",
            {},
            {},
            {"role": "rate", "device_statistic_id": "sensor.device_energy"},
            id="device-rate",
        ),
    ],
)
def test_validation_maps_explicit_private_ids_to_only_safe_locators(
    raw_id: str,
    extra_private: dict[str, dict[str, str]],
    price_locators: dict[int, dict[str, dict[str, str]]],
    expected_locator: dict[str, str],
) -> None:
    """An explicit private mapping discloses only the public-safe locator."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "battery:0",
                "battery",
                "Battery",
                (SafeEnergyMeasureRef("battery_discharge", "sensor.battery_energy"),),
                rate_statistic_id="sensor.battery_rate",
                state_of_charge_statistic_id="sensor.battery_soc",
            ),
        ),
        (SafeEnergyDeviceRecord("device", "Device", "sensor.device_energy", rate_statistic_id="sensor.device_rate"),),
        (),
        None,
        2,
    )
    public = _validation_public_locators(catalog)
    private = {**_validation_private_locators(catalog, {}), **extra_private}

    assert _sanitize_validation(
        {
            "energy_sources": [
                [
                    {
                        "type": "private_issue",
                        "affected_entities": [(raw_id, None)],
                    }
                ]
            ]
        },
        public,
        private,
        price_locators,
    ) == ({"type": "private_issue", "affected": [expected_locator]},)


@pytest.mark.parametrize(
    "affected_entities",
    [
        pytest.param([], id="empty"),
        pytest.param([("sensor.unmapped_price", None)], id="unmapped"),
    ],
)
def test_validation_discards_price_issues_without_a_safe_locator(
    affected_entities: list[tuple[str, None]],
) -> None:
    """Missing or unknown price IDs cannot be guessed from source position."""
    catalog = _catalog(prices=True, devices=False)

    assert (
        _sanitize_validation(
            {
                "energy_sources": [
                    [
                        {
                            "type": "price_issue",
                            "affected_entities": affected_entities,
                        }
                    ]
                ]
            },
            _validation_public_locators(catalog),
            _validation_private_locators(catalog, {}),
            {},
        )
        == ()
    )


@pytest.mark.parametrize(
    (
        "price_states",
        "energy_sources",
        "raw_price_ids",
        "expected_prices",
        "expected_price_locators",
        "raw_validation",
        "expected_validation",
    ),
    [
        pytest.param(
            (
                ("sensor.import_price", "0.31", "USD/kWh"),
                ("sensor.export_price", "0.08", "USD/kWh"),
            ),
            [
                {
                    "type": "grid",
                    "stat_energy_from": "sensor.grid_import",
                    "stat_energy_to": "sensor.grid_export",
                    "entity_energy_price": "sensor.import_price",
                    "entity_energy_price_export": "sensor.export_price",
                }
            ],
            ("sensor.import_price", "sensor.export_price"),
            {
                "grid:0": {
                    "grid_import": SafeEnergyCurrentPrice(0.31, "USD/kWh", "entity"),
                    "grid_export": SafeEnergyCurrentPrice(0.08, "USD/kWh", "entity"),
                }
            },
            {
                0: {
                    "sensor.import_price": {"role": "current_price", "source_id": "grid:0"},
                    "sensor.export_price": {"role": "current_price", "source_id": "grid:0"},
                }
            },
            {
                "energy_sources": [
                    [
                        {
                            "type": "invalid_price",
                            "affected_entities": [
                                ("sensor.import_price", None),
                                ("sensor.export_price", None),
                            ],
                        }
                    ]
                ]
            },
            (
                {
                    "type": "invalid_price",
                    "affected": [{"role": "current_price", "source_id": "grid:0"}],
                },
            ),
            id="visible-import-and-export",
        ),
        pytest.param(
            (("sensor.nonnumeric_price", "unavailable", "USD/kWh"),),
            [
                {
                    "type": "grid",
                    "stat_energy_from": "sensor.grid_import",
                    "entity_energy_price": "sensor.nonnumeric_price",
                }
            ],
            ("sensor.nonnumeric_price",),
            {"grid:0": {"grid_import": None}},
            {0: {"sensor.nonnumeric_price": {"role": "current_price", "source_id": "grid:0"}}},
            {
                "energy_sources": [
                    [
                        {
                            "type": "invalid_price",
                            "affected_entities": [("sensor.nonnumeric_price", None)],
                        }
                    ]
                ]
            },
            ({"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "grid:0"}]},),
            id="visible-nonnumeric",
        ),
        pytest.param(
            (),
            [
                {
                    "type": "grid",
                    "stat_energy_from": "sensor.grid_import",
                    "entity_energy_price": "sensor.hidden_price",
                }
            ],
            ("sensor.hidden_price",),
            {"grid:0": {"grid_import": None}},
            {},
            {
                "energy_sources": [
                    [
                        {
                            "type": "invalid_price",
                            "affected_entities": [("sensor.hidden_price", None)],
                        }
                    ]
                ]
            },
            (),
            id="hidden-price",
        ),
        pytest.param(
            (("sensor.shared_price", "0.19", "USD/kWh"),),
            [
                {
                    "type": "grid",
                    "stat_energy_from": "sensor.grid_import",
                    "entity_energy_price": "sensor.shared_price",
                },
                {
                    "type": "gas",
                    "stat_energy_from": "sensor.gas_consumption",
                    "entity_energy_price": "sensor.shared_price",
                },
            ],
            ("sensor.shared_price",),
            {
                "grid:0": {"grid_import": SafeEnergyCurrentPrice(0.19, "USD/kWh", "entity")},
                "gas:0": {"gas_consumption": SafeEnergyCurrentPrice(0.19, "USD/kWh", "entity")},
            },
            {
                0: {"sensor.shared_price": {"role": "current_price", "source_id": "grid:0"}},
                1: {"sensor.shared_price": {"role": "current_price", "source_id": "gas:0"}},
            },
            {
                "energy_sources": [
                    [
                        {
                            "type": "invalid_price",
                            "affected_entities": [("sensor.shared_price", None)],
                        }
                    ],
                    [
                        {
                            "type": "invalid_price",
                            "affected_entities": [("sensor.shared_price", None)],
                        }
                    ],
                ]
            },
            (
                {"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "grid:0"}]},
                {"type": "invalid_price", "affected": [{"role": "current_price", "source_id": "gas:0"}]},
            ),
            id="shared-visible-price",
        ),
    ],
)
def test_price_validation_sidecar_preserves_safe_source_correlation(
    price_states: tuple[tuple[str, str, str], ...],
    energy_sources: list[dict[str, object]],
    raw_price_ids: tuple[str, ...],
    expected_prices: dict[str, dict[str, SafeEnergyCurrentPrice | None]],
    expected_price_locators: dict[int, dict[str, dict[str, str]]],
    raw_validation: dict[str, object],
    expected_validation: tuple[dict[str, object], ...],
) -> None:
    """Visible price validation maps to safe source IDs without retaining raw IDs."""
    snapshot = _snapshot_with_states(
        _state("sensor.grid_import", "1"),
        _state("sensor.grid_export", "1"),
        _state("sensor.gas_consumption", "1", "m³"),
        *(_state(entity_id, value, unit) for entity_id, value, unit in price_states),
    )
    price_locators: dict[int, dict[str, dict[str, str]]] = {}
    catalog, forecasts = sanitize_energy_preferences(
        snapshot,
        {"energy_sources": energy_sources},
        {},
        validation_price_locators=price_locators,
    )

    prices = {
        source.source_id: {
            measure.role: measure.current_price
            for measure in source.measures
            if measure.role in {"grid_import", "grid_export", "gas_consumption"}
        }
        for source in catalog.sources
    }
    validation = _sanitize_validation(
        raw_validation,
        _validation_public_locators(catalog),
        _validation_private_locators(catalog, forecasts),
        price_locators,
    )
    serialized = json.dumps({"catalog": catalog, "validation": validation}, default=str)

    assert prices == expected_prices
    assert price_locators == expected_price_locators
    assert validation == expected_validation
    assert all(raw_price_id not in serialized for raw_price_id in raw_price_ids)


@pytest.mark.parametrize(
    ("args", "expected_validation"),
    [
        pytest.param(
            {"source_types": ["solar"], "include": ["validation"]},
            ({"type": "out-of-scope", "affected": [{"role": "current_price", "source_id": "solar:0"}]},),
            id="source-types",
        ),
        pytest.param(
            {
                "source_types": ["device"],
                "device_statistic_ids": ["sensor.child_one"],
                "include": ["validation"],
            },
            (
                {
                    "type": "out-of-scope",
                    "affected": [{"role": "rate", "device_statistic_id": "sensor.child_one"}],
                },
            ),
            id="device-filter",
        ),
    ],
)
async def test_validation_omits_locators_outside_selected_energy_scope(
    args: dict[str, object], expected_validation: tuple[dict[str, object], ...]
) -> None:
    """Source and device filters also constrain private validation locators."""
    validation = (
        {
            "type": "out-of-scope",
            "affected": [
                {"role": "current_price", "source_id": "grid:0"},
                {"role": "current_price", "source_id": "solar:0"},
                {"role": "rate", "device_statistic_id": "sensor.parent"},
                {"role": "rate", "device_statistic_id": "sensor.child_one"},
            ],
        },
    )
    result = await run_energy_query(validate_energy_args(args), _source(_catalog(), {}, {}, validation=validation))

    assert result["validation"] == list(expected_validation)


def test_response_fitting_enforces_aggregate_historical_point_limit() -> None:
    """The hard historical budget applies even when the byte budget has room."""
    payload = {
        "sources": [
            {
                "measures": [
                    {"series": [[(_START + timedelta(hours=index)).isoformat(), float(index)] for index in range(501)]}
                ]
            }
        ]
    }

    fitted = fit_energy_result(payload, limit=1_000_000)
    series = cast(
        list[list[object]],
        cast(list[dict[str, object]], cast(list[dict[str, object]], fitted["sources"])[0]["measures"])[0]["series"],
    )
    assert len(series) == 500
    assert series[0][0] == (_START + timedelta(hours=1)).isoformat()
    assert fitted["overflow"] == {
        "truncated": True,
        "limit": 1_000_000,
        "omitted_series_points": 1,
        "omitted_forecast_points": 0,
    }


def test_response_fitting_truncation_removes_oldest_history_then_latest_forecast() -> None:
    """Response fitting keeps totals while trimming history before future points."""
    source_series = [
        ["2026-01-01T00:00:00+00:00", 1.0],
        ["2026-01-02T00:00:00+00:00", 2.0],
        ["2026-01-03T00:00:00+00:00", 3.0],
        ["2026-01-04T00:00:00+00:00", 4.0],
        ["2026-01-05T00:00:00+00:00", 5.0],
        ["2026-01-06T00:00:00+00:00", 6.0],
        ["2026-01-07T00:00:00+00:00", 7.0],
        ["2026-01-08T00:00:00+00:00", 8.0],
        ["2026-01-09T00:00:00+00:00", 9.0],
        ["2026-01-10T00:00:00+00:00", 10.0],
    ]
    device_series = [
        {"start": "2026-01-11T00:00:00+00:00", "inclusive_value": 11.0, "exclusive_value": 6.0},
        {"start": "2026-01-12T00:00:00+00:00", "inclusive_value": 12.0, "exclusive_value": 7.0},
    ]
    forecast_points = [
        ["2026-07-01T00:00:00+00:00", 100.0],
        ["2026-07-02T00:00:00+00:00", 200.0],
        ["2026-07-03T00:00:00+00:00", 300.0],
        ["2026-07-04T00:00:00+00:00", 400.0],
        ["2026-07-05T00:00:00+00:00", 500.0],
        ["2026-07-06T00:00:00+00:00", 600.0],
        ["2026-07-07T00:00:00+00:00", 700.0],
        ["2026-07-08T00:00:00+00:00", 800.0],
    ]

    def _payload(*, history: bool) -> dict[str, object]:
        return {
            "sources": [
                {
                    "source_id": "grid:0",
                    "measures": [
                        {
                            "role": "grid_import",
                            "statistic_id": "sensor.grid_import",
                            "value": 10.0,
                            "unit": "kWh",
                            "series": [point.copy() for point in source_series] if history else [],
                        }
                    ],
                }
            ],
            "devices": [
                {
                    "source_type": "device",
                    "name": "Dryer",
                    "statistic_id": "sensor.dryer",
                    "inclusive_value": 8.0,
                    "exclusive_value": 3.0,
                    "unit": "kWh",
                    "series": [point.copy() for point in device_series] if history else [],
                }
            ],
            "forecast": [
                {
                    "source_id": "solar:0",
                    "name": "Solar",
                    "unit": "Wh",
                    "total": 3600.0,
                    "points": [point.copy() for point in forecast_points],
                }
            ],
        }

    historical = fit_energy_result(_payload(history=True), limit=1040)
    historical_source = cast(list[dict[str, object]], historical["sources"])[0]
    historical_measure = cast(list[dict[str, object]], historical_source["measures"])[0]
    historical_device = cast(list[dict[str, object]], historical["devices"])[0]
    historical_forecast = cast(list[dict[str, object]], historical["forecast"])[0]

    assert historical_measure["value"] == 10.0
    assert historical_device["inclusive_value"] == 8.0
    assert historical_device["exclusive_value"] == 3.0
    assert historical_forecast["total"] == 3600.0
    assert historical_measure["series"] == source_series[7:]
    assert historical_device["series"] == device_series
    assert historical_forecast["points"] == forecast_points
    assert historical["overflow"] == {
        "truncated": True,
        "limit": 1040,
        "omitted_series_points": 7,
        "omitted_forecast_points": 0,
    }

    forecast = fit_energy_result(_payload(history=False), limit=600)
    forecast_source = cast(list[dict[str, object]], forecast["sources"])[0]
    forecast_measure = cast(list[dict[str, object]], forecast_source["measures"])[0]
    forecast_device = cast(list[dict[str, object]], forecast["devices"])[0]
    fitted_forecast = cast(list[dict[str, object]], forecast["forecast"])[0]

    assert forecast_measure["value"] == 10.0
    assert forecast_device["inclusive_value"] == 8.0
    assert forecast_device["exclusive_value"] == 3.0
    assert fitted_forecast["total"] == 3600.0
    assert fitted_forecast["points"] == forecast_points[:3]
    assert forecast["overflow"] == {
        "truncated": True,
        "limit": 600,
        "omitted_series_points": 0,
        "omitted_forecast_points": 5,
    }


async def test_direct_energy_tool_envelopes_recoverable_errors() -> None:
    """The direct surface returns a stable error envelope instead of leaking core errors."""
    result = await GetEnergyTool.run_query(
        validate_energy_args({"hours": 1}), _source(SafeEnergyCatalog((), (), (), None, 0), {}, {})
    )
    assert result["status"] == "error"
    assert cast(dict[str, object], result["error"])["key"] == "no_visible_energy_sources"


async def test_direct_energy_tool_rebuilds_source_across_availability_races(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Sequential direct calls rebuild their source and preserve stable race envelopes."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    tool = GetEnergyTool(mock_config_entry.entry_id)
    tool_input = llm.ToolInput(tool_name=TOOL_GET_ENERGY, tool_args={"hours": 1})
    build_source = AsyncMock(
        side_effect=[
            _source(SafeEnergyCatalog((), (), (), None, 1), {}, {}),
            RecoverableToolError("energy_not_configured", {}),
        ]
    )

    with patch(
        "custom_components.llm_sandbox.llm_api.tools.energy.async_build_energy_source",
        new=build_source,
    ):
        hidden_result = await tool.async_call(hass, tool_input, context)
        reset_result = await tool.async_call(hass, tool_input, context)

    assert hidden_result["status"] == "error"
    assert cast(dict[str, object], hidden_result["error"])["key"] == "no_visible_energy_sources"
    assert reset_result["status"] == "error"
    assert cast(dict[str, object], reset_result["error"])["key"] == "energy_not_configured"
    assert build_source.await_count == 2


async def test_battery_current_state_of_charge_has_percentage_shape() -> None:
    """A configured visible battery SOC is exposed as a percentage, not a raw state."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "battery:0",
                "battery",
                "Battery",
                (SafeEnergyMeasureRef("battery_discharge", "sensor.battery_discharge"),),
                state_of_charge_value=63.0,
                state_of_charge_statistic_id="sensor.battery_soc",
            ),
        ),
        (),
        (),
        None,
        1,
    )

    result = await run_energy_query(
        validate_energy_args({"hours": 1, "include": ["current"]}), _source(catalog, {}, {})
    )

    battery = cast(list[dict[str, object]], result["sources"])[0]
    assert battery["state_of_charge"] == {"value": 63.0, "unit": "%"}


async def test_carbon_result_reports_ambiguous_co2_signal() -> None:
    """Multiple visible CO₂ signals remain ambiguous at the public result boundary."""
    catalog, _ = sanitize_energy_preferences(
        _snapshot_with_states(
            _state("sensor.grid_import", "1"),
            _state("sensor.co2_alpha", "42", "%", platform="co2signal"),
            _state("sensor.co2_beta", "43", "%", platform="co2signal"),
        ),
        {
            "energy_sources": [
                {
                    "type": "grid",
                    "stat_energy_from": "sensor.grid_import",
                }
            ]
        },
        {},
    )

    result = await run_energy_query(
        validate_energy_args({"hours": 1, "include": ["carbon"]}), _source(catalog, {}, {})
    )

    assert result["carbon"] == {"available": False, "reason": "ambiguous"}


async def test_carbon_result_reports_unusable_or_missing_co2_metadata() -> None:
    """Wrong-unit and absent CO₂ metadata have their stable public reasons."""
    grid = SafeEnergySourceRecord(
        "grid:0",
        "grid",
        "Grid",
        (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
    )
    wrong_unit_catalog = SafeEnergyCatalog((grid,), (), (), "sensor.co2", 1)
    missing_catalog = SafeEnergyCatalog((grid,), (), (), None, 1)
    query = {
        "start": _START.isoformat(),
        "end": (_START + timedelta(hours=1)).isoformat(),
        "period": "hour",
        "include": ["carbon"],
    }

    wrong_unit = await run_energy_query(
        validate_energy_args(query),
        _source(
            wrong_unit_catalog,
            {
                "sensor.grid_import": [{"start": _START.isoformat(), "change": 10.0}],
                "sensor.co2": [{"start": _START.isoformat(), "mean": 25.0}],
            },
            {
                "sensor.grid_import": _metadata("sensor.grid_import")["sensor.grid_import"],
                "sensor.co2": {"has_mean": True, "unit_of_measurement": "g/kWh"},
            },
        ),
    )
    missing = await run_energy_query(
        validate_energy_args(query), _source(missing_catalog, {}, _metadata("sensor.grid_import"))
    )

    assert wrong_unit["carbon"] == {"available": False, "reason": "metadata_unavailable"}
    assert missing["carbon"] == {"available": False, "reason": "not_configured"}


async def test_carbon_result_uses_full_fossil_fallback_for_missing_buckets() -> None:
    """Missing CO₂ buckets count as fully fossil while present buckets retain their percentage."""
    catalog = SafeEnergyCatalog(
        (
            SafeEnergySourceRecord(
                "grid:0",
                "grid",
                "Grid",
                (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
            ),
        ),
        (),
        (),
        "sensor.co2",
        1,
    )
    result = await run_energy_query(
        validate_energy_args(
            {
                "start": _START.isoformat(),
                "end": (_START + timedelta(hours=2)).isoformat(),
                "period": "hour",
                "include": ["carbon"],
            }
        ),
        _source(
            catalog,
            {
                "sensor.grid_import": [
                    {"start": _START.isoformat(), "change": 10.0},
                    {"start": (_START + timedelta(hours=1)).isoformat(), "change": 8.0},
                ],
                "sensor.co2": [{"start": _START.isoformat(), "mean": 25.0}],
            },
            {
                "sensor.grid_import": _metadata("sensor.grid_import")["sensor.grid_import"],
                "sensor.co2": {"has_mean": True, "unit_of_measurement": "%"},
            },
        ),
    )

    assert result["carbon"] == {
        "available": True,
        "value": 10.5,
        "unit": "kWh",
        "assumed_full_fossil_points": 1,
    }


@pytest.mark.parametrize(
    ("limit", "expected_key", "expected_placeholders"),
    [
        pytest.param(
            "statistic_ids",
            "energy_query_too_large",
            {"statistic_count": "41", "bucket_count": "1", "max_points": "40"},
            id="statistic-ids",
        ),
        pytest.param(
            "source_records",
            "energy_source_limit_exceeded",
            {"source_count": "101", "max_sources": "100"},
            id="source-records",
        ),
        pytest.param(
            "forecast_sources",
            "energy_source_limit_exceeded",
            {"source_count": "9", "max_sources": "8"},
            id="forecast-sources",
        ),
    ],
)
async def test_cardinality_limits_reject_before_every_host_fetcher(
    limit: Literal["statistic_ids", "source_records", "forecast_sources"],
    expected_key: str,
    expected_placeholders: dict[str, str],
) -> None:
    """Every cardinality guard rejects before metadata, recorder, forecast, or validation work."""
    if limit == "statistic_ids":
        devices = tuple(
            SafeEnergyDeviceRecord("device", f"Device {index}", f"sensor.device_{index:02}") for index in range(41)
        )
        catalog = SafeEnergyCatalog((), devices, (), None, len(devices))
        args: dict[str, object] = {
            "hours": 1,
            "period": "hour",
            "source_types": ["device"],
            "include": ["summary", "validation"],
        }
    elif limit == "source_records":
        catalog = _gas_catalog(101)
        args = {
            "hours": 1,
            "period": "hour",
            "source_types": ["gas"],
            "include": ["summary", "validation"],
        }
    else:
        sources = tuple(
            SafeEnergySourceRecord(
                f"solar:{index}",
                "solar",
                f"Solar {index}",
                (SafeEnergyMeasureRef("solar_generation", f"sensor.solar_{index:02}"),),
            )
            for index in range(9)
        )
        catalog = SafeEnergyCatalog(sources, (), (), None, len(sources))
        args = {
            "hours": 1,
            "period": "hour",
            "source_types": ["solar"],
            "include": ["forecast", "validation"],
        }

    recorded_source, metadata_calls, statistics_calls = _recording_source(catalog, {}, {})
    forecast_calls: list[tuple[str, ...]] = []
    validation_calls: list[bool] = []

    async def fetch_forecasts(source_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        forecast_calls.append(source_ids)
        return await recorded_source.fetch_forecasts(source_ids)

    async def fetch_validation() -> tuple[dict[str, object], ...]:
        validation_calls.append(True)
        return await recorded_source.fetch_validation()

    source = replace(recorded_source, fetch_forecasts=fetch_forecasts, fetch_validation=fetch_validation)

    with pytest.raises(RecoverableToolError) as error:
        await run_energy_query(validate_energy_args(args), source)

    assert error.value.key == expected_key
    assert error.value.placeholders == expected_placeholders
    assert metadata_calls == []
    assert statistics_calls == []
    assert forecast_calls == []
    assert validation_calls == []


async def test_direct_energy_tool_unconfigured_preferences_returns_not_configured(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """A live Energy domain without saved preferences returns the stable setup key."""
    from homeassistant.components.energy.const import DOMAIN as energy_domain

    hass.data[energy_domain] = {}

    result = await _call_direct_energy_tool(hass, recorder_entry)

    assert result["status"] == "error"
    assert cast(dict[str, object], result["error"])["key"] == "energy_not_configured"


async def test_direct_energy_tool_missing_recorder_runtime_returns_unavailable(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The real Energy source builder preserves the recorder availability boundary."""
    from homeassistant.components.energy.const import DOMAIN as energy_domain

    hass.data[energy_domain] = {}

    result = await _call_direct_energy_tool(hass, loaded_entry)

    assert result["status"] == "error"
    assert cast(dict[str, object], result["error"])["key"] == "recorder_unavailable"


async def test_direct_energy_tool_all_hidden_external_catalog_returns_no_visible_sources(
    hass: HomeAssistant,
    recorder_entry: MockConfigEntry,
) -> None:
    """Configured external statistics cannot bypass the fresh snapshot visibility boundary."""
    from homeassistant.components.energy.const import DOMAIN as energy_domain
    from homeassistant.components.energy.data import async_get_manager

    hass.data[energy_domain] = {"cost_sensors": {}}
    manager = await async_get_manager(hass)
    await manager.async_update(
        {
            "energy_sources": [
                {
                    "type": "solar",
                    "stat_energy_from": "sensor.external_energy",
                    "config_entry_solar_forecast": None,
                }
            ],
            "device_consumption": [],
        }
    )

    result = await _call_direct_energy_tool(hass, recorder_entry)

    assert result["status"] == "error"
    assert cast(dict[str, object], result["error"])["key"] == "no_visible_energy_sources"


async def _call_direct_energy_tool(
    hass: HomeAssistant,
    entry: MockConfigEntry,
) -> dict[str, object]:
    """Call the direct Energy surface without replacing its production source builder."""
    result = await GetEnergyTool(entry.entry_id).async_call(
        hass,
        llm.ToolInput(tool_name=TOOL_GET_ENERGY, tool_args={"hours": 1}),
        llm.LLMContext(
            platform="test",
            context=Context(),
            language="en",
            assistant=None,
            device_id=None,
        ),
    )
    return cast(dict[str, object], result)
