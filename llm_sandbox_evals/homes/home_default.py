"""Richer frozen home fixture for registry, recorder, action, and complex eval cases."""

from collections.abc import Mapping

from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeConfig,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeLabelEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)
from homeassistant.core import SupportsResponse

NAME: str = "home_default"
CREATED_AT: str = "2026-06-29T12:00:00+00:00"

type RecorderData = dict[str, object]
type StateRecord = tuple[str, str, str, str, dict[str, object]]
type EntityRecord = tuple[
    str,
    str,
    str | None,
    str | None,
    tuple[str, ...],
    str | None,
    str | None,
    str | None,
    str | None,
]
type DeviceRecord = tuple[str, str, str | None, tuple[str, ...]]
type AreaRecord = tuple[str, str, str | None, tuple[str, ...], str | None, str | None]
type FloorRecord = tuple[str, str, int | None]
type LabelRecord = tuple[str, str, str, str]

_STATES: tuple[StateRecord, ...] = (
    ("light.living", "on", "Living Room Light", "2026-06-29T12:00:00+00:00", {"brightness": 210}),
    ("light.bedroom", "off", "Bedroom Light", "2026-06-29T12:00:00+00:00", {}),
    ("light.office_desk", "on", "Office Desk Light", "2026-06-29T12:00:00+00:00", {"brightness": 120}),
    (
        "sensor.living_temp",
        "25.2",
        "Living Temperature",
        "2026-06-29T12:00:00+00:00",
        {"device_class": "temperature", "unit_of_measurement": "°C"},
    ),
    (
        "sensor.bedroom_humidity",
        "64",
        "Bedroom Humidity",
        "2026-06-29T12:00:00+00:00",
        {"device_class": "humidity", "unit_of_measurement": "%"},
    ),
    (
        "sensor.office_power",
        "130",
        "Office Power",
        "2026-06-29T12:00:00+00:00",
        {"device_class": "power", "unit_of_measurement": "W"},
    ),
    (
        "sensor.router_uptime",
        "7200",
        "Router Uptime",
        "2026-06-29T12:00:00+00:00",
        {"entity_category": "diagnostic", "unit_of_measurement": "s"},
    ),
    ("switch.dehumidifier", "off", "Bedroom Dehumidifier", "2026-06-29T12:00:00+00:00", {}),
    ("switch.garage_opener", "off", "Garage Door Opener", "2026-06-29T12:00:00+00:00", {}),
    ("fan.living_fan", "off", "Living Room Fan", "2026-06-29T12:00:00+00:00", {"percentage": 0}),
    (
        "climate.bedroom",
        "heat",
        "Bedroom Thermostat",
        "2026-06-29T12:00:00+00:00",
        {"current_temperature": 21.2, "temperature": 20.0},
    ),
)

_ENTITIES: tuple[EntityRecord, ...] = (
    ("light.living", "uid-light-living", "device_living_light", None, ("label_evening",), None, None, None, None),
    ("light.bedroom", "uid-light-bedroom", "device_bedroom_lamp", None, ("label_evening",), None, None, None, None),
    (
        "light.office_desk",
        "uid-light-office-desk",
        "device_office_desk",
        None,
        ("label_work",),
        None,
        None,
        None,
        None,
    ),
    (
        "sensor.living_temp",
        "uid-sensor-living-temp",
        "device_living_climate",
        None,
        ("label_climate",),
        None,
        None,
        "temperature",
        None,
    ),
    (
        "sensor.bedroom_humidity",
        "uid-sensor-bedroom-humidity",
        "device_bedroom_climate",
        None,
        ("label_climate",),
        None,
        None,
        "humidity",
        None,
    ),
    (
        "sensor.office_power",
        "uid-sensor-office-power",
        "device_office_desk",
        None,
        ("label_work",),
        None,
        None,
        "power",
        None,
    ),
    (
        "sensor.router_uptime",
        "uid-sensor-router-uptime",
        "device_router",
        "area_office",
        ("label_work",),
        None,
        "diagnostic",
        None,
        None,
    ),
    (
        "switch.dehumidifier",
        "uid-switch-dehumidifier",
        "device_dehumidifier",
        None,
        ("label_climate",),
        None,
        None,
        None,
        None,
    ),
    ("switch.garage_opener", "uid-switch-garage-opener", "device_garage", None, (), "integration", None, None, None),
    ("fan.living_fan", "uid-fan-living", "device_living_fan", None, ("label_climate",), None, None, None, None),
    (
        "climate.bedroom",
        "uid-climate-bedroom",
        "device_bedroom_climate",
        None,
        ("label_climate",),
        None,
        None,
        None,
        None,
    ),
)

_DEVICES: tuple[DeviceRecord, ...] = (
    ("device_assist_living", "Living Assist Satellite", "area_living", ("label_evening",)),
    ("device_living_light", "Living Light Controller", "area_living", ("label_evening",)),
    ("device_living_climate", "Living Climate Sensor", "area_living", ("label_climate",)),
    ("device_living_fan", "Living Fan", "area_living", ("label_climate",)),
    ("device_bedroom_lamp", "Bedroom Lamp", "area_bedroom", ("label_evening",)),
    ("device_bedroom_climate", "Bedroom Thermostat", "area_bedroom", ("label_climate",)),
    ("device_dehumidifier", "Bedroom Dehumidifier", "area_bedroom", ("label_climate",)),
    ("device_office_desk", "Office Desk", "area_office", ("label_work",)),
    ("device_router", "Office Router", "area_office", ("label_work",)),
    ("device_garage", "Garage Opener", None, ()),
)

_AREAS: tuple[AreaRecord, ...] = (
    ("area_living", "Living Room", "floor_main", ("label_evening",), "sensor.living_temp", None),
    ("area_bedroom", "Bedroom", "floor_upstairs", ("label_climate",), None, "sensor.bedroom_humidity"),
    ("area_office", "Office", "floor_upstairs", ("label_work",), None, None),
)
_FLOORS: tuple[FloorRecord, ...] = (("floor_main", "Main Floor", 1), ("floor_upstairs", "Upstairs", 2))
_LABELS: tuple[LabelRecord, ...] = (
    ("label_evening", "Evening", "evening", "Evening comfort controls"),
    ("label_climate", "Climate", "climate", "Climate and air quality devices"),
    ("label_work", "Work", "work", "Office and work devices"),
)


def snapshot() -> HomeSnapshot:
    """Return a fresh frozen richer home snapshot."""
    states = {
        entity_id: _state(entity_id, state, name, changed, attrs) for entity_id, state, name, changed, attrs in _STATES
    }
    entities = {
        entity_id: _entity(
            entity_id, unique_id, device_id, area_id, labels, hidden_by, category, device_class, original_device_class
        )
        for entity_id, unique_id, device_id, area_id, labels, hidden_by, category, device_class, original_device_class in _ENTITIES
    }
    devices = {device_id: _device(device_id, name, area_id, labels) for device_id, name, area_id, labels in _DEVICES}
    areas = {
        area_id: _area(area_id, name, floor_id, labels, temperature_entity_id, humidity_entity_id)
        for area_id, name, floor_id, labels, temperature_entity_id, humidity_entity_id in _AREAS
    }
    floors = {floor_id: _floor(floor_id, name, level) for floor_id, name, level in _FLOORS}
    labels = {
        label_id: _label(label_id, name, normalized_name, description)
        for label_id, name, normalized_name, description in _LABELS
    }
    return HomeSnapshot(
        created_at=CREATED_AT,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        config=_config(),
        services={
            "climate": ("set_temperature",),
            "fan": ("set_percentage",),
            "light": ("turn_off", "turn_on"),
            "switch": ("toggle",),
        },
        services_supports_response={
            "climate": {"set_temperature": SupportsResponse.NONE.value},
            "fan": {"set_percentage": SupportsResponse.NONE.value},
            "light": {"turn_off": SupportsResponse.NONE.value, "turn_on": SupportsResponse.NONE.value},
            "switch": {"toggle": SupportsResponse.NONE.value},
        },
        indexes=_indexes(entities, devices, areas, floors),
        labels=labels,
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema={},
    )


def recorder() -> RecorderData:
    """Return canned recorder rows for the richer home."""
    return {
        "history": {
            "sensor.living_temp": [
                {
                    "state": "24.4",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-28T12:00:00+00:00",
                    "last_updated": "2026-06-28T12:00:00+00:00",
                },
                {
                    "state": "24.9",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T00:00:00+00:00",
                    "last_updated": "2026-06-29T00:00:00+00:00",
                },
                {
                    "state": "25.2",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T12:00:00+00:00",
                    "last_updated": "2026-06-29T12:00:00+00:00",
                },
            ],
            "sensor.bedroom_humidity": [
                {
                    "state": "61",
                    "attributes": {"unit_of_measurement": "%"},
                    "last_changed": "2026-06-28T12:00:00+00:00",
                    "last_updated": "2026-06-28T12:00:00+00:00",
                },
                {
                    "state": "63",
                    "attributes": {"unit_of_measurement": "%"},
                    "last_changed": "2026-06-29T00:00:00+00:00",
                    "last_updated": "2026-06-29T00:00:00+00:00",
                },
                {
                    "state": "64",
                    "attributes": {"unit_of_measurement": "%"},
                    "last_changed": "2026-06-29T12:00:00+00:00",
                    "last_updated": "2026-06-29T12:00:00+00:00",
                },
            ],
        },
        "statistics": {
            "sensor.bedroom_humidity": [
                {
                    "start": "2026-06-28T12:00:00+00:00",
                    "end": "2026-06-28T13:00:00+00:00",
                    "state": 61.0,
                    "sum": 61.0,
                    "min": 60.0,
                    "max": 62.0,
                    "mean": 61.0,
                },
                {
                    "start": "2026-06-29T11:00:00+00:00",
                    "end": "2026-06-29T12:00:00+00:00",
                    "state": 64.0,
                    "sum": 64.0,
                    "min": 63.0,
                    "max": 65.0,
                    "mean": 64.0,
                },
            ]
        },
        "logbook": {
            "light.living": [
                {"when": "2026-06-29T08:00:00+00:00", "name": "Living Room Light", "message": "turned off"},
                {"when": "2026-06-29T11:30:00+00:00", "name": "Living Room Light", "message": "turned on"},
            ],
            "switch.dehumidifier": [
                {"when": "2026-06-29T09:15:00+00:00", "name": "Bedroom Dehumidifier", "message": "turned off"},
            ],
        },
    }


def _config() -> SafeConfig:
    """Build a minimal frozen config record for snapshot helpers."""
    return SafeConfig(
        location_name="Eval Home",
        latitude=0.0,
        longitude=0.0,
        elevation=0,
        time_zone="UTC",
        language="en",
        country=None,
        currency="USD",
        internal_url=None,
        external_url=None,
        units=SafeUnitSystem(
            temperature_unit="°C",
            length_unit="m",
            mass_unit="kg",
            pressure_unit="Pa",
            volume_unit="L",
            area_unit="m²",
            wind_speed_unit="m/s",
            accumulated_precipitation_unit="mm",
        ),
    )


def _state(entity_id: str, state: str, name: str, last_changed: str, attributes: dict[str, object]) -> SafeState:
    """Build a minimal visible state record for target validation."""
    domain, object_id = entity_id.split(".", 1)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=state,
        attributes={"friendly_name": name, **attributes},
        last_changed=last_changed,
        last_reported=last_changed,
        last_updated=last_changed,
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
    )


def _entity(
    entity_id: str,
    unique_id: str,
    device_id: str | None,
    area_id: str | None,
    labels: tuple[str, ...],
    hidden_by: str | None,
    entity_category: str | None,
    device_class: str | None,
    original_device_class: str | None,
) -> SafeRegistryEntry:
    """Build a frozen registry entry matching the state entity."""
    domain, _ = entity_id.split(".", 1)
    return SafeRegistryEntry(
        entity_id=entity_id,
        unique_id=unique_id,
        platform=domain,
        config_entry_id="entry_default",
        device_id=device_id,
        area_id=area_id,
        name=None,
        original_name=None,
        aliases=(),
        labels=labels,
        disabled_by=None,
        hidden_by=hidden_by,
        entity_category=entity_category,
        device_class=device_class,
        original_device_class=original_device_class,
        capabilities=None,
        supported_features=0,
        translation_key=None,
        has_entity_name=True,
    )


def _device(device_id: str, name: str, area_id: str | None, labels: tuple[str, ...]) -> SafeDeviceEntry:
    """Build a frozen device registry entry."""
    return SafeDeviceEntry(
        id=device_id,
        name=name,
        name_by_user=None,
        manufacturer="Eval",
        model="Fixture",
        model_id=None,
        sw_version=None,
        hw_version=None,
        serial_number=None,
        area_id=area_id,
        labels=labels,
        identifiers=(("llm_sandbox_evals", device_id),),
        connections=(),
        configuration_url=None,
        entry_type=None,
        config_entries=("entry_default",),
        via_device_id=None,
        disabled_by=None,
    )


def _area(
    area_id: str,
    name: str,
    floor_id: str | None,
    labels: tuple[str, ...],
    temperature_entity_id: str | None,
    humidity_entity_id: str | None,
) -> SafeAreaEntry:
    """Build a frozen area registry entry."""
    return SafeAreaEntry(
        id=area_id,
        area_id=area_id,
        name=name,
        aliases=(),
        floor_id=floor_id,
        labels=labels,
        icon=None,
        picture=None,
        humidity_entity_id=humidity_entity_id,
        temperature_entity_id=temperature_entity_id,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _floor(floor_id: str, name: str, level: int | None) -> SafeFloorEntry:
    """Build a frozen floor registry entry."""
    return SafeFloorEntry(
        floor_id=floor_id,
        id=floor_id,
        name=name,
        aliases=(),
        level=level,
        icon=None,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _label(label_id: str, name: str, normalized_name: str, description: str) -> SafeLabelEntry:
    """Build a frozen label registry entry."""
    return SafeLabelEntry(
        label_id=label_id,
        name=name,
        normalized_name=normalized_name,
        description=description,
        color=None,
        icon=None,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _indexes(
    entities: Mapping[str, SafeRegistryEntry],
    devices: Mapping[str, SafeDeviceEntry],
    areas: Mapping[str, SafeAreaEntry],
    floors: Mapping[str, SafeFloorEntry],
) -> SnapshotIndexes:
    """Build sorted tuple indexes using Home Assistant's effective-area rule."""
    by_device: dict[str, list[str]] = {}
    by_area: dict[str, list[str]] = {}
    by_config: dict[str, list[str]] = {}
    by_label: dict[str, list[str]] = {}
    for entity in entities.values():
        # Device membership is present only for registry-backed entities.
        if entity.device_id is not None:
            by_device.setdefault(entity.device_id, []).append(entity.entity_id)
        effective_area_id = entity.area_id or (devices[entity.device_id].area_id if entity.device_id else None)
        # Effective area mirrors production: entity override wins, otherwise device area.
        if effective_area_id is not None:
            by_area.setdefault(effective_area_id, []).append(entity.entity_id)
        # Config-entry membership is included when fixtures provide an entry id.
        if entity.config_entry_id is not None:
            by_config.setdefault(entity.config_entry_id, []).append(entity.entity_id)
        for label_id in entity.labels:
            by_label.setdefault(label_id, []).append(entity.entity_id)
    by_area_device: dict[str, list[str]] = {}
    by_device_label: dict[str, list[str]] = {}
    for device in devices.values():
        # Area-to-device indexes contain only devices assigned to an area.
        if device.area_id is not None:
            by_area_device.setdefault(device.area_id, []).append(device.id)
        for label_id in device.labels:
            by_device_label.setdefault(label_id, []).append(device.id)
    return SnapshotIndexes(
        entity_ids_by_device_id={key: tuple(sorted(value)) for key, value in by_device.items()},
        entity_ids_by_area_id={key: tuple(sorted(value)) for key, value in by_area.items()},
        device_ids_by_area_id={key: tuple(sorted(value)) for key, value in by_area_device.items()},
        entity_ids_by_config_entry_id={key: tuple(sorted(value)) for key, value in by_config.items()},
        entity_ids_by_label={key: tuple(sorted(value)) for key, value in by_label.items()},
        device_ids_by_label={key: tuple(sorted(value)) for key, value in by_device_label.items()},
        area_ids_by_floor_id={
            floor.floor_id: tuple(sorted(area.area_id for area in areas.values() if area.floor_id == floor.floor_id))
            for floor in floors.values()
        },
    )
