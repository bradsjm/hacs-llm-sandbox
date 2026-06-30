"""Small frozen home fixture for simple state and action eval cases."""

from collections.abc import Mapping

from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeConfig,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)
from homeassistant.core import SupportsResponse

NAME: str = "home_minimal"
CREATED_AT: str = "2026-06-29T00:00:00+00:00"

type RecorderData = dict[str, object]
type StateRecord = tuple[str, str, str, str, dict[str, object]]
type EntityRecord = tuple[str, str, str | None, str | None, str | None, str | None]
type DeviceRecord = tuple[str, str, str | None]
type AreaRecord = tuple[str, str, str | None, str | None, str | None]

_STATES: tuple[StateRecord, ...] = (
    ("light.living", "on", "Living Light", "2026-06-29T12:00:00+00:00", {"brightness": 180}),
    ("light.kitchen", "off", "Kitchen Light", "2026-06-29T12:00:00+00:00", {}),
    (
        "sensor.living_temp",
        "23.4",
        "Living Temperature",
        "2026-06-29T12:00:00+00:00",
        {"device_class": "temperature", "unit_of_measurement": "°C"},
    ),
    ("switch.fan", "off", "Fan", "2026-06-29T12:00:00+00:00", {}),
)

_ENTITIES: tuple[EntityRecord, ...] = (
    ("light.living", "uid-light-living", "device_living", None, None, None),
    ("light.kitchen", "uid-light-kitchen", "device_living", None, None, None),
    ("sensor.living_temp", "uid-sensor-living-temp", "device_living", None, None, "temperature"),
    ("switch.fan", "uid-switch-fan", "device_living", None, None, None),
)

_DEVICES: tuple[DeviceRecord, ...] = (("device_living", "Living Room Hub", "area_living"),)
_AREAS: tuple[AreaRecord, ...] = (("area_living", "Living Room", None, "sensor.living_temp", None),)


def snapshot() -> HomeSnapshot:
    """Return a fresh frozen minimal home snapshot."""
    states = {
        entity_id: _state(entity_id, state, name, changed, attrs) for entity_id, state, name, changed, attrs in _STATES
    }
    entities = {
        entity_id: _entity(entity_id, unique_id, device_id, area_id, hidden_by, device_class)
        for entity_id, unique_id, device_id, area_id, hidden_by, device_class in _ENTITIES
    }
    devices = {device_id: _device(device_id, name, area_id) for device_id, name, area_id in _DEVICES}
    areas = {
        area_id: _area(area_id, name, floor_id, temperature_entity_id, humidity_entity_id)
        for area_id, name, floor_id, temperature_entity_id, humidity_entity_id in _AREAS
    }
    return HomeSnapshot(
        created_at=CREATED_AT,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors={},
        config=_config(),
        services={
            "light": ("turn_off", "turn_on"),
            "switch": ("toggle",),
        },
        services_supports_response={
            "light": {"turn_off": SupportsResponse.NONE.value, "turn_on": SupportsResponse.NONE.value},
            "switch": {"toggle": SupportsResponse.NONE.value},
        },
        indexes=_indexes(entities, devices, areas, floors={}),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema={},
    )


def recorder() -> RecorderData:
    """Return canned recorder rows for the minimal home."""
    return {
        "history": {
            "sensor.living_temp": [
                {
                    "state": "22.8",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-28T12:00:00+00:00",
                    "last_updated": "2026-06-28T12:00:00+00:00",
                },
                {
                    "state": "23.1",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T06:00:00+00:00",
                    "last_updated": "2026-06-29T06:00:00+00:00",
                },
                {
                    "state": "23.4",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T12:00:00+00:00",
                    "last_updated": "2026-06-29T12:00:00+00:00",
                },
            ]
        },
        "statistics": {},
        "logbook": {},
    }


def _config() -> SafeConfig:
    """Build a minimal frozen config record for snapshot helpers."""
    return SafeConfig(
        location_name="Home",
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
    hidden_by: str | None,
    device_class: str | None,
) -> SafeRegistryEntry:
    """Build a frozen registry entry matching the state entity."""
    domain, _ = entity_id.split(".", 1)
    return SafeRegistryEntry(
        entity_id=entity_id,
        unique_id=unique_id,
        platform=domain,
        config_entry_id="entry_minimal",
        device_id=device_id,
        area_id=area_id,
        name=None,
        original_name=None,
        aliases=(),
        labels=(),
        disabled_by=None,
        hidden_by=hidden_by,
        entity_category=None,
        device_class=device_class,
        original_device_class=device_class,
        capabilities=None,
        supported_features=0,
        translation_key=None,
        has_entity_name=True,
    )


def _device(device_id: str, name: str, area_id: str | None) -> SafeDeviceEntry:
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
        labels=(),
        identifiers=(("llm_sandbox_evals", device_id),),
        connections=(),
        configuration_url=None,
        entry_type=None,
        config_entries=("entry_minimal",),
        via_device_id=None,
        disabled_by=None,
    )


def _area(
    area_id: str,
    name: str,
    floor_id: str | None,
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
        labels=(),
        icon=None,
        picture=None,
        humidity_entity_id=humidity_entity_id,
        temperature_entity_id=temperature_entity_id,
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
    by_area_device: dict[str, list[str]] = {}
    for device in devices.values():
        # Area-to-device indexes contain only devices assigned to an area.
        if device.area_id is not None:
            by_area_device.setdefault(device.area_id, []).append(device.id)
    return SnapshotIndexes(
        entity_ids_by_device_id={key: tuple(sorted(value)) for key, value in by_device.items()},
        entity_ids_by_area_id={key: tuple(sorted(value)) for key, value in by_area.items()},
        device_ids_by_area_id={key: tuple(sorted(value)) for key, value in by_area_device.items()},
        entity_ids_by_config_entry_id={key: tuple(sorted(value)) for key, value in by_config.items()},
        entity_ids_by_label={},
        device_ids_by_label={},
        area_ids_by_floor_id={
            floor.floor_id: tuple(sorted(area.area_id for area in areas.values() if area.floor_id == floor.floor_id))
            for floor in floors.values()
        },
    )
