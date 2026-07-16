"""Small frozen cover-capability fixture for eval cases."""

from collections.abc import Mapping
from datetime import datetime

from custom_components.llm_sandbox.snapshot.builder import enrich_states
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

NAME: str = "home_cover"
CREATED_AT: str = "2026-06-29T12:00:00+00:00"

type RecorderData = dict[str, object]
type StateRecord = tuple[str, str, str, dict[str, object]]
type EntityRecord = tuple[str, str, str, str]
type DeviceRecord = tuple[str, str, str]
type AreaRecord = tuple[str, str]

_STATES: tuple[StateRecord, ...] = (
    (
        "cover.office_blinds",
        "closed",
        "Office Blinds",
        {"current_position": 0, "device_class": "blind", "supported_features": 7},
    ),
    (
        "cover.bedroom_shade",
        "open",
        "Bedroom Shade",
        {"current_position": 100, "device_class": "shade", "supported_features": 7},
    ),
)
_ENTITIES: tuple[EntityRecord, ...] = (
    ("cover.office_blinds", "uid-cover-office-blinds", "device_office_blinds", "blind"),
    ("cover.bedroom_shade", "uid-cover-bedroom-shade", "device_bedroom_shade", "shade"),
)
_DEVICES: tuple[DeviceRecord, ...] = (
    ("device_office_blinds", "Office Blinds", "area_office"),
    ("device_bedroom_shade", "Bedroom Shade", "area_bedroom"),
)
_AREAS: tuple[AreaRecord, ...] = (
    ("area_office", "Office"),
    ("area_bedroom", "Bedroom"),
)


def snapshot() -> HomeSnapshot:
    """Return a fresh frozen cover-capability snapshot."""
    states = {
        entity_id: _state(entity_id, state, name, attributes)
        for entity_id, state, name, attributes in _STATES
    }
    entities = {
        entity_id: _entity(entity_id, unique_id, device_id, device_class)
        for entity_id, unique_id, device_id, device_class in _ENTITIES
    }
    devices = {device_id: _device(device_id, name, area_id) for device_id, name, area_id in _DEVICES}
    areas = {area_id: _area(area_id, name) for area_id, name in _AREAS}
    states = enrich_states(states, entities, devices, areas)
    return HomeSnapshot(
        created_at=CREATED_AT,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors={},
        config=_config(),
        services={"cover": ("close_cover", "open_cover", "set_cover_position")},
        services_supports_response={
            "cover": {
                "close_cover": SupportsResponse.NONE.value,
                "open_cover": SupportsResponse.NONE.value,
                "set_cover_position": SupportsResponse.NONE.value,
            }
        },
        indexes=_indexes(entities, devices, areas, floors={}),
        labels={},
        categories={},
        issues=(),
        notifications=(),
        config_entries=(),
        services_schema={},
    )


def recorder() -> RecorderData:
    """Return empty recorder rows for the cover fixture."""
    return {"history": {}, "statistics": {}, "logbook": {}}


def _config() -> SafeConfig:
    """Build the frozen Home Assistant config record."""
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


def _state(entity_id: str, state: str, name: str, attributes: dict[str, object]) -> SafeState:
    """Build one visible frozen cover state."""
    domain, object_id = entity_id.split(".", 1)
    timestamp = datetime.fromisoformat(CREATED_AT).timestamp()
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=state,
        attributes={"friendly_name": name, **attributes},
        last_changed=CREATED_AT,
        last_changed_timestamp=timestamp,
        last_reported=CREATED_AT,
        last_reported_timestamp=timestamp,
        last_updated=CREATED_AT,
        last_updated_timestamp=timestamp,
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
    )


def _entity(entity_id: str, unique_id: str, device_id: str, device_class: str) -> SafeRegistryEntry:
    """Build one registry-backed cover entity."""
    domain, _ = entity_id.split(".", 1)
    return SafeRegistryEntry(
        entity_id=entity_id,
        domain=domain,
        unique_id=unique_id,
        platform=domain,
        config_entry_id="entry_cover",
        device_id=device_id,
        area_id=None,
        name=None,
        original_name=None,
        aliases=(),
        labels=(),
        disabled_by=None,
        hidden_by=None,
        entity_category=None,
        device_class=device_class,
        original_device_class=device_class,
        capabilities=None,
        supported_features=7,
        translation_key=None,
        has_entity_name=True,
    )


def _device(device_id: str, name: str, area_id: str) -> SafeDeviceEntry:
    """Build one same-named cover device."""
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
        config_entries=("entry_cover",),
        via_device_id=None,
        disabled_by=None,
    )


def _area(area_id: str, name: str) -> SafeAreaEntry:
    """Build one frozen area registry record."""
    return SafeAreaEntry(
        id=area_id,
        area_id=area_id,
        name=name,
        aliases=(),
        floor_id=None,
        labels=(),
        icon=None,
        picture=None,
        humidity_entity_id=None,
        temperature_entity_id=None,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _indexes(
    entities: Mapping[str, SafeRegistryEntry],
    devices: Mapping[str, SafeDeviceEntry],
    areas: Mapping[str, SafeAreaEntry],
    floors: Mapping[str, SafeFloorEntry],
) -> SnapshotIndexes:
    """Build sorted indexes using Home Assistant's effective-area rule."""
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
        # Config-entry membership is included because every cover has the fixture entry id.
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
