"""Frozen real-home fixture built from the committed home-assistant-prod dataset."""

from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
from typing import NotRequired, TypedDict, cast

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
    ServiceTargetBrief,
    SnapshotIndexes,
)
from homeassistant.core import SupportsResponse

NAME: str = "home_real"
CREATED_AT: str = "2026-06-29T12:00:00+00:00"
CREATED_AT_TIMESTAMP: float = datetime.fromisoformat(CREATED_AT).timestamp()
ENTRY_ID: str = "entry_real"

type RecorderData = dict[str, object]
type JsonAttributes = dict[str, object]


class UnitData(TypedDict):
    """Committed unit-system fields from the real-home dataset."""

    temperature_unit: str
    length_unit: str
    mass_unit: str
    pressure_unit: str
    volume_unit: str
    area_unit: str
    wind_speed_unit: str
    accumulated_precipitation_unit: str


class ConfigData(TypedDict):
    """Committed Home Assistant config fields from the real-home dataset."""

    location_name: str
    latitude: float
    longitude: float
    elevation: int
    time_zone: str
    language: str
    country: str
    currency: str
    units: UnitData


class FloorData(TypedDict):
    """Committed floor registry fields from the real-home dataset."""

    floor_id: str
    name: str
    level: int
    aliases: NotRequired[list[str]]


class AreaData(TypedDict):
    """Committed area registry fields from the real-home dataset."""

    area_id: str
    name: str
    floor_id: str


class DeviceData(TypedDict):
    """Committed device registry fields from the real-home dataset."""

    id: str
    name: str
    area_id: str
    manufacturer: str
    model: str
    identifiers: list[list[str]]


class EntityData(TypedDict):
    """Committed entity/state fields from the real-home dataset."""

    entity_id: str
    name: str
    platform: str
    device_id: str | None
    area_id: str | None
    state: str
    attributes: JsonAttributes


class HomeRealData(TypedDict):
    """Top-level committed real-home dataset shape."""

    config: ConfigData
    floors: list[FloorData]
    areas: list[AreaData]
    devices: list[DeviceData]
    entities: list[EntityData]
    services: dict[str, list[str]]


_DATA_PATH = Path(__file__).with_name("home_real_data.json")
_DATA: HomeRealData = cast(HomeRealData, json.loads(_DATA_PATH.read_text(encoding="utf-8")))


def snapshot() -> HomeSnapshot:
    """Return a fresh frozen real-home snapshot."""
    floors = {floor["floor_id"]: _floor(floor) for floor in _DATA["floors"]}
    areas = {area["area_id"]: _area(area) for area in _DATA["areas"]}
    devices = {device["id"]: _device(device) for device in _DATA["devices"]}
    entities = {entity["entity_id"]: _entity(entity, devices, areas) for entity in _DATA["entities"]}
    states = {entity["entity_id"]: _state(entity) for entity in _DATA["entities"]}
    states = enrich_states(states, entities, devices, areas)
    services = {domain: tuple(services) for domain, services in _DATA["services"].items()}
    return HomeSnapshot(
        created_at=CREATED_AT,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        config=_config(_DATA["config"]),
        services=services,
        services_supports_response=_services_supports_response(services),
        indexes=_indexes(entities, devices, areas, floors),
        labels={},
        categories={},
        issues=(),
        notifications=(),
        config_entries=(),
        services_schema={},
        services_target=_services_target(services),
    )


def recorder() -> RecorderData:
    """Return canned recorder rows for the real home."""
    return {
        "history": {
            "sensor.tempest_temperature": [
                {
                    "state": "78.1",
                    "attributes": {"unit_of_measurement": "°F"},
                    "last_changed": "2026-06-28T12:00:00+00:00",
                    "last_updated": "2026-06-28T12:00:00+00:00",
                },
                {
                    "state": "78.8",
                    "attributes": {"unit_of_measurement": "°F"},
                    "last_changed": "2026-06-29T00:00:00+00:00",
                    "last_updated": "2026-06-29T00:00:00+00:00",
                },
                {
                    "state": "77",
                    "attributes": {"unit_of_measurement": "°F"},
                    "last_changed": "2026-06-29T12:00:00+00:00",
                    "last_updated": "2026-06-29T12:00:00+00:00",
                },
            ]
        },
        "statistics": {
            "sensor.tempest_humidity": [
                {
                    "start": "2026-06-28T12:00:00+00:00",
                    "end": "2026-06-29T12:00:00+00:00",
                    "state": 93.66,
                    "mean": 92.5,
                    "min": 88.0,
                    "max": 95.0,
                    "sum": 0.0,
                }
            ]
        },
        "logbook": {
            "light.living_room_lights_group": [
                {
                    "when": "2026-06-29T10:30:00+00:00",
                    "name": "Living Room Lights Group",
                    "message": "turned off",
                }
            ]
        },
    }


def _config(config: ConfigData) -> SafeConfig:
    """Build the frozen config record from the real-home dataset."""
    units = config["units"]
    return SafeConfig(
        location_name=config["location_name"],
        latitude=config["latitude"],
        longitude=config["longitude"],
        elevation=config["elevation"],
        time_zone=config["time_zone"],
        language=config["language"],
        country=config["country"],
        currency=config["currency"],
        internal_url=None,
        external_url=None,
        units=SafeUnitSystem(
            temperature_unit=units["temperature_unit"],
            length_unit=units["length_unit"],
            mass_unit=units["mass_unit"],
            pressure_unit=units["pressure_unit"],
            volume_unit=units["volume_unit"],
            area_unit=units["area_unit"],
            wind_speed_unit=units["wind_speed_unit"],
            accumulated_precipitation_unit=units["accumulated_precipitation_unit"],
        ),
    )


def _state(entity: EntityData) -> SafeState:
    """Build a visible state record with the fixed eval timestamp."""
    entity_id = entity["entity_id"]
    domain, object_id = entity_id.split(".", 1)
    name = entity["name"]
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=entity["state"],
        attributes={"friendly_name": name, **entity["attributes"]},
        last_changed=CREATED_AT,
        last_changed_timestamp=CREATED_AT_TIMESTAMP,
        last_reported=CREATED_AT,
        last_reported_timestamp=CREATED_AT_TIMESTAMP,
        last_updated=CREATED_AT,
        last_updated_timestamp=CREATED_AT_TIMESTAMP,
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
    )


def _entity(
    entity: EntityData,
    devices: Mapping[str, SafeDeviceEntry],
    areas: Mapping[str, SafeAreaEntry],
) -> SafeRegistryEntry:
    """Build a frozen registry entry and verify referenced registry ids exist."""
    entity_id = entity["entity_id"]
    device_id = entity["device_id"]
    area_id = entity["area_id"]
    # Device references are part of the fixture contract and must resolve eagerly.
    if device_id is not None and device_id not in devices:
        raise ValueError(f"unknown device_id {device_id!r} for {entity_id}")
    # Entity area overrides are part of the fixture contract and must resolve eagerly.
    if area_id is not None and area_id not in areas:
        raise ValueError(f"unknown area_id {area_id!r} for {entity_id}")
    device_class = _string_attr(entity["attributes"], "device_class")
    return SafeRegistryEntry(
        entity_id=entity_id,
        domain=entity_id.split(".", 1)[0],
        unique_id=entity_id,
        platform=entity["platform"],
        config_entry_id=ENTRY_ID,
        device_id=device_id,
        area_id=area_id,
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
        supported_features=0,
        translation_key=None,
        has_entity_name=True,
    )


def _device(device: DeviceData) -> SafeDeviceEntry:
    """Build a frozen device registry entry from the real-home dataset."""
    device_id = device["id"]
    return SafeDeviceEntry(
        id=device_id,
        name=device["name"],
        name_by_user=None,
        manufacturer=device["manufacturer"],
        model=device["model"],
        model_id=None,
        sw_version=None,
        hw_version=None,
        serial_number=None,
        area_id=device["area_id"],
        labels=(),
        identifiers=tuple(tuple(identifier) for identifier in device["identifiers"]),
        connections=(),
        configuration_url=None,
        entry_type=None,
        config_entries=(ENTRY_ID,),
        via_device_id=None,
        disabled_by=None,
    )


def _area(area: AreaData) -> SafeAreaEntry:
    """Build a frozen area registry entry from the real-home dataset."""
    area_id = area["area_id"]
    return SafeAreaEntry(
        id=area_id,
        area_id=area_id,
        name=area["name"],
        aliases=(),
        floor_id=area["floor_id"],
        labels=(),
        icon=None,
        picture=None,
        humidity_entity_id=None,
        temperature_entity_id=None,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _floor(floor: FloorData) -> SafeFloorEntry:
    """Build a frozen floor registry entry from the real-home dataset."""
    floor_id = floor["floor_id"]
    return SafeFloorEntry(
        floor_id=floor_id,
        id=floor_id,
        name=floor["name"],
        aliases=tuple(floor.get("aliases", [])),
        level=floor["level"],
        icon=None,
        created_at=CREATED_AT,
        modified_at=CREATED_AT,
    )


def _services_supports_response(services: Mapping[str, tuple[str, ...]]) -> dict[str, dict[str, str]]:
    """Build service response metadata for every listed service."""
    return {
        domain: dict.fromkeys(domain_services, SupportsResponse.NONE.value)
        for domain, domain_services in services.items()
    }


def _services_target(
    services: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, ServiceTargetBrief]]:
    """Build service target metadata for entity-targeting domains in the real fixture."""
    entity_domains = {
        "alarm_control_panel",
        "climate",
        "cover",
        "fan",
        "light",
        "switch",
        "weather",
    }
    return {
        domain: {service: _domain_target(domain) for service in domain_services}
        for domain, domain_services in services.items()
        if domain in entity_domains
    }


def _domain_target(domain: str) -> ServiceTargetBrief:
    """Build service target metadata for one entity domain."""
    return {"entity": [{"domain": [domain]}]}


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
        # Config-entry membership is included for every real-home fixture entity.
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


def _string_attr(attributes: Mapping[str, object], key: str) -> str | None:
    """Return a string state attribute when present."""
    value = attributes.get(key)
    # Only string-valued attributes can populate registry string fields.
    if isinstance(value, str):
        return value
    return None
