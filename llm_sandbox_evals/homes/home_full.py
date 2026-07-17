"""Full frozen home fixture for inventory-scale discovery eval cases."""

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
    SafeLabelEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)
from homeassistant.core import SupportsResponse

NAME: str = "home_full"
CREATED_AT: str = "2026-06-29T12:00:00+00:00"
_LAST_CHANGED_OVERRIDES: Mapping[str, str] = {
    "light.living_room_ceiling": "2026-06-29T11:45:00+00:00",
    "light.living_room_accent": "2026-06-29T11:50:00+00:00",
    "switch.hallway_outlet": "2026-06-29T09:00:00+00:00",
}
_BASEMENT_CEILING_LIGHTS_OFF: frozenset[str] = frozenset(
    {
        "light.utility_room_ceiling",
        "light.storage_room_ceiling",
        "light.workshop_ceiling",
        "light.wine_cellar_ceiling",
        "light.home_gym_ceiling",
        "light.media_room_ceiling",
        "light.laundry_room_ceiling",
        "light.basement_bathroom_ceiling",
        "light.playroom_ceiling",
        "light.wine_tasting_room_ceiling",
        "light.basement_hallway_ceiling",
        "light.server_room_ceiling",
    }
)

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

_FLOOR_COUNT: int = 4

_LABELS: tuple[LabelRecord, ...] = (
    ("label_evening", "Evening", "evening", "Evening lighting and comfort controls"),
    ("label_climate", "Climate", "climate", "Temperature and humidity equipment"),
    ("label_security", "Security", "security", "Motion and occupancy sensors"),
    ("label_energy", "Energy", "energy", "Energy monitoring devices"),
    ("label_guest", "Guest", "guest", "Guest-accessible rooms and devices"),
)

_FLOORS: tuple[FloorRecord, ...] = (
    ("floor_basement", "Basement", -1),
    ("floor_ground", "Ground Floor", 0),
    ("floor_first", "First Floor", 1),
    ("floor_second", "Second Floor", 2),
)

_ROOM_SLUGS: tuple[tuple[str, ...], ...] = (
    (
        "utility_room",
        "storage_room",
        "workshop",
        "wine_cellar",
        "home_gym",
        "media_room",
        "laundry_room",
        "basement_bathroom",
        "playroom",
        "wine_tasting_room",
        "basement_hallway",
        "server_room",
    ),
    (
        "kitchen",
        "dining_room",
        "family_room",
        "living_room",
        "guest_bathroom",
        "den",
        "garage",
        "pantry",
        "sunroom",
    ),
    (
        "primary_bedroom",
        "primary_bathroom",
        "bedroom_1",
        "bedroom_2",
        "main_bathroom",
        "hallway",
        "nursery",
        "home_office",
        "walk_in_closet",
    ),
    (
        "attic",
        "loft",
        "teen_bedroom",
        "guest_bedroom",
        "reading_nook",
        "upper_bathroom",
        "games_room",
        "hobby_room",
        "balcony",
    ),
)


def _area_labels(floor_index: int, area_index: int) -> tuple[str, ...]:
    """Return deterministic area labels for generated rooms."""
    labels: list[str] = []
    # Every third room is tagged for guest-oriented selector coverage.
    if area_index % 3 == 0:
        labels.append("label_guest")
    # Upper floors expose climate labels at the area level.
    if floor_index in (2, 3):
        labels.append("label_climate")
    return tuple(labels)


# Generate 39 areas across four floors; the Basement has 12 (> _INVENTORY_AREAS_PER_FLOOR)
# so the per-floor truncation tail and floor names are both exercised.
_AREAS: tuple[AreaRecord, ...] = tuple(
    (
        slug,
        slug.replace("_", " ").title(),
        floor_id,
        _area_labels(floor_index, area_index),
        f"sensor.{slug}_temperature",
        f"sensor.{slug}_humidity",
    )
    for floor_index, (floor_id, _, _) in enumerate(_FLOORS, start=1)
    for area_index, slug in enumerate(_ROOM_SLUGS[floor_index - 1], start=1)
)


def _device_records() -> tuple[DeviceRecord, ...]:
    """Generate deterministic area-local devices."""
    records: list[DeviceRecord] = []
    for floor_index in range(1, _FLOOR_COUNT + 1):
        for _area_index, slug in enumerate(_ROOM_SLUGS[floor_index - 1], start=1):
            area_name = slug.replace("_", " ").title()
            records.extend(
                (
                    (
                        f"device_{slug}_lighting",
                        f"{area_name} Lighting",
                        slug,
                        ("label_evening",),
                    ),
                    (f"device_{slug}_outlet", f"{area_name} Outlet", slug, ()),
                    (
                        f"device_{slug}_climate",
                        f"{area_name} Climate",
                        slug,
                        ("label_climate",),
                    ),
                    (
                        f"device_{slug}_environment",
                        f"{area_name} Environment",
                        slug,
                        ("label_climate",),
                    ),
                    (
                        f"device_{slug}_security",
                        f"{area_name} Security",
                        slug,
                        ("label_security",),
                    ),
                    (f"device_{slug}_energy", f"{area_name} Energy", slug, ("label_energy",)),
                )
            )
    return tuple(records)


_DEVICES: tuple[DeviceRecord, ...] = _device_records()


def _entity_labels(domain: str, floor_index: int, area_index: int) -> tuple[str, ...]:
    """Return deterministic entity labels by domain and generated room position."""
    labels: list[str] = []
    # Domain-specific labels exercise label selectors over many entities.
    if domain == "light":
        labels.append("label_evening")
    if domain in ("climate", "sensor"):
        labels.append("label_climate")
    if area_index % 4 == 0:
        labels.append("label_guest")
    if floor_index == 4 and domain in ("sensor", "switch"):
        labels.append("label_energy")
    return tuple(dict.fromkeys(labels))


def _area_entity_records(floor_index: int, area_index: int) -> tuple[EntityRecord, ...]:
    """Generate eight entities per area with device-derived effective areas."""
    slug = _ROOM_SLUGS[floor_index - 1][area_index - 1]
    climate_device = f"device_{slug}_climate"
    entity_specs: tuple[tuple[str, str, str, str | None, str | None], ...] = (
        ("light", f"{slug}_ceiling", f"device_{slug}_lighting", None, None),
        ("light", f"{slug}_accent", f"device_{slug}_lighting", None, None),
        ("switch", f"{slug}_outlet", f"device_{slug}_outlet", None, None),
        ("climate", slug, climate_device, None, None),
        ("sensor", f"{slug}_temperature", f"device_{slug}_environment", "temperature", None),
        ("sensor", f"{slug}_humidity", f"device_{slug}_environment", "humidity", None),
        ("binary_sensor", f"{slug}_motion", f"device_{slug}_security", "motion", None),
        ("sensor", f"{slug}_power", f"device_{slug}_energy", "power", None),
    )
    return tuple(
        (
            f"{domain}.{object_id}",
            f"uid-{domain}-{object_id.replace('_', '-')}",
            device_id,
            slug if domain == "climate" and area_index % 5 == 0 else None,
            _entity_labels(domain, floor_index, area_index),
            None,
            None,
            device_class,
            original_device_class,
        )
        for domain, object_id, device_id, device_class, original_device_class in entity_specs
    )


# Generate 312 visible entities (39 areas x 8 entities each); the inventory digest cannot enumerate them usefully.
_ENTITIES: tuple[EntityRecord, ...] = tuple(
    entity
    for floor_index in range(1, _FLOOR_COUNT + 1)
    for area_index in range(1, len(_ROOM_SLUGS[floor_index - 1]) + 1)
    for entity in _area_entity_records(floor_index, area_index)
)


def _state_record(entity: EntityRecord) -> StateRecord:
    """Build deterministic state data for a generated entity record."""
    entity_id = entity[0]
    domain, object_id = entity_id.split(".", 1)
    name = object_id.replace("_", " ").title()
    last_changed = _LAST_CHANGED_OVERRIDES.get(entity_id, CREATED_AT)
    match domain:
        case "binary_sensor":
            return (entity_id, "off", name, last_changed, {"device_class": "motion"})
        case "climate":
            return (entity_id, "heat", name, last_changed, {"current_temperature": 21.0, "temperature": 20.0})
        case "light":
            state = "off" if entity_id in _BASEMENT_CEILING_LIGHTS_OFF else "on"
            return (entity_id, state, name, last_changed, {"brightness": 180})
        case "sensor" if object_id.endswith("_temperature"):
            return (
                entity_id,
                "21.5",
                name,
                last_changed,
                {"device_class": "temperature", "unit_of_measurement": "°C"},
            )
        case "sensor" if object_id.endswith("_humidity"):
            return (
                entity_id,
                "48",
                name,
                last_changed,
                {"device_class": "humidity", "unit_of_measurement": "%"},
            )
        case "sensor" if object_id.endswith("_power"):
            return (
                entity_id,
                "42",
                name,
                last_changed,
                {
                    "device_class": "power",
                    "state_class": "measurement",
                    "unit_of_measurement": "W",
                },
            )
        case "switch":
            return (entity_id, "off", name, last_changed, {})
        case _:
            return (entity_id, "unknown", name, last_changed, {})


_STATES: tuple[StateRecord, ...] = tuple(_state_record(entity) for entity in _ENTITIES)


def snapshot() -> HomeSnapshot:
    """Return a fresh frozen full home snapshot."""
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
    states = enrich_states(states, entities, devices, areas)
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
            "light": ("turn_off", "turn_on"),
            "switch": ("toggle",),
        },
        services_supports_response={
            "climate": {"set_temperature": SupportsResponse.NONE.value},
            "light": {"turn_off": SupportsResponse.NONE.value, "turn_on": SupportsResponse.NONE.value},
            "switch": {"toggle": SupportsResponse.NONE.value},
        },
        indexes=_indexes(entities, devices, areas, floors),
        labels=labels,
        categories={},
        issues=(),
        notifications=(),
        config_entries=(),
        services_schema={},
    )


def _daily_statistics(field: str, values: tuple[float, ...]) -> list[dict[str, object]]:
    """Return seven aligned daily Energy statistic rows."""
    return [
        {"start": f"2026-06-{22 + index:02d}T00:00:00+00:00", field: value}
        for index, value in enumerate(values)
    ]


def recorder() -> RecorderData:
    """Return modest recorder rows for representative full-home entities."""
    return {
        "history": {
            "sensor.utility_room_temperature": [
                {
                    "state": "20.8",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T08:00:00+00:00",
                    "last_updated": "2026-06-29T08:00:00+00:00",
                },
                {
                    "state": "21.2",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T10:00:00+00:00",
                    "last_updated": "2026-06-29T10:00:00+00:00",
                },
                {
                    "state": "21.5",
                    "attributes": {"unit_of_measurement": "°C"},
                    "last_changed": "2026-06-29T12:00:00+00:00",
                    "last_updated": "2026-06-29T12:00:00+00:00",
                },
            ],
            "light.living_room_ceiling": [
                {
                    "state": "off",
                    "attributes": {"brightness": 0},
                    "last_changed": "2026-06-29T07:30:00+00:00",
                    "last_updated": "2026-06-29T07:30:00+00:00",
                },
                {
                    "state": "on",
                    "attributes": {"brightness": 180},
                    "last_changed": "2026-06-29T11:45:00+00:00",
                    "last_updated": "2026-06-29T11:45:00+00:00",
                },
            ],
            "light.living_room_accent": [
                {
                    "state": "off",
                    "attributes": {"brightness": 0},
                    "last_changed": "2026-06-29T07:00:00+00:00",
                    "last_updated": "2026-06-29T07:00:00+00:00",
                },
                {
                    "state": "on",
                    "attributes": {"brightness": 180},
                    "last_changed": "2026-06-29T11:50:00+00:00",
                    "last_updated": "2026-06-29T11:50:00+00:00",
                },
            ],
        },
        "statistics": {
            "sensor.utility_room_power": _daily_statistics("change", (10, 11, 9, 12, 10, 8, 10)),
            "sensor.balcony_power": [
                *_daily_statistics("change", (6, 7, 5, 8, 7, 6, 8)),
                {
                    "start": "2026-06-29T10:00:00+00:00",
                    "end": "2026-06-29T11:00:00+00:00",
                    "state": 38.0,
                    "sum": 38.0,
                    "min": 34.0,
                    "max": 45.0,
                    "mean": 38.0,
                },
                {
                    "start": "2026-06-29T11:00:00+00:00",
                    "end": "2026-06-29T12:00:00+00:00",
                    "state": 42.0,
                    "sum": 42.0,
                    "min": 36.0,
                    "max": 49.0,
                    "mean": 42.0,
                },
            ],
            "sensor.storage_room_power": _daily_statistics("change", (4, 4, 5, 5, 4, 4, 4)),
            "sensor.workshop_power": _daily_statistics("change", (1, 1, 2, 1, 1, 1, 1)),
            "sensor.server_room_power": _daily_statistics("mean", (800, 850, 700, 900, 820, 760, 810)),
            "sensor.laundry_room_power": _daily_statistics("mean", (300, 320, 280, 350, 310, 290, 305)),
            "sensor.hallway_power": _daily_statistics("change", (2, 2.2, 1.8, 2.4, 2, 1.6, 2)),
            "sensor.living_room_power": _daily_statistics("change", (0.5, 0.6, 0.4, 0.7, 0.5, 0.4, 0.6)),
        },
        "logbook": {
            "light.living_room_ceiling": [
                {"when": "2026-06-29T07:30:00+00:00", "name": "Living Room Ceiling", "message": "turned off"},
                {"when": "2026-06-29T11:45:00+00:00", "name": "Living Room Ceiling", "message": "turned on"},
            ],
            "light.living_room_accent": [
                {"when": "2026-06-29T07:00:00+00:00", "name": "Living Room Accent", "message": "turned off"},
                {"when": "2026-06-29T11:50:00+00:00", "name": "Living Room Accent", "message": "turned on"},
            ],
            "switch.hallway_outlet": [
                {"when": "2026-06-29T09:00:00+00:00", "name": "Hallway Outlet", "message": "turned off"},
            ],
        },
    }


def energy() -> dict[str, object]:
    """Return deterministic raw Energy preferences and copied platform data."""
    return {
        "preferences": {
            "energy_sources": [
                {
                    "type": "grid",
                    "name": "Main grid",
                    "stat_energy_from": "sensor.utility_room_power",
                    "stat_energy_to": None,
                    "stat_cost": "sensor.hallway_power",
                    "entity_energy_price": None,
                    "number_energy_price": 0.28,
                    "stat_compensation": "sensor.living_room_power",
                    "entity_energy_price_export": None,
                    "number_energy_price_export": 0.08,
                    "stat_rate": "sensor.server_room_power",
                    "cost_adjustment_day": 0,
                },
                {
                    "type": "solar",
                    "name": "Roof solar",
                    "stat_energy_from": "sensor.balcony_power",
                    "config_entry_solar_forecast": ["forecast-config-private"],
                },
            ],
            "device_consumption": [
                {
                    "name": "Workshop circuit",
                    "stat_consumption": "sensor.storage_room_power",
                    "stat_rate": "sensor.laundry_room_power",
                },
                {
                    "name": "Workshop tools",
                    "stat_consumption": "sensor.workshop_power",
                    "included_in_stat": "sensor.storage_room_power",
                },
            ],
            "device_consumption_water": [],
        },
        "cost_sensors": {"sensor.external_raw_source": "sensor.raw_cost_private"},
        "metadata": {
            "sensor.utility_room_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": "energy",
                "unit_of_measurement": "kWh",
            },
            "sensor.balcony_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": "energy",
                "unit_of_measurement": "kWh",
            },
            "sensor.storage_room_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": "energy",
                "unit_of_measurement": "kWh",
            },
            "sensor.workshop_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": "energy",
                "unit_of_measurement": "kWh",
            },
            "sensor.server_room_power": {
                "has_sum": False,
                "has_mean": True,
                "unit_class": "power",
                "unit_of_measurement": "W",
            },
            "sensor.laundry_room_power": {
                "has_sum": False,
                "has_mean": True,
                "unit_class": "power",
                "unit_of_measurement": "W",
            },
            "sensor.hallway_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": None,
                "unit_of_measurement": "USD",
            },
            "sensor.living_room_power": {
                "has_sum": True,
                "has_mean": False,
                "unit_class": None,
                "unit_of_measurement": "USD",
            },
        },
        "forecast": {
            "forecast-config-private": {
                "wh_hours": {
                    "2026-06-29T13:00:00+00:00": 900,
                    "2026-06-29T14:00:00+00:00": 700,
                }
            }
        },
        "validation": {
            "energy_sources": [[], []],
            "device_consumption": [[], []],
            "device_consumption_water": [],
        },
    }


def automation() -> dict[str, object]:
    """Return fresh deterministic automation summaries, content, and run rows."""
    living_room_id = "automation.living_room_motion_lights"
    living_room_title = "Living Room Motion Lights"
    server_room_id = "automation.server_room_temperature_protection"
    server_room_title = "Server Room Temperature Protection"
    return {
        "records": [
            {
                "entity_id": living_room_id,
                "title": living_room_title,
                "state": "on",
                "is_on": True,
                "available": True,
                "description": "Turn on the Living Room lights when motion is detected.",
                "references": {
                    "entities": [
                        {"id": "binary_sensor.living_room_motion", "name": "Living Room Motion"},
                        {"id": "light.living_room_ceiling", "name": "Living Room Ceiling"},
                        {"id": "light.living_room_accent", "name": "Living Room Accent"},
                    ]
                },
                "search_terms": (
                    living_room_title,
                    "motion lights",
                    "living room",
                    "binary_sensor.living_room_motion",
                    "light.living_room_ceiling",
                    "light.living_room_accent",
                ),
                "content": {
                    "id": "living_room_motion_lights",
                    "alias": living_room_title,
                    "description": "Turn on the Living Room lights when motion is detected.",
                    "triggers": [
                        {
                            "trigger": "state",
                            "entity_id": "binary_sensor.living_room_motion",
                            "to": "on",
                        }
                    ],
                    "actions": [
                        {
                            "action": "light.turn_on",
                            "target": {
                                "entity_id": [
                                    "light.living_room_ceiling",
                                    "light.living_room_accent",
                                ]
                            },
                        }
                    ],
                    "mode": "single",
                },
            },
            {
                "entity_id": server_room_id,
                "title": server_room_title,
                "state": "on",
                "is_on": True,
                "available": True,
                "description": "Turn off the Server Room outlet when its temperature is too high.",
                "references": {
                    "entities": [
                        {"id": "sensor.server_room_temperature", "name": "Server Room Temperature"},
                        {"id": "switch.server_room_outlet", "name": "Server Room Outlet"},
                    ]
                },
                "search_terms": (
                    server_room_title,
                    "server room temperature",
                    "sensor.server_room_temperature",
                    "switch.server_room_outlet",
                ),
                "content": {
                    "id": "server_room_temperature_protection",
                    "alias": server_room_title,
                    "description": "Turn off the Server Room outlet when its temperature is too high.",
                    "triggers": [
                        {
                            "trigger": "numeric_state",
                            "entity_id": "sensor.server_room_temperature",
                            "above": 28,
                        }
                    ],
                    "actions": [
                        {
                            "action": "switch.turn_off",
                            "target": {"entity_id": "switch.server_room_outlet"},
                        }
                    ],
                    "mode": "single",
                },
            },
        ],
        "runs": {
            living_room_id: [
                {
                    "entity_id": living_room_id,
                    "when": "2026-06-29T11:40:00+00:00",
                    "name": living_room_title,
                    "message": "triggered",
                },
                {
                    "entity_id": living_room_id,
                    "when": "2026-06-29T09:30:00+00:00",
                    "name": living_room_title,
                    "message": "triggered",
                },
            ],
            server_room_id: [
                {
                    "entity_id": server_room_id,
                    "when": "2026-06-29T09:15:00+00:00",
                    "name": server_room_title,
                    "message": "triggered",
                }
            ],
        },
    }


def _config() -> SafeConfig:
    """Build a minimal frozen config record for snapshot helpers."""
    return SafeConfig(
        location_name="Large Eval Home",
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
        last_changed_timestamp=datetime.fromisoformat(last_changed).timestamp(),
        last_reported=last_changed,
        last_reported_timestamp=datetime.fromisoformat(last_changed).timestamp(),
        last_updated=last_changed,
        last_updated_timestamp=datetime.fromisoformat(last_changed).timestamp(),
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
        domain=domain,
        unique_id=unique_id,
        platform=domain,
        config_entry_id="entry_large",
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
        model="Large Fixture",
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
        config_entries=("entry_large",),
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
