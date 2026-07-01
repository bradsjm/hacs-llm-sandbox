"""Frozen, JSON-safe snapshot records mirroring Home Assistant registries.

Each record carries a ``__llm_sandbox_json__`` hook so the executor's
``json_safe`` machinery serializes it without extra adapters. All datetimes
are stored as ISO strings (Monty has limited datetime support). Enums are
stored as their ``.value`` strings. Sets of tuples (device identifiers/
connections) are stored as JSON-safe tuples of tuples.

Field types use plain ``dict``/``tuple`` rather than ``MappingProxyType``
because the Monty VM cannot convert ``mappingproxy`` objects. Read-only
guarantees come from ``@dataclass(frozen=True)`` (no field reassignment)
and Monty's own immutable type system (Monty makes its own copy of every
input value, so Python-side mutability is irrelevant from the sandbox).
"""

# ruff: noqa: D105

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from homeassistant.util.json import JsonValueType

type ServiceFieldBrief = dict[str, str | bool | None]
type ServiceSchemaBrief = dict[str, list[ServiceFieldBrief] | bool]


@dataclass(frozen=True, slots=True)
class SafeContext:
    """Frozen view of a Home Assistant execution context."""

    id: str | None
    parent_id: str | None
    user_id: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {"id": self.id, "parent_id": self.parent_id, "user_id": self.user_id},
        )


@dataclass(frozen=True, slots=True)
class SafeState:
    """Frozen view of a Home Assistant state object."""

    entity_id: str
    domain: str
    object_id: str
    name: str | None
    state: str
    attributes: dict[str, object]
    last_changed: str
    last_reported: str | None
    last_updated: str
    context: SafeContext

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "entity_id": self.entity_id,
                "domain": self.domain,
                "object_id": self.object_id,
                "name": self.name,
                "state": self.state,
                "attributes": self.attributes,
                "last_changed": self.last_changed,
                "last_reported": self.last_reported,
                "last_updated": self.last_updated,
                "context": self.context,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeRegistryEntry:
    """Frozen view of a Home Assistant entity registry entry."""

    entity_id: str
    domain: str
    unique_id: str | None
    platform: str
    config_entry_id: str | None
    device_id: str | None
    area_id: str | None
    name: str | None
    original_name: str | None
    aliases: tuple[str, ...]
    labels: tuple[str, ...]
    disabled_by: str | None
    hidden_by: str | None
    entity_category: str | None
    device_class: str | None
    original_device_class: str | None
    capabilities: dict[str, object] | None
    supported_features: int
    translation_key: str | None
    has_entity_name: bool

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "entity_id": self.entity_id,
                "domain": self.domain,
                "unique_id": self.unique_id,
                "platform": self.platform,
                "config_entry_id": self.config_entry_id,
                "device_id": self.device_id,
                "area_id": self.area_id,
                "name": self.name,
                "original_name": self.original_name,
                "aliases": self.aliases,
                "labels": self.labels,
                "disabled_by": self.disabled_by,
                "hidden_by": self.hidden_by,
                "entity_category": self.entity_category,
                "device_class": self.device_class,
                "original_device_class": self.original_device_class,
                "capabilities": self.capabilities,
                "supported_features": self.supported_features,
                "translation_key": self.translation_key,
                "has_entity_name": self.has_entity_name,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeDeviceEntry:
    """Frozen view of a Home Assistant device registry entry."""

    id: str
    name: str | None
    name_by_user: str | None
    manufacturer: str | None
    model: str | None
    model_id: str | None
    sw_version: str | None
    hw_version: str | None
    serial_number: str | None
    area_id: str | None
    labels: tuple[str, ...]
    identifiers: tuple[tuple[str, ...], ...]
    connections: tuple[tuple[str, ...], ...]
    configuration_url: str | None
    entry_type: str | None
    config_entries: tuple[str, ...]
    via_device_id: str | None
    disabled_by: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "id": self.id,
                "name": self.name,
                "name_by_user": self.name_by_user,
                "manufacturer": self.manufacturer,
                "model": self.model,
                "model_id": self.model_id,
                "sw_version": self.sw_version,
                "hw_version": self.hw_version,
                "serial_number": self.serial_number,
                "area_id": self.area_id,
                "labels": self.labels,
                "identifiers": [list(ident) for ident in self.identifiers],
                "connections": [list(conn) for conn in self.connections],
                "configuration_url": self.configuration_url,
                "entry_type": self.entry_type,
                "config_entries": self.config_entries,
                "via_device_id": self.via_device_id,
                "disabled_by": self.disabled_by,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeAreaEntry:
    """Frozen view of a Home Assistant area registry entry."""

    id: str
    area_id: str
    name: str
    aliases: tuple[str, ...]
    floor_id: str | None
    labels: tuple[str, ...]
    icon: str | None
    picture: str | None
    humidity_entity_id: str | None
    temperature_entity_id: str | None
    created_at: str | None
    modified_at: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "id": self.id,
                "area_id": self.area_id,
                "name": self.name,
                "aliases": self.aliases,
                "floor_id": self.floor_id,
                "labels": self.labels,
                "icon": self.icon,
                "picture": self.picture,
                "humidity_entity_id": self.humidity_entity_id,
                "temperature_entity_id": self.temperature_entity_id,
                "created_at": self.created_at,
                "modified_at": self.modified_at,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeFloorEntry:
    """Frozen view of a Home Assistant floor registry entry."""

    floor_id: str
    id: str
    name: str
    aliases: tuple[str, ...]
    level: int | None
    icon: str | None
    created_at: str | None
    modified_at: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "floor_id": self.floor_id,
                "id": self.id,
                "name": self.name,
                "aliases": self.aliases,
                "level": self.level,
                "icon": self.icon,
                "created_at": self.created_at,
                "modified_at": self.modified_at,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeLabelEntry:
    """Frozen view of a Home Assistant label registry entry."""

    label_id: str
    name: str
    normalized_name: str
    description: str | None
    color: str | None
    icon: str | None
    created_at: str | None
    modified_at: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "label_id": self.label_id,
                "name": self.name,
                "normalized_name": self.normalized_name,
                "description": self.description,
                "color": self.color,
                "icon": self.icon,
                "created_at": self.created_at,
                "modified_at": self.modified_at,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeCategoryEntry:
    """Frozen view of a Home Assistant category registry entry."""

    category_id: str
    scope: str
    name: str
    icon: str | None
    created_at: str | None
    modified_at: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "category_id": self.category_id,
                "scope": self.scope,
                "name": self.name,
                "icon": self.icon,
                "created_at": self.created_at,
                "modified_at": self.modified_at,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeIssueEntry:
    """Frozen view of a Home Assistant repairs issue entry."""

    issue_id: str
    domain: str
    severity: str | None
    active: bool
    dismissed_version: str | None
    translation_key: str | None
    translation_placeholders: dict[str, str] | None
    created: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "issue_id": self.issue_id,
                "domain": self.domain,
                "severity": self.severity,
                "active": self.active,
                "dismissed_version": self.dismissed_version,
                "translation_key": self.translation_key,
                "translation_placeholders": self.translation_placeholders,
                "created": self.created,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeNotificationEntry:
    """Frozen view of a Home Assistant persistent notification."""

    notification_id: str
    title: str | None
    message: str
    created_at: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "notification_id": self.notification_id,
                "title": self.title,
                "message": self.message,
                "created_at": self.created_at,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeConfigEntry:
    """Frozen, secret-stripped view of a Home Assistant config entry.

    Only non-sensitive metadata is exposed. ``data``, ``options``,
    ``runtime_data``, and ``subentries`` (which may contain credentials) are
    intentionally absent and never reach the Monty sandbox.
    """

    entry_id: str
    domain: str
    title: str
    source: str
    state: str
    unique_id: str | None
    disabled_by: str | None
    reason: str | None

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "entry_id": self.entry_id,
                "domain": self.domain,
                "title": self.title,
                "source": self.source,
                "state": self.state,
                "unique_id": self.unique_id,
                "disabled_by": self.disabled_by,
                "reason": self.reason,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeUnitSystem:
    """Frozen view of Home Assistant unit-system preferences."""

    temperature_unit: str
    length_unit: str
    mass_unit: str
    pressure_unit: str
    volume_unit: str
    area_unit: str
    wind_speed_unit: str
    accumulated_precipitation_unit: str

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "temperature_unit": self.temperature_unit,
                "length_unit": self.length_unit,
                "mass_unit": self.mass_unit,
                "pressure_unit": self.pressure_unit,
                "volume_unit": self.volume_unit,
                "area_unit": self.area_unit,
                "wind_speed_unit": self.wind_speed_unit,
                "accumulated_precipitation_unit": self.accumulated_precipitation_unit,
            },
        )


@dataclass(frozen=True, slots=True)
class SafeConfig:
    """Frozen view of Home Assistant instance configuration metadata."""

    location_name: str
    latitude: float
    longitude: float
    elevation: int
    time_zone: str
    language: str
    country: str | None
    currency: str
    internal_url: str | None
    external_url: str | None
    units: SafeUnitSystem

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "location_name": self.location_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "elevation": self.elevation,
                "time_zone": self.time_zone,
                "language": self.language,
                "country": self.country,
                "currency": self.currency,
                "internal_url": self.internal_url,
                "external_url": self.external_url,
                "units": self.units,
            },
        )


@dataclass(frozen=True, slots=True)
class SnapshotIndexes:
    """Precomputed lookup indexes over the snapshot.

    Effective area for an entity is ``entity.area_id or device.area_id``: an
    entity-level area override wins; otherwise the entity inherits its
    device's area. ``entity_ids_by_area_id`` uses this effective rule so area
    traversals match Home Assistant's resolution semantics.
    """

    entity_ids_by_device_id: dict[str, tuple[str, ...]]
    entity_ids_by_area_id: dict[str, tuple[str, ...]]
    device_ids_by_area_id: dict[str, tuple[str, ...]]
    entity_ids_by_config_entry_id: dict[str, tuple[str, ...]]
    entity_ids_by_label: dict[str, tuple[str, ...]]
    device_ids_by_label: dict[str, tuple[str, ...]]
    area_ids_by_floor_id: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SnapshotScope:
    """Optional snapshot scope reduction (noise reduction, NOT security).

    Each restriction is independent and applied additively. An entity is
    visible iff it passes every enabled check. Disabled entities are always
    excluded in any restricted scope because they are absent from the state
    machine; when every restriction is off, all state-bearing entities pass.
    """

    assistant: str
    restrict_to_assist_exposed: bool
    exclude_hidden: bool
    excluded_entity_categories: frozenset[str]


# No-arg build_snapshot default: all optional restrictions off for callers that
# need every state-bearing entity. Product defaults are applied via settings.
DEFAULT_SCOPE: SnapshotScope = SnapshotScope(
    assistant="",
    restrict_to_assist_exposed=False,
    exclude_hidden=False,
    excluded_entity_categories=frozenset(),
)


@dataclass(frozen=True, slots=True)
class HomeSnapshot:
    """Frozen, full snapshot of Home Assistant state and registries.

    Built fresh per ``execute_home_code`` tool call on the event loop, then
    passed to the Monty runtime. Optional visibility filtering reduces noise;
    it is not a security boundary.
    """

    created_at: str
    states: dict[str, SafeState]
    entities: dict[str, SafeRegistryEntry]
    devices: dict[str, SafeDeviceEntry]
    areas: dict[str, SafeAreaEntry]
    floors: dict[str, SafeFloorEntry]
    config: SafeConfig
    services: dict[str, tuple[str, ...]]
    services_supports_response: dict[str, dict[str, str]]
    indexes: SnapshotIndexes
    labels: dict[str, SafeLabelEntry]
    categories: dict[str, dict[str, SafeCategoryEntry]]
    issues: list[SafeIssueEntry]
    notifications: list[SafeNotificationEntry]
    config_entries: list[SafeConfigEntry]
    services_schema: Mapping[str, Mapping[str, ServiceSchemaBrief]] = field(default_factory=dict)
