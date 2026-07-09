"""Frozen, JSON-safe snapshot records mirroring Home Assistant registries.

Each record inherits ``_JsonSafeRecord``, whose ``__llm_sandbox_json__`` hook
exposes its dataclass fields as a raw dict; the executor's ``json_safe``
machinery recurses through that dict and converts nested ``Safe*`` records,
tuples, dicts, and sets to JSON-safe structures. All datetimes are stored as
ISO strings (Monty has limited datetime support). Enums are stored as their
``.value`` strings. Sets of tuples (device identifiers/connections) are stored
as JSON-safe tuples of tuples.

Field types use plain ``dict``/``tuple`` rather than ``MappingProxyType``
because the Monty VM cannot convert ``mappingproxy`` objects. Read-only
guarantees come from ``@dataclass(frozen=True)`` (no field reassignment)
and Monty's own immutable type system (Monty makes its own copy of every
input value, so Python-side mutability is irrelevant from the sandbox).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, cast

from homeassistant.util.json import JsonValueType

# Service field briefs are loosely-structured JSON dicts. Known keys:
# ``name`` (str), ``required`` (bool), ``type_hint`` (str | None),
# ``description`` (str | None), and the optional ``filter`` carrying HA's
# field-level capability filter (``supported_features`` bit constants and/or
# ``attribute`` value-intersection rules such as ``supported_color_modes``).
type ServiceFieldFilter = dict[str, list[int] | dict[str, list[int | str]]]
type ServiceFieldBrief = dict[str, str | bool | None | list[int] | ServiceFieldFilter]
type ServiceSchemaBrief = dict[str, list[ServiceFieldBrief] | bool]

# Per-service target metadata, mirroring HA's automation target matching
# (``websocket_api/automation.py``). ``entity`` holds one filter dict per
# accepted entity group; each may constrain ``domain``, ``device_class``,
# ``integration``, and ``supported_features``. ``primary_entities_only``
# gates non-primary entity categories. Services without a declared target
# are absent from this mapping, which the matcher treats as "accepts any".
type ServiceTargetFilter = dict[str, list[str | int] | str | None]
type ServiceTargetBrief = dict[str, list[ServiceTargetFilter] | bool]


class _JsonSafeRecord:
    """Serialize a frozen record to a dict of its dataclass fields.

    The hook returns raw field values; the executor's ``json_safe`` recursion
    converts nested ``Safe*`` records, tuples (incl. tuple-of-tuples), dicts,
    and sets to JSON-safe structures. ``__slots__`` keeps subclasses with
    ``slots=True`` truly slot-only.
    """

    __slots__ = ()

    def __llm_sandbox_json__(self) -> JsonValueType:
        # Mixin is consumed only by dataclass subclasses; cast satisfies the
        # dataclass-typed ``fields()`` argument without weakening the surface.
        return cast(JsonValueType, {f.name: getattr(self, f.name) for f in fields(cast(Any, self))})


@dataclass(frozen=True, slots=True)
class SafeContext(_JsonSafeRecord):
    """Frozen view of a Home Assistant execution context."""

    id: str | None
    parent_id: str | None
    user_id: str | None


@dataclass(frozen=True, slots=True)
class SafeState(_JsonSafeRecord):
    """Frozen view of a Home Assistant state object.

    Carries HA-native state fields, POSIX timestamp mirrors for easy duration
    math, plus registry-derived join keys (effective ``area_id``, ``floor_id``,
    ``device_id``, ``platform``, ``unique_id``) so an LLM can filter by location without a
    manual state-to-registry join. The effective area mirrors the snapshot index
    rule (``entity.area_id or device.area_id``) and is ``None`` when no entity
    registry entry exists.
    """

    entity_id: str
    domain: str
    object_id: str
    name: str | None
    state: str
    attributes: dict[str, object]
    last_changed: str
    last_changed_timestamp: float
    last_reported: str | None
    last_reported_timestamp: float | None
    last_updated: str
    last_updated_timestamp: float
    context: SafeContext
    # Registry-derived join keys filled by ``enrich_states`` (effective area,
    # device, platform, unique_id). Default None so the base HA-native shape is
    # constructible before the join; ``None`` when no registry entry exists.
    area_id: str | None = None
    floor_id: str | None = None
    device_id: str | None = None
    platform: str | None = None
    unique_id: str | None = None


@dataclass(frozen=True, slots=True)
class SafeRegistryEntry(_JsonSafeRecord):
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


@dataclass(frozen=True, slots=True)
class SafeDeviceEntry(_JsonSafeRecord):
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


@dataclass(frozen=True, slots=True)
class SafeAreaEntry(_JsonSafeRecord):
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


@dataclass(frozen=True, slots=True)
class SafeFloorEntry(_JsonSafeRecord):
    """Frozen view of a Home Assistant floor registry entry."""

    floor_id: str
    id: str
    name: str
    aliases: tuple[str, ...]
    level: int | None
    icon: str | None
    created_at: str | None
    modified_at: str | None


@dataclass(frozen=True, slots=True)
class SafeLabelEntry(_JsonSafeRecord):
    """Frozen view of a Home Assistant label registry entry."""

    label_id: str
    name: str
    normalized_name: str
    description: str | None
    color: str | None
    icon: str | None
    created_at: str | None
    modified_at: str | None


@dataclass(frozen=True, slots=True)
class SafeCategoryEntry(_JsonSafeRecord):
    """Frozen view of a Home Assistant category registry entry."""

    category_id: str
    scope: str
    name: str
    icon: str | None
    created_at: str | None
    modified_at: str | None


@dataclass(frozen=True, slots=True)
class SafeIssueEntry(_JsonSafeRecord):
    """Frozen view of a Home Assistant repairs issue entry."""

    issue_id: str
    domain: str
    severity: str | None
    active: bool
    dismissed_version: str | None
    translation_key: str | None
    translation_placeholders: dict[str, Any] | None
    created: str | None


@dataclass(frozen=True, slots=True)
class SafeNotificationEntry(_JsonSafeRecord):
    """Frozen view of a Home Assistant persistent notification."""

    notification_id: str
    title: str | None
    message: str
    created_at: str | None


@dataclass(frozen=True, slots=True)
class SafeConfigEntry(_JsonSafeRecord):
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


@dataclass(frozen=True, slots=True)
class SafeUnitSystem(_JsonSafeRecord):
    """Frozen view of Home Assistant unit-system preferences."""

    temperature_unit: str
    length_unit: str
    mass_unit: str
    pressure_unit: str
    volume_unit: str
    area_unit: str
    wind_speed_unit: str
    accumulated_precipitation_unit: str


@dataclass(frozen=True, slots=True)
class SafeConfig(_JsonSafeRecord):
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
    include_all_diagnostics: bool


# No-arg build_snapshot default: all optional restrictions off for callers that
# need every state-bearing entity. Product defaults are applied via settings.
DEFAULT_SCOPE: SnapshotScope = SnapshotScope(
    assistant="",
    restrict_to_assist_exposed=False,
    exclude_hidden=False,
    excluded_entity_categories=frozenset(),
    include_all_diagnostics=True,
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
    services_target: Mapping[str, Mapping[str, ServiceTargetBrief]] = field(default_factory=dict)
