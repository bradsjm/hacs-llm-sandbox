"""Build a frozen Home Assistant snapshot for Monty execution.

Runs on the event loop at the start of each ``execute_home_code`` tool call,
reads all registries and the state machine, and returns an immutable
``HomeSnapshot``. Optional scope filtering is applied at build time as a
noise-reduction measure, not a security boundary. Visibility restrictions are
combined additively over state-bearing entity ids, and the service catalog is
never filtered.
"""

from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime
from enum import Enum
from math import isfinite
from typing import cast

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components.persistent_notification import DOMAIN as PERSISTENT_NOTIFICATION_DOMAIN
from homeassistant.components.persistent_notification import Notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import category_registry as cr_core
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers import label_registry as lr
from homeassistant.util import dt as dt_util

from .models import (
    DEFAULT_SCOPE,
    HomeSnapshot,
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfig,
    SafeConfigEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
    SnapshotScope,
)
from .services import _safe_services

USEFUL_DIAGNOSTIC_DEVICE_CLASSES = frozenset(
    {"battery", "battery_charging", "signal_strength", "connectivity", "problem", "power"}
)


def build_snapshot(
    hass: HomeAssistant,
    scope: SnapshotScope = DEFAULT_SCOPE,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Build a frozen snapshot of states, registries, and services."""
    return _build_snapshot(hass, scope=scope, anchor_device_id=anchor_device_id, flavor="full")


def build_recorder_snapshot(
    hass: HomeAssistant,
    scope: SnapshotScope = DEFAULT_SCOPE,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Build a fresh, recorder-focused snapshot with selector indexes."""
    return _build_snapshot(hass, scope=scope, anchor_device_id=anchor_device_id, flavor="recorder")


def build_vision_snapshot(
    hass: HomeAssistant,
    scope: SnapshotScope = DEFAULT_SCOPE,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Build a fresh, vision-focused snapshot containing only visible states and config."""
    return _build_snapshot(hass, scope=scope, anchor_device_id=anchor_device_id, flavor="vision")


def _build_snapshot(
    hass: HomeAssistant,
    *,
    scope: SnapshotScope,
    anchor_device_id: str | None,
    flavor: str,
) -> HomeSnapshot:
    """Build one snapshot flavor from live HA data without retaining live objects."""
    live_states = hass.states.async_all()
    entity_registry = er.async_get(hass)
    live_entities = entity_registry.entities

    # Visibility is decided from live state ids plus live registry metadata before
    # any Safe* materialization, so narrow snapshots never allocate excluded records.
    visible = _visible_entity_ids_from_live(hass, live_states, live_entities, scope)
    visible_states = [state for state in live_states if state.entity_id in visible]

    entities: dict[str, SafeRegistryEntry] = {}
    devices: dict[str, SafeDeviceEntry] = {}
    areas: dict[str, SafeAreaEntry] = {}
    floors: dict[str, SafeFloorEntry] = {}
    labels: dict[str, SafeLabelEntry] = {}
    indexes = _empty_indexes()
    live_visible_entities: dict[str, er.RegistryEntry] = {}
    live_selected_devices: dict[str, dr.DeviceEntry] = {}
    live_selected_areas: dict[str, ar.AreaEntry] = {}

    if flavor != "vision":
        device_registry = dr.async_get(hass)
        area_registry = ar.async_get(hass)
        floor_registry = fr.async_get(hass)
        label_registry = lr.async_get(hass)

        live_visible_entities = {
            entity_id: entry for entity_id in visible if (entry := live_entities.get(entity_id)) is not None
        }
        device_ids: set[str] = {entry.device_id for entry in live_visible_entities.values() if entry.device_id}
        # Force-include the initiating device so snapshot-based location works.
        if anchor_device_id is not None:
            device_ids.add(anchor_device_id)
        live_selected_devices = {
            device_id: device
            for device_id in device_ids
            if (device := device_registry.async_get(device_id)) is not None
        }

        area_ids: set[str] = {device.area_id for device in live_selected_devices.values() if device.area_id}
        area_ids.update(entry.area_id for entry in live_visible_entities.values() if entry.area_id)
        live_selected_areas = {area.id: area for area in area_registry.async_list_areas() if area.id in area_ids}

        floor_ids: set[str] = {area.floor_id for area in live_selected_areas.values() if area.floor_id}
        live_selected_floors = {
            floor.floor_id: floor for floor in floor_registry.async_list_floors() if floor.floor_id in floor_ids
        }

        entities = {entity_id: _safe_entity(entry) for entity_id, entry in live_visible_entities.items()}
        devices = {device_id: _safe_device(device) for device_id, device in live_selected_devices.items()}
        areas = {area_id: _safe_area(area) for area_id, area in live_selected_areas.items()}
        floors = {floor_id: _safe_floor(floor) for floor_id, floor in live_selected_floors.items()}
        indexes = _build_indexes(entities, devices, areas)

        if flavor == "full":
            labels = {label.label_id: _safe_label(label) for label in label_registry.async_list_labels()}
        else:
            label_ids: set[str] = set()
            for entry in live_visible_entities.values():
                label_ids.update(entry.labels)
            for device in live_selected_devices.values():
                label_ids.update(device.labels)
            labels = {
                label.label_id: _safe_label(label)
                for label in label_registry.async_list_labels()
                if label.label_id in label_ids
            }

    states: dict[str, SafeState] = {}
    for state in visible_states:
        live_entry = live_visible_entities.get(state.entity_id)
        live_device = None
        if live_entry is not None and live_entry.device_id is not None:
            live_device = live_selected_devices.get(live_entry.device_id)
        states[state.entity_id] = _safe_state(state, live_entry, live_device, live_selected_areas)

    if flavor == "full":
        category_registry = cr_core.async_get(hass)
        issue_registry = ir.async_get(hass)
        categories = {
            scope_name: {cid: _safe_category(scope_name, cat) for cid, cat in entries.items()}
            for scope_name, entries in category_registry.categories.items()
        }
        issues = [_safe_issue(issue) for issue in issue_registry.issues.values()]
        notification_store = hass.data.get(PERSISTENT_NOTIFICATION_DOMAIN)
        notifications = (
            [_safe_notification(notification) for notification in notification_store.values()]
            if isinstance(notification_store, dict)
            else []
        )
        config_entries = [_safe_config_entry(entry) for entry in hass.config_entries.async_entries()]
        service_catalog, service_response, services_schema, services_target = _safe_services(hass)
    else:
        service_catalog = {}
        service_response = {}
        services_schema = {}
        services_target = {}
        categories = {}
        issues = []
        notifications = []
        config_entries = []

    return HomeSnapshot(
        created_at=dt_util.utcnow().isoformat(),
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        config=_safe_config(hass),
        services=service_catalog,
        services_supports_response=service_response,
        indexes=indexes,
        labels=labels,
        categories=categories,
        issues=issues,
        notifications=notifications,
        config_entries=config_entries,
        services_schema=services_schema,
        services_target=services_target,
    )


def finalize_snapshot(
    snapshot: HomeSnapshot,
    *,
    visible: set[str],
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Return ``snapshot`` with collections finalized from visible entity ids.

    Pure helper for offline snapshot fixtures: callers supply the visibility set,
    while the production collection cascade, state enrichment, and index rebuild
    stay centralized here.
    """
    states, entities, devices, areas, floors, indexes = _finalize_snapshot_collections(
        snapshot.states,
        snapshot.entities,
        snapshot.devices,
        snapshot.areas,
        snapshot.floors,
        visible=visible,
        anchor_device_id=anchor_device_id,
    )
    return replace(
        snapshot,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        indexes=indexes,
    )


def _finalize_snapshot_collections(
    states: dict[str, SafeState],
    entities: dict[str, SafeRegistryEntry],
    devices: dict[str, SafeDeviceEntry],
    areas: dict[str, SafeAreaEntry],
    floors: dict[str, SafeFloorEntry],
    *,
    visible: set[str],
    anchor_device_id: str | None,
) -> tuple[
    dict[str, SafeState],
    dict[str, SafeRegistryEntry],
    dict[str, SafeDeviceEntry],
    dict[str, SafeAreaEntry],
    dict[str, SafeFloorEntry],
    SnapshotIndexes,
]:
    """Apply the canonical visible-collection cascade and index rebuild."""
    states, entities, devices, areas, floors = _derive_collections(
        states,
        entities,
        devices,
        areas,
        floors,
        visible,
        anchor_device_id,
    )
    # Fill registry-derived join keys (effective area/floor, device, platform, unique_id)
    # now that the filtered entity/device collections exist, mirroring the index rule.
    states = enrich_states(states, entities, devices, areas)
    return states, entities, devices, areas, floors, _build_indexes(entities, devices, areas)


def _visible_entity_ids(
    hass: HomeAssistant,
    states: dict[str, SafeState],
    entities: dict[str, SafeRegistryEntry],
    scope: SnapshotScope,
) -> set[str]:
    """Return entity_ids visible under the combined ``scope`` restrictions.

    Each enabled restriction independently narrows the set: Assist exposure
    delegates to HA's exposure logic, and the registry-characteristic flags
    drop hidden or category-excluded entities. State-only entities (no
    registry entry) pass the registry checks. Candidates are state-bearing
    entities; disabled entities are absent from the state machine and thus
    always excluded.
    """
    return {entity_id for entity_id in states if _passes_visibility(hass, entities.get(entity_id), entity_id, scope)}


def _visible_entity_ids_from_live(
    hass: HomeAssistant,
    states: list[State],
    entities: Mapping[str, er.RegistryEntry],
    scope: SnapshotScope,
) -> set[str]:
    """Return visible state-bearing entity ids using live registry metadata."""
    return {
        state.entity_id
        for state in states
        if _passes_live_visibility(hass, entities.get(state.entity_id), state.entity_id, scope)
    }


def _passes_visibility(
    hass: HomeAssistant,
    entry: SafeRegistryEntry | None,
    entity_id: str,
    scope: SnapshotScope,
) -> bool:
    """Whether an entity passes every enabled visibility restriction."""
    # Assist exposure: delegate to HA's exposure logic (sync @callback; may
    # idempotently persist computed exposure, matching HA's default_agent).
    if scope.restrict_to_assist_exposed and not async_should_expose(hass, scope.assistant, entity_id):
        return False
    # Registry-characteristic checks do not apply to state-only entities.
    if entry is None:
        return True
    if scope.exclude_hidden and entry.hidden_by is not None:
        return False
    # Category exclusions drop explicitly disabled noise categories before diagnostic selectivity applies.
    if entry.entity_category in scope.excluded_entity_categories:
        return False
    # Selective diagnostics keep useful status signals while hiding low-value diagnostic noise by default.
    if entry.entity_category == "diagnostic" and not scope.include_all_diagnostics:
        return _has_useful_diagnostic_device_class(entry)
    return True


def _passes_live_visibility(
    hass: HomeAssistant,
    entry: er.RegistryEntry | None,
    entity_id: str,
    scope: SnapshotScope,
) -> bool:
    """Whether an entity passes every enabled restriction before safe conversion."""
    # Assist exposure: delegate to HA's exposure logic (sync @callback; may
    # idempotently persist computed exposure, matching HA's default_agent).
    if scope.restrict_to_assist_exposed and not async_should_expose(hass, scope.assistant, entity_id):
        return False
    # Registry-characteristic checks do not apply to state-only entities.
    if entry is None:
        return True
    if scope.exclude_hidden and entry.hidden_by is not None:
        return False
    entity_category = _enum_value(entry.entity_category)
    # Category exclusions drop explicitly disabled noise categories before diagnostic selectivity applies.
    if entity_category in scope.excluded_entity_categories:
        return False
    # Selective diagnostics keep useful status signals while hiding low-value diagnostic noise by default.
    if entity_category == "diagnostic" and not scope.include_all_diagnostics:
        return _has_useful_diagnostic_device_class(entry)
    return True


def _has_useful_diagnostic_device_class(entry: SafeRegistryEntry | er.RegistryEntry) -> bool:
    """Return whether registry metadata marks a diagnostic entity as useful."""
    return (entry.device_class or entry.original_device_class) in USEFUL_DIAGNOSTIC_DEVICE_CLASSES


def _derive_collections(
    states: dict[str, SafeState],
    entities: dict[str, SafeRegistryEntry],
    devices: dict[str, SafeDeviceEntry],
    areas: dict[str, SafeAreaEntry],
    floors: dict[str, SafeFloorEntry],
    visible: set[str],
    anchor_device_id: str | None,
) -> tuple[
    dict[str, SafeState],
    dict[str, SafeRegistryEntry],
    dict[str, SafeDeviceEntry],
    dict[str, SafeAreaEntry],
    dict[str, SafeFloorEntry],
]:
    """Derive filtered collections from the visible entity set."""
    filtered_states = {entity_id: states[entity_id] for entity_id in visible}
    filtered_entities = {entity_id: entities[entity_id] for entity_id in visible if entity_id in entities}

    device_ids: set[str] = {entry.device_id for entry in filtered_entities.values() if entry.device_id}
    # Force-include the initiating device so snapshot-based location works.
    if anchor_device_id is not None:
        device_ids.add(anchor_device_id)
    filtered_devices = {device_id: devices[device_id] for device_id in device_ids if device_id in devices}

    area_ids: set[str] = {device.area_id for device in filtered_devices.values() if device.area_id}
    area_ids.update(entry.area_id for entry in filtered_entities.values() if entry.area_id)
    filtered_areas = {area_id: areas[area_id] for area_id in area_ids if area_id in areas}

    floor_ids: set[str] = {area.floor_id for area in filtered_areas.values() if area.floor_id}
    filtered_floors = {floor_id: floors[floor_id] for floor_id in floor_ids if floor_id in floors}

    return filtered_states, filtered_entities, filtered_devices, filtered_areas, filtered_floors


def enrich_states(
    states: dict[str, SafeState],
    entities: dict[str, SafeRegistryEntry],
    devices: dict[str, SafeDeviceEntry],
    areas: dict[str, SafeAreaEntry],
) -> dict[str, SafeState]:
    """Fill registry-derived join keys on each ``SafeState``.

    Effective area uses the same rule as ``_build_indexes``: an entity-level
    ``area_id`` override wins, otherwise the entity inherits its device's area.
    Join keys are ``None`` for state-bearing entities with no registry entry.
    Pure and snapshot-derived so eval fixtures can reuse the canonical join.
    """
    if not entities:
        return states
    enriched: dict[str, SafeState] = {}
    for entity_id, safe_state in states.items():
        entry = entities.get(entity_id)
        if entry is None:
            continue
        area_id = _effective_area_id(entry, devices.get(entry.device_id) if entry.device_id is not None else None)
        area = areas.get(area_id) if area_id is not None else None
        enriched[entity_id] = replace(
            safe_state,
            area_id=area_id,
            floor_id=area.floor_id if area is not None else None,
            device_id=entry.device_id,
            platform=entry.platform,
            unique_id=entry.unique_id,
        )
    # Preserve any state-bearing entity without a registry entry (None join keys).
    return states | enriched


def _effective_area_id(entry: SafeRegistryEntry, device: SafeDeviceEntry | None) -> str | None:
    """Return the canonical entity-area override or inherited device area."""
    return entry.area_id or (device.area_id if device is not None else None)


def _safe_state(
    state: State,
    entry: er.RegistryEntry | None = None,
    device: dr.DeviceEntry | None = None,
    areas: Mapping[str, ar.AreaEntry] | None = None,
) -> SafeState:
    """Convert a live HA state into a frozen safe state record."""
    context = state.context
    safe_context = SafeContext(
        id=context.id if context else None,
        parent_id=context.parent_id if context else None,
        user_id=context.user_id if context else None,
    )
    area_id = entry.area_id or (device.area_id if device is not None else None) if entry is not None else None
    area = areas.get(area_id) if areas is not None and area_id is not None else None
    return SafeState(
        entity_id=state.entity_id,
        domain=state.domain,
        object_id=state.object_id,
        name=state.name,
        state=state.state,
        attributes=_json_normalized_dict(state.attributes),
        last_changed=state.last_changed.isoformat(),
        last_changed_timestamp=state.last_changed.timestamp(),
        last_reported=_iso(getattr(state, "last_reported", None)),
        last_reported_timestamp=_timestamp(getattr(state, "last_reported", None)),
        last_updated=state.last_updated.isoformat(),
        last_updated_timestamp=state.last_updated.timestamp(),
        context=safe_context,
        area_id=area_id,
        floor_id=area.floor_id if area is not None else None,
        device_id=entry.device_id if entry is not None else None,
        platform=entry.platform if entry is not None else None,
        unique_id=entry.unique_id if entry is not None else None,
    )


def _safe_entity(entry: er.RegistryEntry) -> SafeRegistryEntry:
    """Convert a live entity registry entry into a frozen safe record."""
    return SafeRegistryEntry(
        entity_id=entry.entity_id,
        domain=entry.entity_id.split(".", 1)[0],
        unique_id=entry.unique_id,
        platform=entry.platform,
        config_entry_id=entry.config_entry_id,
        device_id=entry.device_id,
        area_id=entry.area_id,
        name=entry.name,
        original_name=entry.original_name,
        aliases=tuple(sorted(str(a) for a in entry.aliases)),
        labels=tuple(sorted(entry.labels)),
        disabled_by=_enum_value(entry.disabled_by),
        hidden_by=_enum_value(entry.hidden_by),
        entity_category=_enum_value(entry.entity_category),
        device_class=entry.device_class,
        original_device_class=entry.original_device_class,
        capabilities=_json_normalized_dict(entry.capabilities) if entry.capabilities else None,
        supported_features=entry.supported_features,
        translation_key=entry.translation_key,
        has_entity_name=entry.has_entity_name,
    )


def _safe_device(device: dr.DeviceEntry) -> SafeDeviceEntry:
    """Convert a live device registry entry into a frozen safe record."""
    return SafeDeviceEntry(
        id=device.id,
        name=device.name,
        name_by_user=device.name_by_user,
        manufacturer=device.manufacturer,
        model=device.model,
        model_id=device.model_id,
        sw_version=device.sw_version,
        hw_version=device.hw_version,
        serial_number=device.serial_number,
        area_id=device.area_id,
        labels=tuple(sorted(device.labels)),
        identifiers=tuple(tuple(ident) for ident in sorted(device.identifiers)),
        connections=tuple(tuple(conn) for conn in sorted(device.connections)),
        configuration_url=device.configuration_url,
        entry_type=_enum_value(device.entry_type),
        config_entries=tuple(sorted(device.config_entries)),
        via_device_id=device.via_device_id,
        disabled_by=_enum_value(device.disabled_by),
    )


def _safe_area(area: ar.AreaEntry) -> SafeAreaEntry:
    """Convert a live area registry entry into a frozen safe record."""
    return SafeAreaEntry(
        id=area.id,
        area_id=area.id,
        name=area.name,
        aliases=tuple(sorted(area.aliases)),
        floor_id=area.floor_id,
        labels=tuple(sorted(area.labels)),
        icon=area.icon,
        picture=area.picture,
        humidity_entity_id=area.humidity_entity_id,
        temperature_entity_id=area.temperature_entity_id,
        created_at=_iso(area.created_at),
        modified_at=_iso(area.modified_at),
    )


def _safe_floor(floor: fr.FloorEntry) -> SafeFloorEntry:
    """Convert a live floor registry entry into a frozen safe record."""
    return SafeFloorEntry(
        floor_id=floor.floor_id,
        id=floor.floor_id,
        name=floor.name,
        aliases=tuple(sorted(floor.aliases)),
        level=floor.level,
        icon=floor.icon,
        created_at=_iso(floor.created_at),
        modified_at=_iso(floor.modified_at),
    )


def _safe_label(label: lr.LabelEntry) -> SafeLabelEntry:
    """Convert a live label registry entry into a frozen safe record."""
    return SafeLabelEntry(
        label_id=label.label_id,
        name=label.name,
        normalized_name=label.normalized_name,
        description=label.description,
        color=label.color,
        icon=label.icon,
        created_at=_iso(label.created_at),
        modified_at=_iso(label.modified_at),
    )


def _safe_category(scope: str, category: cr_core.CategoryEntry) -> SafeCategoryEntry:
    """Convert a live scoped category entry into a frozen safe record."""
    return SafeCategoryEntry(
        category_id=category.category_id,
        scope=scope,
        name=category.name,
        icon=category.icon,
        created_at=_iso(category.created_at),
        modified_at=_iso(category.modified_at),
    )


def _safe_issue(issue: ir.IssueEntry) -> SafeIssueEntry:
    """Convert a live repairs issue entry into a frozen safe record."""
    return SafeIssueEntry(
        issue_id=issue.issue_id,
        domain=issue.domain,
        severity=_enum_value(issue.severity),
        active=issue.active,
        dismissed_version=issue.dismissed_version,
        translation_key=issue.translation_key,
        translation_placeholders=(
            _json_normalized_dict(issue.translation_placeholders) if issue.translation_placeholders else None
        ),
        created=_iso(issue.created),
    )


def _safe_notification(notification: Notification) -> SafeNotificationEntry:
    """Convert a live persistent notification into a frozen safe record."""
    return SafeNotificationEntry(
        notification_id=notification["notification_id"],
        title=notification["title"],
        message=notification["message"],
        created_at=_iso(notification["created_at"]),
    )


def _safe_config_entry(entry: ConfigEntry[object]) -> SafeConfigEntry:
    """Convert a live config entry into a frozen, secret-stripped record."""
    return SafeConfigEntry(
        entry_id=entry.entry_id,
        domain=entry.domain,
        title=entry.title,
        source=entry.source,
        state=cast(str, _enum_value(entry.state)),
        unique_id=entry.unique_id,
        disabled_by=_enum_value(entry.disabled_by),
        reason=entry.reason,
    )


def _safe_config(hass: HomeAssistant) -> SafeConfig:
    """Convert live Home Assistant config into a frozen safe record."""
    cfg = hass.config
    return SafeConfig(
        location_name=cfg.location_name,
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        elevation=cfg.elevation,
        time_zone=cfg.time_zone,
        language=cfg.language,
        country=cfg.country,
        currency=cfg.currency,
        internal_url=cfg.internal_url,
        external_url=cfg.external_url,
        units=SafeUnitSystem(
            temperature_unit=cfg.units.temperature_unit,
            length_unit=cfg.units.length_unit,
            mass_unit=cfg.units.mass_unit,
            pressure_unit=cfg.units.pressure_unit,
            volume_unit=cfg.units.volume_unit,
            area_unit=cfg.units.area_unit,
            wind_speed_unit=cfg.units.wind_speed_unit,
            accumulated_precipitation_unit=cfg.units.accumulated_precipitation_unit,
        ),
    )


def _build_indexes(
    entities: dict[str, SafeRegistryEntry],
    devices: dict[str, SafeDeviceEntry],
    areas: dict[str, SafeAreaEntry],
) -> SnapshotIndexes:
    """Precompute lookup indexes over the snapshot.

    Effective area for an entity is ``entity.area_id or device.area_id`` so an
    entity-level area override wins and entities otherwise inherit their
    device's area, matching Home Assistant's resolution semantics.
    """
    by_device: dict[str, list[str]] = {}
    by_area: dict[str, list[str]] = {}
    by_config_entry: dict[str, list[str]] = {}
    by_label: dict[str, list[str]] = {}

    for entity_id, entry in entities.items():
        if entry.device_id:
            by_device.setdefault(entry.device_id, []).append(entity_id)
        effective_area = _effective_area_id(
            entry, devices.get(entry.device_id) if entry.device_id is not None else None
        )
        if effective_area is not None:
            by_area.setdefault(effective_area, []).append(entity_id)
        if entry.config_entry_id:
            by_config_entry.setdefault(entry.config_entry_id, []).append(entity_id)
        for label in entry.labels:
            by_label.setdefault(label, []).append(entity_id)

    device_by_area: dict[str, list[str]] = {}
    for device_id, device in devices.items():
        if device.area_id:
            device_by_area.setdefault(device.area_id, []).append(device_id)

    device_by_label: dict[str, list[str]] = {}
    for device_id, device in devices.items():
        for label in device.labels:
            device_by_label.setdefault(label, []).append(device_id)

    area_by_floor: dict[str, list[str]] = {}
    for area_id, area in areas.items():
        if area.floor_id:
            area_by_floor.setdefault(area.floor_id, []).append(area_id)

    return SnapshotIndexes(
        entity_ids_by_device_id=_freeze_index(by_device),
        entity_ids_by_area_id=_freeze_index(by_area),
        device_ids_by_area_id=_freeze_index(device_by_area),
        entity_ids_by_config_entry_id=_freeze_index(by_config_entry),
        entity_ids_by_label=_freeze_index(by_label),
        device_ids_by_label=_freeze_index(device_by_label),
        area_ids_by_floor_id=_freeze_index(area_by_floor),
    )


def _empty_indexes() -> SnapshotIndexes:
    """Return an empty index set for snapshot flavors that omit registry joins."""
    return SnapshotIndexes(
        entity_ids_by_device_id={},
        entity_ids_by_area_id={},
        device_ids_by_area_id={},
        entity_ids_by_config_entry_id={},
        entity_ids_by_label={},
        device_ids_by_label={},
        area_ids_by_floor_id={},
    )


def _freeze_index(index: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    """Freeze an index dict into sorted tuples keyed by id."""
    return {key: tuple(sorted(values)) for key, values in index.items()}


def _iso(value: object) -> str | None:
    """Convert a datetime to an ISO string, preserving None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _json_normalized_dict(value: Mapping[str, object]) -> dict[str, object]:
    """Return a recursive, plain-dict JSON-compatible copy of a mapping."""
    return {str(key): _json_normalize(item) for key, item in value.items()}


def _json_normalize(value: object) -> object:
    """Return a recursive JSON-compatible copy of a snapshot leaf value."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_normalize(value.value)
    if isinstance(value, Mapping):
        return _json_normalized_dict(cast(Mapping[str, object], value))
    if isinstance(value, list | tuple | set):
        return [_json_normalize(item) for item in value]
    return str(value)


def _timestamp(value: object) -> float | None:
    """Convert a datetime to a POSIX timestamp, preserving None."""
    if value is None:
        return None
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())
    return None


def _enum_value(value: object) -> str | None:
    """Convert an enum to its string value, preserving None."""
    if value is None:
        return None
    return getattr(value, "value", str(value))
