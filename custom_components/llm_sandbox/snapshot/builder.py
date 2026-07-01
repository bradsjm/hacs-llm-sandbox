"""Build a frozen Home Assistant snapshot for Monty execution.

Runs on the event loop at the start of each ``execute_home_code`` tool call,
reads all registries and the state machine, and returns an immutable
``HomeSnapshot``. Optional scope filtering is applied at build time as a
noise-reduction measure, not a security boundary. Visibility restrictions are
combined additively over state-bearing entity ids, and the service catalog is
never filtered.
"""

from dataclasses import replace
from typing import cast

import voluptuous as vol
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
from homeassistant.helpers.service import async_get_cached_service_description
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
    ServiceFieldBrief,
    ServiceSchemaBrief,
    SnapshotIndexes,
    SnapshotScope,
)

SERVICE_SCHEMA_FIELD_LIMIT = 12


def build_snapshot(
    hass: HomeAssistant,
    scope: SnapshotScope = DEFAULT_SCOPE,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Build a frozen snapshot of states, registries, and services."""
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)
    floor_registry = fr.async_get(hass)
    label_registry = lr.async_get(hass)
    category_registry = cr_core.async_get(hass)
    issue_registry = ir.async_get(hass)

    states = {s.entity_id: _safe_state(s) for s in hass.states.async_all()}
    entities = {entry.entity_id: _safe_entity(entry) for entry in entity_registry.entities.values()}
    devices = {device.id: _safe_device(device) for device in device_registry.devices.values()}
    areas = {area.id: _safe_area(area) for area in area_registry.async_list_areas()}
    floors = {floor.floor_id: _safe_floor(floor) for floor in floor_registry.async_list_floors()}
    labels = {label.label_id: _safe_label(label) for label in label_registry.async_list_labels()}
    categories = {
        scope: {cid: _safe_category(scope, cat) for cid, cat in entries.items()}
        for scope, entries in category_registry.categories.items()
    }
    issues = [_safe_issue(issue) for issue in issue_registry.issues.values()]
    notification_store = hass.data.get(PERSISTENT_NOTIFICATION_DOMAIN)
    notifications = (
        [_safe_notification(notification) for notification in notification_store.values()]
        if isinstance(notification_store, dict)
        else []
    )
    config_entries = [_safe_config_entry(entry) for entry in hass.config_entries.async_entries()]

    service_catalog, service_response, services_schema = _safe_services(hass)

    # Visibility restrictions always run through one combined predicate so the
    # no-restrictions scope still derives from state-bearing entities.
    visible = _visible_entity_ids(hass, states, entities, scope)
    states, entities, devices, areas, floors, indexes = _finalize_snapshot_collections(
        states,
        entities,
        devices,
        areas,
        floors,
        visible=visible,
        anchor_device_id=anchor_device_id,
    )

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
    # Fill registry-derived join keys (effective area, device, platform, unique_id)
    # now that the filtered entity/device collections exist, mirroring the index rule.
    states = enrich_states(states, entities, devices)
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
    return entry.entity_category not in scope.excluded_entity_categories


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
        area_id = entry.area_id
        if area_id is None and entry.device_id is not None:
            device = devices.get(entry.device_id)
            if device is not None:
                area_id = device.area_id
        enriched[entity_id] = replace(
            safe_state,
            area_id=area_id,
            device_id=entry.device_id,
            platform=entry.platform,
            unique_id=entry.unique_id,
        )
    # Preserve any state-bearing entity without a registry entry (None join keys).
    return states | enriched


def _safe_state(state: State) -> SafeState:
    """Convert a live HA state into a frozen safe state record."""
    context = state.context
    safe_context = SafeContext(
        id=context.id if context else None,
        parent_id=context.parent_id if context else None,
        user_id=context.user_id if context else None,
    )
    return SafeState(
        entity_id=state.entity_id,
        domain=state.domain,
        object_id=state.object_id,
        name=state.name,
        state=state.state,
        attributes=dict(state.attributes),
        last_changed=state.last_changed.isoformat(),
        last_reported=_iso(getattr(state, "last_reported", None)),
        last_updated=state.last_updated.isoformat(),
        context=safe_context,
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
        capabilities=dict(entry.capabilities) if entry.capabilities else None,
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
        translation_placeholders=(dict(issue.translation_placeholders) if issue.translation_placeholders else None),
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


def _safe_services(
    hass: HomeAssistant,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, dict[str, str]],
    dict[str, dict[str, ServiceSchemaBrief]],
]:
    """Snapshot the service catalog as domain -> service names.

    Returns ``(services, services_supports_response, services_schema)`` where
    response support preserves each service's enum value as a JSON-safe string,
    and service schemas are eager JSON-safe parameter briefs.
    """
    catalog = hass.services.async_services()
    services: dict[str, tuple[str, ...]] = {}
    supports_response: dict[str, dict[str, str]] = {}
    services_schema: dict[str, dict[str, ServiceSchemaBrief]] = {}
    for domain, domain_services in catalog.items():
        names: list[str] = []
        response_values: dict[str, str] = {}
        schema_values: dict[str, ServiceSchemaBrief] = {}
        for service_name, service in domain_services.items():
            names.append(service_name)
            response_values[service_name] = service.supports_response.value
            schema_values[service_name] = _service_schema_brief(hass, domain, service_name, service.schema)
        services[domain] = tuple(sorted(names))
        supports_response[domain] = {name: response_values[name] for name in sorted(response_values)}
        services_schema[domain] = {name: schema_values[name] for name in sorted(schema_values)}
    return services, supports_response, services_schema


def _service_schema_brief(
    hass: HomeAssistant,
    domain: str,
    service_name: str,
    schema: object,
) -> ServiceSchemaBrief:
    """Build a JSON-safe brief for one service schema."""
    if schema is None:
        return {"fields": [], "dynamic": False}

    raw_schema = (
        schema.schema
        if isinstance(schema, vol.Schema)
        else vol.Schema(schema).schema
        if isinstance(schema, dict)
        else None
    )
    if not isinstance(raw_schema, dict):
        return {"fields": [], "dynamic": True}

    description_fields = _service_description_fields(hass, domain, service_name)
    fields: list[ServiceFieldBrief] = []
    dynamic = False
    for key, validator in raw_schema.items():
        name, required = _service_field_name_and_required(key)
        if name is None:
            dynamic = True
            continue

        field_description = description_fields.get(name, {})
        # Selector metadata and non-primitive validators often encode dynamic
        # value spaces; keep the brief coarse and mark the schema as dynamic.
        dynamic = dynamic or "selector" in field_description or not _is_plain_service_validator(validator)
        description = field_description.get("description")
        fields.append(
            {
                "name": name,
                "required": required,
                "type_hint": _service_type_hint(validator),
                "description": description if isinstance(description, str) else None,
            }
        )

    fields = sorted(fields, key=lambda field: str(field["name"]))
    if len(fields) > SERVICE_SCHEMA_FIELD_LIMIT:
        dynamic = True
        fields = fields[:SERVICE_SCHEMA_FIELD_LIMIT]
    return {"fields": fields, "dynamic": dynamic}


def _service_description_fields(hass: HomeAssistant, domain: str, service_name: str) -> dict[str, dict[str, object]]:
    """Return cached services.yaml fields for a service, if HA has them."""
    description = async_get_cached_service_description(hass, domain, service_name)
    if not isinstance(description, dict):
        return {}
    fields = description.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {str(name): field for name, field in fields.items() if isinstance(field, dict)}


def _service_field_name_and_required(key: object) -> tuple[str | None, bool]:
    """Extract a service field name and required flag from a voluptuous key."""
    if isinstance(key, vol.Required):
        name = key.schema
        required = True
    elif isinstance(key, vol.Optional):
        name = key.schema
        required = False
    else:
        # Voluptuous treats bare (non-Marker) schema keys as optional by
        # default; only ``vol.Required`` (or schema-level ``required=True``)
        # makes a field required. Mark plain keys optional to match the
        # validation behavior Home Assistant services actually enforce.
        name = key
        required = False
    return (name, required) if isinstance(name, str) else (None, required)


def _service_type_hint(validator: object, depth: int = 0) -> str | None:
    """Derive a coarse JSON-safe type hint from a voluptuous validator."""
    if depth > 4:
        return None
    primitive_hint = _primitive_type_hint(validator)
    if primitive_hint is not None:
        return primitive_hint

    coerced_type = getattr(validator, "type", None)
    primitive_hint = _primitive_type_hint(coerced_type)
    if primitive_hint is not None:
        return primitive_hint

    child_validators = getattr(validator, "validators", None)
    if isinstance(child_validators, tuple):
        hints = {_service_type_hint(child, depth + 1) for child in child_validators}
        hints.discard(None)
        if hints == {"integer", "number"}:
            return "number"
        if len(hints) == 1:
            return hints.pop()
    return None


def _primitive_type_hint(validator: object) -> str | None:
    """Map simple Python validators and container schemas to coarse types."""
    if validator is str:
        return "string"
    if validator is bool:
        return "boolean"
    if validator is int:
        return "integer"
    if validator is float:
        return "number"
    if isinstance(validator, list | tuple):
        return "array"
    if isinstance(validator, dict):
        return "object"
    return None


def _is_plain_service_validator(validator: object) -> bool:
    """Whether a validator is simple enough that values are not dynamic."""
    if _primitive_type_hint(validator) is not None:
        return True
    coerced_type = getattr(validator, "type", None)
    return _primitive_type_hint(coerced_type) is not None


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
        # Effective area: entity override wins, else device area.
        effective_area = entry.area_id
        if effective_area is None and entry.device_id is not None:
            device = devices.get(entry.device_id)
            if device is not None:
                effective_area = device.area_id
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


def _enum_value(value: object) -> str | None:
    """Convert an enum to its string value, preserving None."""
    if value is None:
        return None
    return getattr(value, "value", str(value))
