"""Build a frozen Home Assistant snapshot for Monty execution.

Runs on the event loop at the start of each ``execute_home_code`` tool call,
reads all registries and the state machine, and returns an immutable
``HomeSnapshot``. Optional scope filtering is applied at build time as a
noise-reduction measure, not a security boundary. Visibility restrictions are
combined additively over state-bearing entity ids, and the service catalog is
never filtered.
"""

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.util import dt as dt_util

from .models import (
    DEFAULT_SCOPE,
    HomeSnapshot,
    SafeAreaEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeRegistryEntry,
    SafeState,
    SnapshotIndexes,
    SnapshotScope,
)


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

    states = {s.entity_id: _safe_state(s) for s in hass.states.async_all()}
    entities = {entry.entity_id: _safe_entity(entry) for entry in entity_registry.entities.values()}
    devices = {device.id: _safe_device(device) for device in device_registry.devices.values()}
    areas = {area.id: _safe_area(area) for area in area_registry.async_list_areas()}
    floors = {floor.floor_id: _safe_floor(floor) for floor in floor_registry.async_list_floors()}

    service_catalog, service_response = _safe_services(hass)

    # Visibility restrictions always run through one combined predicate so the
    # no-restrictions scope still derives from state-bearing entities.
    visible = _visible_entity_ids(hass, states, entities, scope)
    states, entities, devices, areas, floors = _derive_collections(
        states,
        entities,
        devices,
        areas,
        floors,
        visible,
        anchor_device_id,
    )
    indexes = _build_indexes(entities, devices, areas)

    return HomeSnapshot(
        created_at=dt_util.utcnow().isoformat(),
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        services=service_catalog,
        services_supports_response=service_response,
        indexes=indexes,
    )


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


def _safe_services(hass: HomeAssistant) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, str]]]:
    """Snapshot the service catalog as domain -> service names.

    Returns (services, services_supports_response) where the second mapping
    preserves each service's ``supports_response`` enum value as a JSON-safe
    string.
    """
    catalog = hass.services.async_services()
    services: dict[str, tuple[str, ...]] = {}
    supports_response: dict[str, dict[str, str]] = {}
    for domain, domain_services in catalog.items():
        names: list[str] = []
        response_values: dict[str, str] = {}
        for service_name, service in domain_services.items():
            names.append(service_name)
            # Preserve the HA enum value instead of collapsing optional/only
            # services into a bool; the facade uses the value for HA-parity
            # propose-only validation.
            response_values[service_name] = service.supports_response.value
        services[domain] = tuple(sorted(names))
        supports_response[domain] = {name: response_values[name] for name in sorted(response_values)}
    return services, supports_response


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
