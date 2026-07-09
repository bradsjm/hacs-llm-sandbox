"""Pure snapshot-side matching of services to entities.

Replicates Home Assistant's automation target matching
(``homeassistant.components.websocket_api.automation``: ``async_get_services_for_target``)
and the field-level capability filters declared in each service's
``services.yaml``, operating entirely on frozen snapshot records. The matching
inputs are all snapshot-derivable:

* service ``target`` entity filters (``domain``, ``device_class``,
  ``integration``, ``supported_features``) captured per service;
* field ``filter`` rules (``supported_features`` bit constants and/or
  ``attribute`` value-intersection such as ``supported_color_modes``);
* entity facts resolved the same way HA resolves them: state attributes first,
  then the registry entry (``get_device_class`` / ``get_supported_features``).

This module never touches live Home Assistant and performs no I/O. A service
absent from ``snapshot.services_target`` has no declared entity target; the
matchers here treat that as "no signal" (neither match nor mismatch) so callers
can let Home Assistant make the final live decision.
"""

from collections.abc import Mapping
from typing import cast

from ..snapshot.models import (
    HomeSnapshot,
    SafeRegistryEntry,
    SafeState,
    ServiceFieldFilter,
    ServiceTargetBrief,
    ServiceTargetFilter,
)

__all__ = (
    "entities_for_service",
    "field_filter_matches",
    "raw_service_field_names",
    "service_accepts_domain",
    "service_field_names",
    "service_targets_entity",
    "services_for_entity",
)


def _entity_device_class(state: SafeState, entry: SafeRegistryEntry | None) -> str | None:
    """Mirror ``homeassistant.helpers.entity.get_device_class`` over the snapshot.

    State attribute wins; otherwise the registry's effective device class
    (``device_class`` then ``original_device_class``) is used.
    """
    value = state.attributes.get("device_class")
    if isinstance(value, str):
        return value
    if entry is not None:
        return entry.device_class or entry.original_device_class
    return None


def _entity_supported_features(state: SafeState, entry: SafeRegistryEntry | None) -> int:
    """Mirror ``homeassistant.helpers.entity.get_supported_features`` over the snapshot.

    State attribute wins; otherwise the registry bitfield is used.
    """
    value = state.attributes.get("supported_features")
    if isinstance(value, int):
        return value
    if entry is not None and entry.supported_features:
        return entry.supported_features
    return 0


def _entity_integration(state: SafeState, entry: SafeRegistryEntry | None) -> str | None:
    """Return the integration (platform) that owns the entity, mirroring entity_sources."""
    if entry is not None and entry.platform:
        return entry.platform
    return state.platform


def _coerce(value: object) -> object:
    """Resolve enum members to their underlying value for set comparison."""
    return getattr(value, "value", value)


def _filter_matches(
    target_filter: ServiceTargetFilter,
    state: SafeState,
    entry: SafeRegistryEntry | None,
) -> bool:
    """Whether one entity-filter dict accepts the entity (all constraints must hold).

    Mirrors ``_EntityFilter.matches`` in ``websocket_api/automation.py``: an empty
    constraint accepts everything; ``supported_features`` matches when the entity's
    bitfield includes any of the filter's feature constants.
    """
    domains = target_filter.get("domain")
    if isinstance(domains, list | tuple) and domains and state.domain not in domains:
        return False

    integration = target_filter.get("integration")
    if isinstance(integration, str) and integration and _entity_integration(state, entry) != integration:
        return False

    device_classes = target_filter.get("device_class")
    if isinstance(device_classes, list | tuple) and device_classes:
        device_class = _entity_device_class(state, entry)
        if device_class is None or device_class not in device_classes:
            return False

    features = target_filter.get("supported_features")
    if isinstance(features, list | tuple) and features:
        supported = _entity_supported_features(state, entry)
        # ``feature & entity_features == feature`` checks the entity has every bit.
        if not any(isinstance(feature, int) and feature & supported == feature for feature in features):
            return False

    return True


def service_targets_entity(
    brief: ServiceTargetBrief,
    state: SafeState,
    entry: SafeRegistryEntry | None,
) -> bool:
    """Whether a service's captured target accepts this entity (HA semantics).

    A target with no entity filters (``entity`` absent or empty) matches any
    entity, matching HA's ``_AutomationComponentLookupData.matches`` returning
    ``True`` when ``filters`` is empty.
    """
    filters = brief.get("entity")
    if not isinstance(filters, list | tuple) or not filters:
        return True
    return any(_filter_matches(target_filter, state, entry) for target_filter in filters)


def service_accepts_domain(brief: ServiceTargetBrief, domain: str) -> bool | None:
    """Stable-fact domain check used for conservative pre-blocking.

    Returns ``True`` when the service accepts the domain, ``False`` when the
    target declares domains and the domain is excluded, or ``None`` when the
    target does not constrain domains (no stable mismatch to block on).
    """
    filters = brief.get("entity")
    if not isinstance(filters, list | tuple) or not filters:
        return None
    constrains_domain = False
    for target_filter in filters:
        domains = target_filter.get("domain")
        if isinstance(domains, list | tuple) and domains:
            constrains_domain = True
            if domain in domains:
                return True
    return False if constrains_domain else None


def field_filter_matches(
    field_filter: ServiceFieldFilter,
    state: SafeState,
    entry: SafeRegistryEntry | None,
) -> bool:
    """Whether an entity satisfies a service field's capability filter.

    ``supported_features`` mirrors the entity-feature bit test; ``attribute``
    (e.g. ``supported_color_modes``) matches when the entity's attribute value
    intersects the filter's allowed values.
    """
    features = field_filter.get("supported_features")
    if isinstance(features, list | tuple) and features:
        supported = _entity_supported_features(state, entry)
        if not any(isinstance(feature, int) and feature & supported == feature for feature in features):
            return False

    attribute = field_filter.get("attribute")
    if isinstance(attribute, Mapping) and attribute:
        for attr_name, allowed in attribute.items():
            raw_value = state.attributes.get(attr_name)
            if isinstance(raw_value, Mapping):
                return False
            values = (
                raw_value
                if isinstance(raw_value, list | tuple | set)
                else ([raw_value] if raw_value is not None else [])
            )
            entity_values = {_coerce(value) for value in values}
            allowed_values = {_coerce(value) for value in allowed}
            if not entity_values.intersection(allowed_values):
                return False

    return True


def services_for_entity(snapshot: HomeSnapshot, entity_id: str) -> tuple[str, ...]:
    """Return service ids (``domain.service``) whose target accepts the entity."""
    state = snapshot.states.get(entity_id)
    if state is None:
        return ()
    entry = snapshot.entities.get(entity_id)
    matched: list[str] = []
    for domain, service_briefs in snapshot.services_target.items():
        for service, brief in service_briefs.items():
            if service_targets_entity(brief, state, entry):
                matched.append(f"{domain}.{service}")
    return tuple(sorted(matched))


def entities_for_service(
    snapshot: HomeSnapshot,
    domain: str,
    service: str,
) -> tuple[str, ...]:
    """Return visible entity ids the service targets, or ``()`` when unknown.

    ``()`` means the service has no captured target metadata, so there is no
    preferential set for ranking and Home Assistant should decide live.
    """
    brief = snapshot.services_target.get(domain, {}).get(service)
    if brief is None:
        return ()
    matched = [
        entity_id
        for entity_id, state in snapshot.states.items()
        if service_targets_entity(brief, state, snapshot.entities.get(entity_id))
    ]
    return tuple(sorted(matched))


def service_field_names(
    snapshot: HomeSnapshot,
    domain: str,
    service: str,
    entity_id: str,
) -> tuple[str, ...] | None:
    """Return field names the entity is known to support for a service.

    Capability-filtered via each field's declared ``filter``. Returns ``None``
    when the service has no schema brief (unknown surface); returns all field
    names when the service declares no per-field filters.
    """
    state = snapshot.states.get(entity_id)
    if state is None:
        return None
    entry = snapshot.entities.get(entity_id)
    brief = snapshot.services_schema.get(domain, {}).get(service)
    if not isinstance(brief, Mapping):
        return None
    supported: list[str] = []
    for raw_field in raw_service_field_names(brief):
        name = cast(str, raw_field["name"])
        field_filter = raw_field.get("filter")
        if not isinstance(field_filter, Mapping) or field_filter_matches(
            cast(ServiceFieldFilter, field_filter), state, entry
        ):
            supported.append(name)
    return tuple(supported)


def raw_service_field_names(brief: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Return named service-field dicts from a captured schema brief."""
    raw_fields = brief.get("fields")
    if not isinstance(raw_fields, list | tuple):
        return ()
    return tuple(
        raw_field for raw_field in raw_fields if isinstance(raw_field, dict) and isinstance(raw_field.get("name"), str)
    )
