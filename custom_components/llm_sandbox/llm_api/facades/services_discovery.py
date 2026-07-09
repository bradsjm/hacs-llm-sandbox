"""Snapshot-derived service discovery facts for the service facade."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...snapshot.models import HomeSnapshot, ServiceTargetBrief
from ..data.selectors import AGGREGATE_SELECTOR_KEYS


@dataclass(frozen=True, slots=True)
class ServiceDiscoveryFacts:
    """Minimal service-discovery facts derived from one frozen snapshot."""

    entities: Mapping[str, _ServiceEntityFacts]
    entity_ids_by_device_id: Mapping[str, tuple[str, ...]]
    entity_ids_by_area_id: Mapping[str, tuple[str, ...]]
    entity_ids_by_label: Mapping[str, tuple[str, ...]]
    area_ids_by_floor_id: Mapping[str, tuple[str, ...]]
    services_target: Mapping[str, Mapping[str, ServiceTargetBrief]]


@dataclass(frozen=True, slots=True)
class _ServiceEntityFacts:
    """Entity facts needed for service-target discovery matching."""

    domain: str
    device_class: str | None
    supported_features: int
    integration: str | None


def service_discovery_facts(snapshot: HomeSnapshot) -> ServiceDiscoveryFacts:
    """Build the bounded facts ``hass.services`` needs for sync discovery."""
    return ServiceDiscoveryFacts(
        entities={
            entity_id: _ServiceEntityFacts(
                domain=state.domain,
                device_class=_discovery_device_class(state.attributes, snapshot.entities.get(entity_id)),
                supported_features=_discovery_supported_features(state.attributes, snapshot.entities.get(entity_id)),
                integration=_discovery_integration(state.platform, snapshot.entities.get(entity_id)),
            )
            for entity_id, state in snapshot.states.items()
        },
        entity_ids_by_device_id=dict(snapshot.indexes.entity_ids_by_device_id),
        entity_ids_by_area_id=dict(snapshot.indexes.entity_ids_by_area_id),
        entity_ids_by_label=dict(snapshot.indexes.entity_ids_by_label),
        area_ids_by_floor_id=dict(snapshot.indexes.area_ids_by_floor_id),
        services_target=dict(snapshot.services_target),
    )


def _discovery_device_class(attributes: Mapping[str, object], entry: object) -> str | None:
    """Return only the device-class fact used by service-target matching."""
    value = attributes.get("device_class")
    if isinstance(value, str):
        return value
    return getattr(entry, "device_class", None) or getattr(entry, "original_device_class", None)


def _discovery_supported_features(attributes: Mapping[str, object], entry: object) -> int:
    """Return only the supported-features fact used by service-target matching."""
    value = attributes.get("supported_features")
    if isinstance(value, int):
        return value
    entry_features = getattr(entry, "supported_features", 0)
    return entry_features if isinstance(entry_features, int) else 0


def _discovery_integration(state_platform: str | None, entry: object) -> str | None:
    """Return only the integration/platform fact used by service-target matching."""
    entry_platform = getattr(entry, "platform", None)
    return entry_platform if isinstance(entry_platform, str) else state_platform


def _target_values(value: object) -> list[str]:
    """Return HA target selector values as strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _expand_target_entities(facts: ServiceDiscoveryFacts, target: Mapping[str, object] | None) -> set[str]:
    """Resolve HA target selectors to visible entity ids for read-only discovery.

    Unlike ``_visible_target`` this performs no fuzzy auto-resolution and records
    no action: it is a pure selector expansion used by ``async_services_for_target``.
    """
    if not isinstance(target, Mapping):
        return set()
    entity_ids: set[str] = set()
    if "entity_id" in target:
        for entity_id in _target_values(target["entity_id"]):
            if entity_id in facts.entities:
                entity_ids.add(entity_id)
    for requested_expansions in _expand_discovery_selectors(facts, target).values():
        for _requested, resolved in requested_expansions:
            entity_ids.update(resolved)
    return entity_ids


def _expand_discovery_selectors(
    facts: ServiceDiscoveryFacts,
    target: Mapping[str, object],
) -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """Expand aggregate selectors from bounded service-discovery facts."""
    expansions: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    for selector in AGGREGATE_SELECTOR_KEYS:
        if selector not in target:
            continue
        resolved_values = tuple(
            (requested, resolved)
            for requested in _target_values(target[selector])
            if (resolved := _expand_discovery_selector(facts, selector, requested))
        )
        if resolved_values:
            expansions[selector] = resolved_values
    return expansions


def _expand_discovery_selector(facts: ServiceDiscoveryFacts, selector: str, value: object) -> tuple[str, ...]:
    """Expand one aggregate selector without access to the full snapshot."""
    requested = str(value)
    if selector == "device_id":
        return tuple(facts.entity_ids_by_device_id.get(requested, ()))
    if selector == "area_id":
        return tuple(facts.entity_ids_by_area_id.get(requested, ()))
    if selector in {"label", "label_id"}:
        return tuple(facts.entity_ids_by_label.get(requested, ()))
    if selector == "floor_id":
        entity_ids: list[str] = []
        for area_id in facts.area_ids_by_floor_id.get(requested, ()):
            entity_ids.extend(facts.entity_ids_by_area_id.get(area_id, ()))
        return tuple(entity_ids)
    return ()


def _services_for_entity(facts: ServiceDiscoveryFacts, entity_id: str) -> tuple[str, ...]:
    """Return service ids whose bounded target facts accept the entity."""
    entity = facts.entities.get(entity_id)
    if entity is None:
        return ()
    matched: list[str] = []
    for domain, service_briefs in facts.services_target.items():
        for service, brief in service_briefs.items():
            if _service_targets_entity(brief, entity):
                matched.append(f"{domain}.{service}")
    return tuple(sorted(matched))


def _service_targets_entity(brief: ServiceTargetBrief, entity: _ServiceEntityFacts) -> bool:
    """Whether a service target accepts an entity using bounded facts only."""
    filters = brief.get("entity")
    if not isinstance(filters, list) or not filters:
        return True
    return any(_service_target_filter_matches(target_filter, entity) for target_filter in filters)


def _service_target_filter_matches(target_filter: Mapping[str, object], entity: _ServiceEntityFacts) -> bool:
    """Mirror HA service target filtering without retaining state/registry records."""
    domains = target_filter.get("domain")
    if isinstance(domains, list) and domains and entity.domain not in domains:
        return False
    integration = target_filter.get("integration")
    if isinstance(integration, str) and integration and entity.integration != integration:
        return False
    device_classes = target_filter.get("device_class")
    if isinstance(device_classes, list) and device_classes and entity.device_class not in device_classes:
        return False
    features = target_filter.get("supported_features")
    return not (
        isinstance(features, list)
        and features
        and not any(
            isinstance(feature, int) and feature & entity.supported_features == feature for feature in features
        )
    )
