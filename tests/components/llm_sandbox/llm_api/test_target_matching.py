"""Unit tests for the pure snapshot-side target matcher.

The matcher replicates Home Assistant's ``async_get_services_for_target`` target
filters plus each service field's ``filter`` (capability) rules, operating only
on frozen snapshot records.
"""

from collections.abc import Mapping

from custom_components.llm_sandbox.llm_api.target_matching import (
    entities_for_service,
    field_filter_matches,
    service_accepts_domain,
    service_field_names,
    service_targets_entity,
    services_for_entity,
)
from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeConfig,
    SafeContext,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    ServiceFieldFilter,
    ServiceTargetBrief,
    SnapshotIndexes,
)


def _state(entity_id: str, attributes: Mapping[str, object] | None = None) -> SafeState:
    """Build a visible state record carrying capability attributes."""
    domain, object_id = entity_id.split(".", 1)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=object_id.replace("_", " ").title(),
        state="off",
        attributes=dict(attributes or {}),
        last_changed="t",
        last_changed_timestamp=0.0,
        last_reported="t",
        last_reported_timestamp=0.0,
        last_updated="t",
        last_updated_timestamp=0.0,
        context=SafeContext(id="c", parent_id=None, user_id=None),
        area_id=None,
        device_id=None,
        platform=domain,
        unique_id=entity_id,
    )


def _entry(entity_id: str, supported_features: int = 0, device_class: str | None = None) -> SafeRegistryEntry:
    """Build a registry entry mirroring capability fields HA matches against."""
    domain, _ = entity_id.split(".", 1)
    return SafeRegistryEntry(
        entity_id=entity_id,
        domain=domain,
        unique_id=entity_id,
        platform=domain,
        config_entry_id=None,
        device_id=None,
        area_id=None,
        name=None,
        original_name=None,
        aliases=(),
        labels=(),
        disabled_by=None,
        hidden_by=None,
        entity_category=None,
        device_class=device_class,
        original_device_class=None,
        capabilities=None,
        supported_features=supported_features,
        translation_key=None,
        has_entity_name=False,
    )


def _snapshot(
    states: Mapping[str, SafeState],
    entities: Mapping[str, SafeRegistryEntry] | None = None,
    services_target: Mapping[str, Mapping[str, ServiceTargetBrief]] | None = None,
    services_schema: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> HomeSnapshot:
    """Build a minimal snapshot populated with matcher-relevant data."""
    return HomeSnapshot(
        created_at="t",
        states=dict(states),
        entities=dict(entities or {}),
        devices={},
        areas={},
        floors={},
        config=SafeConfig(
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
        ),
        services={},
        services_supports_response={},
        indexes=SnapshotIndexes(
            entity_ids_by_device_id={},
            entity_ids_by_area_id={},
            device_ids_by_area_id={},
            entity_ids_by_config_entry_id={},
            entity_ids_by_label={},
            device_ids_by_label={},
            area_ids_by_floor_id={},
        ),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema=services_schema or {},
        services_target=services_target or {},
    )


def test_services_for_entity_includes_targeted_and_accepts_any_services() -> None:
    """A service whose target accepts the entity is listed; accepts-any always matches."""
    snapshot = _snapshot(
        states={"light.color_bulb": _state("light.color_bulb"), "switch.outlet": _state("switch.outlet")},
        services_target={
            "light": {
                "turn_on": {"entity": [{"domain": ["light"]}]},
                "turn_off": {"entity": [{"domain": ["light"]}]},
            },
            "homeassistant": {"turn_on": {"entity": []}},
        },
    )

    assert services_for_entity(snapshot, "light.color_bulb") == (
        "homeassistant.turn_on",
        "light.turn_off",
        "light.turn_on",
    )
    # An accepts-any service still applies; light-domain services do not.
    assert services_for_entity(snapshot, "switch.outlet") == ("homeassistant.turn_on",)


def test_services_for_entity_skips_services_without_target_metadata() -> None:
    """Services absent from services_target are not entity-targeting and are excluded."""
    snapshot = _snapshot(
        states={"light.color_bulb": _state("light.color_bulb")},
        services_target={"light": {"turn_on": {"entity": [{"domain": ["light"]}]}}},
    )

    assert services_for_entity(snapshot, "light.color_bulb") == ("light.turn_on",)


def test_entities_for_service_returns_matching_entities_or_empty_when_unknown() -> None:
    """The inverse matcher returns visible matches; () means no metadata (no signal)."""
    snapshot = _snapshot(
        states={
            "light.color_bulb": _state("light.color_bulb"),
            "light.plain_bulb": _state("light.plain_bulb"),
            "switch.outlet": _state("switch.outlet"),
        },
        services_target={"light": {"turn_on": {"entity": [{"domain": ["light"]}]}}},
    )

    assert entities_for_service(snapshot, "light", "turn_on") == ("light.color_bulb", "light.plain_bulb")
    assert entities_for_service(snapshot, "script", "reload") == ()


def test_service_targets_entity_matches_via_device_class_and_supported_features() -> None:
    """Entity filters honour device_class and supported_features bit constants."""
    brief: ServiceTargetBrief = {
        "entity": [{"domain": ["cover"], "device_class": ["blind"], "supported_features": [4]}]
    }
    blind_with_stop = _state("cover.window", {"supported_features": 4, "device_class": "blind"})
    blind_without_stop = _state("cover.skylight", {"supported_features": 0, "device_class": "blind"})
    shutter = _state("cover.door", {"supported_features": 4, "device_class": "shutter"})

    assert service_targets_entity(brief, blind_with_stop, None) is True
    assert service_targets_entity(brief, blind_without_stop, None) is False
    assert service_targets_entity(brief, shutter, None) is False


def test_service_targets_entity_falls_back_to_registry_when_attributes_absent() -> None:
    """device_class/supported_features resolve via the registry entry when state lacks them."""
    brief: ServiceTargetBrief = {"entity": [{"domain": ["cover"], "supported_features": [4]}]}
    state = _state("cover.window")
    entry = _entry("cover.window", supported_features=4)

    assert service_targets_entity(brief, state, entry) is True


def test_service_accepts_domain_reports_stable_domain_signal() -> None:
    """The domain check is the conservative pre-block signal (True/False/None)."""
    constrained: ServiceTargetBrief = {"entity": [{"domain": ["light"]}]}
    unconstrained: ServiceTargetBrief = {"entity": [{"device_class": ["outlet"]}]}
    accepts_any: ServiceTargetBrief = {"entity": []}

    assert service_accepts_domain(constrained, "light") is True
    assert service_accepts_domain(constrained, "switch") is False
    # No domain constraint -> no stable mismatch to block on.
    assert service_accepts_domain(unconstrained, "switch") is None
    assert service_accepts_domain(accepts_any, "media_player") is None


def test_field_filter_matches_supported_features_and_attribute_intersection() -> None:
    """Field filters use bit features and attribute value-intersection (color modes)."""
    color_temp: ServiceFieldFilter = {"attribute": {"supported_color_modes": ["color_temp", "xy"]}}
    transition: ServiceFieldFilter = {"supported_features": [2]}

    color_bulb = _state("light.bulb", {"supported_color_modes": ["color_temp"], "supported_features": 2})
    plain_bulb = _state("light.plain", {"supported_color_modes": ["onoff"], "supported_features": 0})

    assert field_filter_matches(color_temp, color_bulb, None) is True
    assert field_filter_matches(color_temp, plain_bulb, None) is False
    assert field_filter_matches(transition, color_bulb, None) is True
    assert field_filter_matches(transition, plain_bulb, None) is False


def test_field_filter_matches_dict_attribute_fails_closed() -> None:
    """Dict-valued attributes do not match scalar/list filters and do not raise."""
    field_filter: ServiceFieldFilter = {"attribute": {"effect_options": ["rainbow"]}}
    state = _state("light.bulb", {"effect_options": {"preset": "rainbow"}})

    assert field_filter_matches(field_filter, state, None) is False


def test_service_field_names_filters_by_capability() -> None:
    """The color-temperature example: only capability-supporting fields are returned."""
    snapshot = _snapshot(
        states={
            "light.color_bulb": _state("light.color_bulb", {"supported_color_modes": ["color_temp", "xy"]}),
            "light.plain_bulb": _state("light.plain_bulb", {"supported_color_modes": ["onoff"]}),
        },
        services_schema={
            "light": {
                "turn_on": {
                    "fields": [
                        {
                            "name": "brightness",
                            "filter": {"attribute": {"supported_color_modes": ["brightness", "color_temp"]}},
                        },
                        {
                            "name": "color_temp_kelvin",
                            "filter": {"attribute": {"supported_color_modes": ["color_temp"]}},
                        },
                        {"name": "effect"},
                    ],
                    "dynamic": False,
                }
            }
        },
    )

    assert service_field_names(snapshot, "light", "turn_on", "light.color_bulb") == (
        "brightness",
        "color_temp_kelvin",
        "effect",
    )
    assert service_field_names(snapshot, "light", "turn_on", "light.plain_bulb") == ("effect",)
