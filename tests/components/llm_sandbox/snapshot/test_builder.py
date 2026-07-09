"""Tests for the snapshot builder."""

from datetime import UTC, datetime
from typing import cast

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.snapshot import (
    DEFAULT_SCOPE,
    SnapshotScope,
    build_recorder_snapshot,
    build_snapshot,
    build_vision_snapshot,
)
from homeassistant.components.homeassistant.const import DATA_EXPOSED_ENTITIES
from homeassistant.components.homeassistant.exposed_entities import ExposedEntities, async_expose_entity
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers.service import async_set_service_schema
from pytest_homeassistant_custom_component.common import MockConfigEntry

DEFAULT_PRODUCT_SCOPE = SnapshotScope(
    assistant="conversation",
    restrict_to_assist_exposed=False,
    exclude_hidden=True,
    excluded_entity_categories=frozenset({"config"}),
    include_all_diagnostics=False,
)


def _add_device_owner_entry(hass: HomeAssistant) -> str:
    """Add a mock config entry that can own device-registry records."""
    entry = MockConfigEntry(domain="test", title="Device Owner")
    entry.add_to_hass(hass)
    return str(entry.entry_id)


def _add_entity(
    hass: HomeAssistant,
    entity_id: str,
    unique_id: str,
    *,
    device_id: str | None = None,
    area_id: str | None = None,
    entity_category: er.EntityCategory | None = None,
    hidden_by: er.RegistryEntryHider | None = None,
    disabled_by: er.RegistryEntryDisabler | None = None,
    device_class: str | None = None,
) -> None:
    """Add a registry entity and matching state unless disabled."""
    domain, object_id = entity_id.split(".", 1)
    registry = er.async_get(hass)
    registry.async_get_or_create(
        domain,
        "test",
        unique_id,
        suggested_object_id=object_id,
        device_id=device_id,
        entity_category=entity_category,
        disabled_by=disabled_by,
    )
    registry.async_update_entity(entity_id, area_id=area_id, hidden_by=hidden_by, device_class=device_class)
    if disabled_by is None:
        hass.states.async_set(entity_id, "on", {"friendly_name": object_id.replace("_", " ").title()})


def _add_area_device(hass: HomeAssistant, area_name: str) -> tuple[str, str, str]:
    """Add a floor, area, and device for scope-filter tests."""
    floor_registry = fr.async_get(hass)
    area_registry = ar.async_get(hass)
    device_registry = dr.async_get(hass)

    floor = floor_registry.async_create(f"{area_name} Floor")
    area = area_registry.async_create(area_name)
    area_registry.async_update(area.id, floor_id=floor.floor_id)
    device = device_registry.async_get_or_create(
        config_entry_id=_add_device_owner_entry(hass),
        identifiers={("test", f"{area_name.lower()}-device")},
    )
    device_registry.async_update_device(device.id, area_id=area.id)
    return floor.floor_id, area.id, device.id


async def test_snapshot_captures_states_and_registry_entries(hass: HomeAssistant) -> None:
    entity_registry = er.async_get(hass)
    entity_registry.async_get_or_create("light", "test", "bedroom", suggested_object_id="bedroom")
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})

    snapshot = build_snapshot(hass)

    assert "light.bedroom" in snapshot.states
    state = snapshot.states["light.bedroom"]
    assert state.state == "on"
    assert state.domain == "light"
    assert state.name == "Bedroom Light"
    assert state.attributes["friendly_name"] == "Bedroom Light"

    assert "light.bedroom" in snapshot.entities
    entry = snapshot.entities["light.bedroom"]
    assert entry.platform == "test"
    assert entry.unique_id == "bedroom"
    assert entry.domain == "light"


async def test_snapshot_area_and_floor_alias_fields_match_canonical(hass: HomeAssistant) -> None:
    """Alias fields (area.area_id, floor.id) mirror their canonical keys."""
    floor_id, _area_id, device_id = _add_area_device(hass, "Loft")

    snapshot = build_snapshot(hass, anchor_device_id=device_id)

    floor = snapshot.floors[floor_id]
    # Canonical floor_id mirrors the denormalized id alias.
    assert floor.id == floor.floor_id == floor_id

    area = next(a for a in snapshot.areas.values() if a.floor_id == floor_id)
    # Canonical id mirrors the denormalized area_id alias.
    assert area.area_id == area.id


async def test_snapshot_datetimes_are_iso_strings(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temp", "23.5", {"friendly_name": "Temp"})

    snapshot = build_snapshot(hass)

    state = snapshot.states["sensor.temp"]
    assert isinstance(state.last_changed, str)
    assert "T" in state.last_changed
    assert isinstance(state.last_updated, str)
    assert isinstance(state.last_changed_timestamp, float)
    assert state.last_changed_timestamp == datetime.fromisoformat(state.last_changed).timestamp()
    assert isinstance(state.last_updated_timestamp, float)
    assert state.last_updated_timestamp == datetime.fromisoformat(state.last_updated).timestamp()
    assert isinstance(state.last_reported_timestamp, float)
    assert state.last_reported is not None
    assert state.last_reported_timestamp == datetime.fromisoformat(state.last_reported).timestamp()


async def test_snapshot_state_attributes_are_json_normalized_deep_copies(hass: HomeAssistant) -> None:
    nested_numbers = [1, 2]
    hass.states.async_set(
        "sensor.deep_payload",
        "on",
        {
            "nested": {"numbers": nested_numbers, "options": ("auto", "eco"), "labels": {"warm", "cool"}},
            "seen_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        },
    )

    snapshot = build_snapshot(hass)
    live_state = hass.states.get("sensor.deep_payload")
    assert live_state is not None

    live_nested = cast(dict[str, object], live_state.attributes["nested"])
    cast(list[int], live_nested["numbers"]).append(3)

    attributes = snapshot.states["sensor.deep_payload"].attributes
    nested = cast(dict[str, object], attributes["nested"])
    assert isinstance(attributes, dict)
    assert isinstance(nested, dict)
    assert isinstance(nested["numbers"], tuple)
    assert nested["numbers"] == (1, 2)
    assert isinstance(nested["options"], tuple)
    assert nested["options"] == ("auto", "eco")
    assert isinstance(nested["labels"], tuple)
    assert set(cast(tuple[str, ...], nested["labels"])) == {"warm", "cool"}
    assert attributes["seen_at"] == "2026-01-02T03:04:05+00:00"


async def test_snapshot_entity_capabilities_are_json_normalized_deep_copies(hass: HomeAssistant) -> None:
    entity_registry = er.async_get(hass)
    entity_registry.async_get_or_create("light", "test", "capable", suggested_object_id="capable")
    entity_registry.async_update_entity(
        "light.capable",
        capabilities={"modes": ["auto"], "range": (1, 2), "flags": {"fast", "quiet"}},
    )
    hass.states.async_set("light.capable", "on")

    snapshot = build_snapshot(hass)
    live_entry = entity_registry.async_get("light.capable")
    assert live_entry is not None
    assert live_entry.capabilities is not None

    cast(list[str], live_entry.capabilities["modes"]).append("manual")

    capabilities = snapshot.entities["light.capable"].capabilities
    assert isinstance(capabilities, dict)
    assert isinstance(capabilities["modes"], tuple)
    assert capabilities["modes"] == ("auto",)
    assert isinstance(capabilities["range"], tuple)
    assert capabilities["range"] == (1, 2)
    assert isinstance(capabilities["flags"], tuple)
    assert set(cast(tuple[str, ...], capabilities["flags"])) == {"fast", "quiet"}


async def test_snapshot_issue_placeholders_are_json_normalized_deep_copies(hass: HomeAssistant) -> None:
    placeholders = cast(
        dict[str, str],
        {"device": "Kitchen", "details": {"ids": ["one"], "aliases": ("main",), "tags": {"urgent"}}},
    )
    ir.async_create_issue(
        hass,
        "test",
        "deep_payload",
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="deep_payload",
        translation_placeholders=placeholders,
    )

    snapshot = build_snapshot(hass)
    issue = ir.async_get(hass).async_get_issue("test", "deep_payload")
    assert issue is not None
    assert issue.translation_placeholders is not None

    live_details = cast(dict[str, object], cast(dict[str, object], issue.translation_placeholders)["details"])
    cast(list[str], live_details["ids"]).append("two")

    safe_issue = next(issue for issue in snapshot.issues if issue.issue_id == "deep_payload")
    safe_placeholders = safe_issue.translation_placeholders
    assert isinstance(safe_placeholders, dict)
    details = cast(dict[str, object], safe_placeholders["details"])
    assert isinstance(details, dict)
    assert isinstance(details["ids"], tuple)
    assert details["ids"] == ("one",)
    assert isinstance(details["aliases"], tuple)
    assert details["aliases"] == ("main",)
    assert isinstance(details["tags"], tuple)
    assert set(cast(tuple[str, ...], details["tags"])) == {"urgent"}


async def test_snapshot_effective_area_index_uses_entity_override_then_device(hass: HomeAssistant) -> None:
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)

    living_room = area_registry.async_create("Living Room")
    bedroom = area_registry.async_create("Bedroom")

    # Device assigned to Living Room.
    device = device_registry.async_get_or_create(
        config_entry_id=_add_device_owner_entry(hass),
        connections={("test", "dev1")},
        identifiers={("test", "dev1")},
    )
    device_registry.async_update_device(device.id, area_id=living_room.id)

    # Entity on the device, but with an explicit area override (Bedroom).
    entity_registry.async_get_or_create(
        "light",
        "test",
        "override_light",
        suggested_object_id="override_light",
        device_id=device.id,
    )
    entity_registry.async_update_entity("light.override_light", area_id=bedroom.id)
    hass.states.async_set("light.override_light", "on")
    # Entity on the same device, no area override -> inherits Living Room.
    entity_registry.async_get_or_create(
        "light",
        "test",
        "inherit_light",
        suggested_object_id="inherit_light",
        device_id=device.id,
    )
    hass.states.async_set("light.inherit_light", "on")

    snapshot = build_snapshot(hass)

    bedroom_entities = snapshot.indexes.entity_ids_by_area_id.get(bedroom.id, ())
    living_room_entities = snapshot.indexes.entity_ids_by_area_id.get(living_room.id, ())

    assert "light.override_light" in bedroom_entities
    assert "light.override_light" not in living_room_entities
    assert "light.inherit_light" in living_room_entities
    override_state = snapshot.states["light.override_light"]
    inherited_state = snapshot.states["light.inherit_light"]
    assert override_state.area_id == bedroom.id
    assert override_state.device_id == device.id
    assert override_state.platform == "test"
    assert override_state.unique_id == "override_light"
    assert inherited_state.area_id == living_room.id
    assert inherited_state.device_id == device.id
    assert inherited_state.platform == "test"
    assert inherited_state.unique_id == "inherit_light"


async def test_snapshot_service_catalog(hass: HomeAssistant) -> None:
    hass.services.async_register("light", "turn_on", lambda call: None)
    hass.services.async_register("light", "turn_off", lambda call: None)
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: None,
        supports_response=SupportsResponse.ONLY,
    )
    await hass.async_block_till_done()

    snapshot = build_snapshot(hass)

    assert "turn_on" in snapshot.services.get("light", ())
    assert "turn_off" in snapshot.services.get("light", ())
    assert snapshot.services_supports_response["light"]["turn_on"] == "none"
    assert snapshot.services_supports_response["test_response"]["required"] == "only"


async def test_snapshot_service_schema_brief(hass: HomeAssistant) -> None:
    hass.services.async_register(
        "schema_test",
        "do_thing",
        lambda call: None,
        schema=vol.Schema(
            {
                vol.Required("count"): vol.Coerce(int),
                vol.Optional("label"): str,
            }
        ),
    )
    await hass.async_block_till_done()

    snapshot = build_snapshot(hass)

    brief = snapshot.services_schema["schema_test"]["do_thing"]
    assert isinstance(brief["dynamic"], bool)
    assert brief["dynamic"] is False
    assert brief["fields"] == (
        {"name": "count", "required": True, "type_hint": "integer", "description": None},
        {"name": "label", "required": False, "type_hint": "string", "description": None},
    )
    assert all(isinstance(field["name"], str) for field in brief["fields"])
    assert all(isinstance(field["required"], bool) for field in brief["fields"])


async def test_snapshot_captures_service_target_and_field_capability_filters(hass: HomeAssistant) -> None:
    """Service target entity-filters and per-field capability filters are captured."""
    hass.services.async_register(
        "cover",
        "stop_cover",
        lambda call: None,
        schema=vol.Schema({vol.Optional("color_temp_kelvin"): int}),
    )
    async_set_service_schema(
        hass,
        "cover",
        "stop_cover",
        {
            "target": {"entity": [{"domain": ["cover"], "supported_features": [4]}]},
            "fields": {
                "color_temp_kelvin": {"filter": {"attribute": {"supported_color_modes": ["color_temp"]}}},
            },
        },
    )
    await hass.async_block_till_done()

    snapshot = build_snapshot(hass)

    assert snapshot.services_target["cover"]["stop_cover"] == {
        "entity": ({"domain": ("cover",), "device_class": (), "integration": None, "supported_features": (4,)},)
    }
    fields = {field["name"]: field for field in snapshot.services_schema["cover"]["stop_cover"]["fields"]}
    assert fields["color_temp_kelvin"]["filter"] == {"attribute": {"supported_color_modes": ("color_temp",)}}


async def test_snapshot_device_label_index(hass: HomeAssistant) -> None:
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=_add_device_owner_entry(hass),
        identifiers={("test", "labelled-device")},
    )
    device_registry.async_update_device(device.id, labels={"fav"})

    # Anchor the device so it survives snapshot visibility filtering:
    # entity-less devices are dropped unless anchored (or linked to a visible entity).
    snapshot = build_snapshot(hass, anchor_device_id=device.id)

    assert "fav" in snapshot.indexes.device_ids_by_label
    assert isinstance(snapshot.indexes.device_ids_by_label["fav"], tuple)
    assert device.id in snapshot.indexes.device_ids_by_label["fav"]


async def test_snapshot_config_freezes_hass_config(hass: HomeAssistant) -> None:
    hass.config.location_name = "Snapshot Home"
    hass.config.country = "BE"

    snapshot = build_snapshot(hass)

    assert snapshot.config.location_name == hass.config.location_name
    assert snapshot.config.time_zone == hass.config.time_zone
    assert snapshot.config.units.temperature_unit == hass.config.units.temperature_unit
    assert isinstance(snapshot.config.country, str | None)


async def test_snapshot_includes_label_registry(hass: HomeAssistant) -> None:
    from homeassistant.helpers import label_registry as lr

    lr.async_get(hass).async_create("Favourites", color="red", icon="mdi:star")

    snapshot = build_snapshot(hass)

    assert snapshot.labels
    label = next(iter(snapshot.labels.values()))
    assert label.name == "Favourites"
    assert label.normalized_name == "favourites"
    assert label.color == "red"
    assert label.icon == "mdi:star"
    assert isinstance(label.created_at, str)


async def test_snapshot_includes_config_entries_without_secrets(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain="test_secret",
        title="Secret Integration",
        data={"api_key": "super-secret-token"},
        options={"password": "hunter2"},
        unique_id="uniq",
    )
    entry.add_to_hass(hass)

    snapshot = build_snapshot(hass)

    matches = [e for e in snapshot.config_entries if e.entry_id == entry.entry_id]
    assert matches
    safe = matches[0]
    assert safe.domain == "test_secret"
    assert safe.title == "Secret Integration"
    assert safe.unique_id == "uniq"
    assert isinstance(safe.state, str)
    # Critical safety: no secret-bearing fields exist on the record.
    assert not hasattr(safe, "data")
    assert not hasattr(safe, "options")
    assert not hasattr(safe, "runtime_data")
    assert not hasattr(safe, "subentries")
    # And the serialized JSON form carries only the allowed keys.
    payload = safe.__llm_sandbox_json__()
    assert set(payload) == {
        "entry_id",
        "domain",
        "title",
        "source",
        "state",
        "unique_id",
        "disabled_by",
        "reason",
    }
    assert "api_key" not in str(payload)
    assert "hunter2" not in str(payload)


async def test_snapshot_device_identifiers_and_connections_are_json_safe(hass: HomeAssistant) -> None:
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=_add_device_owner_entry(hass),
        connections={("mac", "aa:bb:cc:dd:ee:ff")},
        identifiers={("test", "device1")},
    )

    snapshot = build_snapshot(hass, anchor_device_id=device.id)

    # Find the device by its identifier.
    safe_device = next(d for d in snapshot.devices.values() if ("test", "device1") in d.identifiers)
    assert ("test", "device1") in safe_device.identifiers
    assert ("mac", "aa:bb:cc:dd:ee:ff") in safe_device.connections
    # Tuples are JSON-safe (frozen as tuples).
    assert all(isinstance(ident, tuple) for ident in safe_device.identifiers)


@pytest.mark.parametrize(
    ("entity_category", "hidden_by", "expected_visible"),
    [
        pytest.param(None, None, True, id="normal"),
        pytest.param(er.EntityCategory.CONFIG, None, False, id="config"),
        pytest.param(er.EntityCategory.DIAGNOSTIC, None, False, id="diagnostic-without-useful-device-class"),
        pytest.param(None, er.RegistryEntryHider.USER, False, id="hidden"),
    ],
)
async def test_build_snapshot_filters_by_category_and_hidden(
    hass: HomeAssistant,
    entity_category: er.EntityCategory | None,
    hidden_by: er.RegistryEntryHider | None,
    expected_visible: bool,
) -> None:
    _add_entity(
        hass,
        "sensor.scoped_entity",
        "scoped_entity",
        entity_category=entity_category,
        hidden_by=hidden_by,
    )

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert ("sensor.scoped_entity" in snapshot.states) is expected_visible
    assert ("sensor.scoped_entity" in snapshot.entities) is expected_visible


async def test_build_snapshot_visibility_keeps_state_only_entity(hass: HomeAssistant) -> None:
    hass.states.async_set("input_boolean.foo", "on")

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert "input_boolean.foo" in snapshot.states
    assert "input_boolean.foo" not in snapshot.entities


@pytest.mark.parametrize(
    ("device_class", "include_all_diagnostics", "expected_visible"),
    [
        pytest.param("battery", False, True, id="useful-device-class"),
        pytest.param("uptime", False, False, id="non-useful-device-class"),
        pytest.param(None, False, False, id="no-device-class"),
        pytest.param("uptime", True, True, id="include-all"),
    ],
)
async def test_build_snapshot_selective_diagnostic_visibility(
    hass: HomeAssistant,
    device_class: str | None,
    include_all_diagnostics: bool,
    expected_visible: bool,
) -> None:
    _add_entity(
        hass,
        "sensor.diagnostic_detail",
        "diagnostic_detail",
        entity_category=er.EntityCategory.DIAGNOSTIC,
        device_class=device_class,
    )
    scope = SnapshotScope(
        assistant="conversation",
        restrict_to_assist_exposed=False,
        exclude_hidden=True,
        excluded_entity_categories=frozenset({"config"}),
        include_all_diagnostics=include_all_diagnostics,
    )

    snapshot = build_snapshot(hass, scope=scope)

    assert ("sensor.diagnostic_detail" in snapshot.states) is expected_visible
    assert ("sensor.diagnostic_detail" in snapshot.entities) is expected_visible


async def test_build_snapshot_visibility_excludes_entity_drops_orphan_device(hass: HomeAssistant) -> None:
    _floor_id, _area_id, device_id = _add_area_device(hass, "Utility")
    _add_entity(
        hass,
        "sensor.utility_config",
        "utility_config",
        device_id=device_id,
        entity_category=er.EntityCategory.CONFIG,
    )

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert device_id not in snapshot.devices
    assert device_id not in snapshot.indexes.entity_ids_by_device_id
    assert all(device_id not in devices for devices in snapshot.indexes.device_ids_by_area_id.values())


async def test_build_snapshot_disabled_entity_excluded_from_restricted_scope(hass: HomeAssistant) -> None:
    _add_entity(
        hass,
        "sensor.disabled_value",
        "disabled_value",
        disabled_by=er.RegistryEntryDisabler.USER,
    )

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert "sensor.disabled_value" not in snapshot.states
    assert "sensor.disabled_value" not in snapshot.entities


@pytest.mark.parametrize(
    "anchor_has_only_excluded_entity",
    [pytest.param(False, id="no-entities"), pytest.param(True, id="only-excluded-entity")],
)
async def test_build_snapshot_anchor_device_force_included(
    hass: HomeAssistant,
    anchor_has_only_excluded_entity: bool,
) -> None:
    floor_id, area_id, device_id = _add_area_device(hass, "Kitchen")
    if anchor_has_only_excluded_entity:
        _add_entity(
            hass,
            "sensor.kitchen_config",
            "kitchen_config",
            device_id=device_id,
            entity_category=er.EntityCategory.CONFIG,
        )

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE, anchor_device_id=device_id)

    assert device_id in snapshot.devices
    assert area_id in snapshot.areas
    assert floor_id in snapshot.floors
    assert device_id not in snapshot.indexes.entity_ids_by_device_id
    assert device_id in snapshot.indexes.device_ids_by_area_id[area_id]


async def test_build_recorder_snapshot_keeps_selector_surface_without_admin_metadata(hass: HomeAssistant) -> None:
    """Recorder snapshots keep visible selector data but omit services/admin surfaces."""
    floor_id, area_id, device_id = _add_area_device(hass, "Pantry")
    label = lr.async_get(hass).async_create("Recorder Visible", color="blue")
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    device_registry.async_update_device(device_id, labels={label.label_id})
    _add_entity(hass, "sensor.pantry_temperature", "pantry_temperature", device_id=device_id)
    entity_registry.async_update_entity("sensor.pantry_temperature", labels={label.label_id})
    _add_entity(hass, "sensor.hidden_temperature", "hidden_temperature", hidden_by=er.RegistryEntryHider.USER)
    anchor_floor_id, anchor_area_id, anchor_device_id = _add_area_device(hass, "Recorder Anchor")
    hass.services.async_register("light", "turn_on", lambda call: None)
    MockConfigEntry(domain="test_secret", title="Secret").add_to_hass(hass)
    await hass.async_block_till_done()

    snapshot = build_recorder_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE, anchor_device_id=anchor_device_id)

    assert set(snapshot.states) == {"sensor.pantry_temperature"}
    assert set(snapshot.entities) == {"sensor.pantry_temperature"}
    assert device_id in snapshot.devices
    assert area_id in snapshot.areas
    assert floor_id in snapshot.floors
    assert anchor_device_id in snapshot.devices
    assert anchor_area_id in snapshot.areas
    assert anchor_floor_id in snapshot.floors
    assert snapshot.indexes.entity_ids_by_device_id[device_id] == ("sensor.pantry_temperature",)
    assert snapshot.indexes.entity_ids_by_area_id[area_id] == ("sensor.pantry_temperature",)
    assert snapshot.indexes.entity_ids_by_label[label.label_id] == ("sensor.pantry_temperature",)
    assert snapshot.indexes.device_ids_by_label[label.label_id] == (device_id,)
    assert label.label_id in snapshot.labels
    assert snapshot.services == {}
    assert snapshot.services_supports_response == {}
    assert snapshot.services_schema == {}
    assert snapshot.services_target == {}
    assert snapshot.categories == {}
    assert snapshot.issues == ()
    assert snapshot.notifications == ()
    assert snapshot.config_entries == ()


async def test_build_vision_snapshot_keeps_visible_states_only(hass: HomeAssistant) -> None:
    """Vision snapshots contain visible state/config data and empty unused surfaces."""
    _add_entity(hass, "camera.front_door", "front_door")
    _add_entity(hass, "image.porch_snapshot", "porch_snapshot")
    _add_entity(hass, "camera.hidden", "hidden", hidden_by=er.RegistryEntryHider.USER)
    hass.services.async_register("camera", "snapshot", lambda call: None)
    await hass.async_block_till_done()

    snapshot = build_vision_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert set(snapshot.states) == {"camera.front_door", "image.porch_snapshot"}
    assert snapshot.entities == {}
    assert snapshot.devices == {}
    assert snapshot.areas == {}
    assert snapshot.floors == {}
    assert snapshot.labels == {}
    assert snapshot.categories == {}
    assert snapshot.issues == ()
    assert snapshot.notifications == ()
    assert snapshot.config_entries == ()
    assert snapshot.services == {}
    assert snapshot.services_supports_response == {}
    assert snapshot.services_schema == {}
    assert snapshot.services_target == {}
    assert snapshot.indexes.entity_ids_by_device_id == {}
    assert snapshot.indexes.entity_ids_by_area_id == {}
    assert snapshot.indexes.device_ids_by_area_id == {}
    assert snapshot.indexes.entity_ids_by_config_entry_id == {}
    assert snapshot.indexes.entity_ids_by_label == {}
    assert snapshot.indexes.device_ids_by_label == {}
    assert snapshot.indexes.area_ids_by_floor_id == {}


async def test_build_snapshot_filtered_collections_have_no_orphans(hass: HomeAssistant) -> None:
    floor_id, area_id, device_id = _add_area_device(hass, "Den")
    _add_entity(hass, "light.den", "den", device_id=device_id)
    override_area = ar.async_get(hass).async_create("Den Override")
    _add_entity(hass, "light.den_override", "den_override", device_id=device_id, area_id=override_area.id)

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert floor_id in snapshot.floors
    assert all(device.area_id in snapshot.areas for device in snapshot.devices.values() if device.area_id)
    assert all(entity.area_id in snapshot.areas for entity in snapshot.entities.values() if entity.area_id)
    assert all(area.floor_id in snapshot.floors for area in snapshot.areas.values() if area.floor_id)
    assert all(entity.device_id in snapshot.devices for entity in snapshot.entities.values() if entity.device_id)
    assert area_id in snapshot.areas
    assert override_area.id in snapshot.areas


async def test_build_snapshot_services_never_filtered(hass: HomeAssistant) -> None:
    _add_entity(
        hass,
        "light.excluded",
        "excluded",
        entity_category=er.EntityCategory.CONFIG,
    )
    hass.services.async_register("light", "turn_on", lambda call: None)
    hass.services.async_register("light", "turn_off", lambda call: None)
    await hass.async_block_till_done()

    snapshot = build_snapshot(hass, scope=DEFAULT_PRODUCT_SCOPE)

    assert "light.excluded" not in snapshot.states
    assert "turn_on" in snapshot.services["light"]
    assert "turn_off" in snapshot.services["light"]


async def test_build_snapshot_assist_restrict_delegates_to_ha_exposure(hass: HomeAssistant) -> None:
    _add_entity(hass, "light.exposed", "exposed")
    _add_entity(hass, "light.hidden_from_assist", "hidden_from_assist")
    exposed_entities = ExposedEntities(hass)
    hass.data[DATA_EXPOSED_ENTITIES] = exposed_entities
    await exposed_entities.async_initialize()
    async_expose_entity(hass, "conversation", "light.exposed", True)
    async_expose_entity(hass, "conversation", "light.hidden_from_assist", False)
    scope = SnapshotScope(
        assistant="conversation",
        restrict_to_assist_exposed=True,
        exclude_hidden=False,
        excluded_entity_categories=frozenset(),
        include_all_diagnostics=True,
    )

    snapshot = build_snapshot(hass, scope=scope)

    assert "light.exposed" in snapshot.states
    assert "light.hidden_from_assist" not in snapshot.states


async def test_build_snapshot_combined_restrictions_intersect(hass: HomeAssistant) -> None:
    _add_entity(hass, "light.visible", "visible")
    _add_entity(hass, "light.hidden", "hidden", hidden_by=er.RegistryEntryHider.USER)
    exposed_entities = ExposedEntities(hass)
    hass.data[DATA_EXPOSED_ENTITIES] = exposed_entities
    await exposed_entities.async_initialize()
    async_expose_entity(hass, "conversation", "light.visible", True)
    async_expose_entity(hass, "conversation", "light.hidden", True)
    scope = SnapshotScope(
        assistant="conversation",
        restrict_to_assist_exposed=True,
        exclude_hidden=True,
        excluded_entity_categories=frozenset(),
        include_all_diagnostics=True,
    )

    snapshot = build_snapshot(hass, scope=scope)

    assert "light.visible" in snapshot.states
    assert "light.hidden" not in snapshot.states


async def test_build_snapshot_all_restrictions_off_keeps_every_state_entity(hass: HomeAssistant) -> None:
    _add_entity(hass, "sensor.normal", "normal")
    _add_entity(hass, "sensor.config", "config", entity_category=er.EntityCategory.CONFIG)
    _add_entity(hass, "sensor.diagnostic", "diagnostic", entity_category=er.EntityCategory.DIAGNOSTIC)
    _add_entity(hass, "sensor.hidden", "hidden", hidden_by=er.RegistryEntryHider.USER)
    hass.states.async_set("input_boolean.state_only", "on")

    snapshot = build_snapshot(hass, scope=DEFAULT_SCOPE)

    assert {
        "sensor.normal",
        "sensor.config",
        "sensor.diagnostic",
        "sensor.hidden",
        "input_boolean.state_only",
    }.issubset(snapshot.states)
