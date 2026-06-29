"""Tests for the snapshot builder."""

from custom_components.llm_sandbox.snapshot import build_snapshot
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _add_device_owner_entry(hass: HomeAssistant) -> str:
    """Add a mock config entry that can own device-registry records."""
    entry = MockConfigEntry(domain="test", title="Device Owner")
    entry.add_to_hass(hass)
    return str(entry.entry_id)


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


async def test_snapshot_datetimes_are_iso_strings(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.temp", "23.5", {"friendly_name": "Temp"})

    snapshot = build_snapshot(hass)

    state = snapshot.states["sensor.temp"]
    assert isinstance(state.last_changed, str)
    assert "T" in state.last_changed
    assert isinstance(state.last_updated, str)


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
    # Entity on the same device, no area override -> inherits Living Room.
    entity_registry.async_get_or_create(
        "light",
        "test",
        "inherit_light",
        suggested_object_id="inherit_light",
        device_id=device.id,
    )

    snapshot = build_snapshot(hass)

    bedroom_entities = snapshot.indexes.entity_ids_by_area_id.get(bedroom.id, ())
    living_room_entities = snapshot.indexes.entity_ids_by_area_id.get(living_room.id, ())

    assert "light.override_light" in bedroom_entities
    assert "light.override_light" not in living_room_entities
    assert "light.inherit_light" in living_room_entities


async def test_snapshot_service_catalog(hass: HomeAssistant) -> None:
    hass.services.async_register("light", "turn_on", lambda call: None)
    hass.services.async_register("light", "turn_off", lambda call: None)
    await hass.async_block_till_done()

    snapshot = build_snapshot(hass)

    assert "turn_on" in snapshot.services.get("light", ())
    assert "turn_off" in snapshot.services.get("light", ())


async def test_snapshot_device_identifiers_and_connections_are_json_safe(hass: HomeAssistant) -> None:
    device_registry = dr.async_get(hass)
    device_registry.async_get_or_create(
        config_entry_id=_add_device_owner_entry(hass),
        connections={("mac", "aa:bb:cc:dd:ee:ff")},
        identifiers={("test", "device1")},
    )

    snapshot = build_snapshot(hass)

    # Find the device by its identifier.
    device = next(d for d in snapshot.devices.values() if ("test", "device1") in d.identifiers)
    assert ("test", "device1") in device.identifiers
    assert ("mac", "aa:bb:cc:dd:ee:ff") in device.connections
    # Tuples are JSON-safe (frozen as tuples).
    assert all(isinstance(ident, tuple) for ident in device.identifiers)
