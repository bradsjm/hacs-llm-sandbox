"""Shared fixtures for LLM Sandbox component tests."""

import pytest
from custom_components.llm_sandbox.const import (
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_NAME,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ASSISTANT,
    DEFAULT_NAME,
    DOMAIN,
)
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry


@pytest.fixture(autouse=True)
def _auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integration discovery for every test in this package."""


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return a default MockConfigEntry for llm_sandbox."""
    return MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: DEFAULT_NAME},
        options={CONF_ACTIONS_ENABLED: True, CONF_RESTRICT_TO_ASSIST_EXPOSED: False},
        unique_id=f"{DOMAIN}:{DEFAULT_ASSISTANT}",
    )


@pytest.fixture
async def loaded_entry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> MockConfigEntry:
    """Set up a loaded config entry with a minimal registry/state fixture.

    Registers ``light.bedroom`` and ``light.living_room`` in the entity
    registry, a Bedroom area, and sets live states so tests can exercise the
    snapshot-backed read facades and propose-only action path.
    """
    from homeassistant.helpers import area_registry as ar
    from homeassistant.helpers import entity_registry as er

    entity_registry = er.async_get(hass)
    area_registry = ar.async_get(hass)

    bedroom_area = area_registry.async_create("Bedroom")

    entity_registry.async_get_or_create(
        "light",
        "test",
        "bedroom",
        suggested_object_id="bedroom",
    )
    # Assign the bedroom light to the Bedroom area via the registry.
    entity_registry.async_update_entity("light.bedroom", area_id=bedroom_area.id)
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})

    entity_registry.async_get_or_create(
        "light",
        "test",
        "living_room",
        suggested_object_id="living_room",
    )
    hass.states.async_set("light.living_room", "off", {"friendly_name": "Living Room Light"})

    # Register a couple of services so the catalog snapshot is non-empty.
    hass.services.async_register("light", "turn_on", lambda call: None)
    hass.services.async_register("light", "turn_off", lambda call: None)

    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
