"""Tests for LLM Sandbox setup and unload lifecycle."""

from custom_components.llm_sandbox.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_setup_assigns_runtime_data(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.runtime_data is not None
    assert mock_config_entry.runtime_data.settings is not None


async def test_setup_registers_llm_api(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    apis = llm.async_get_apis(hass)
    sandbox_apis = [api for api in apis if api.id.startswith(DOMAIN)]
    assert len(sandbox_apis) == 1


async def test_unload_cleans_up(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # After unload the API should be unregistered.
    apis = llm.async_get_apis(hass)
    sandbox_apis = [api for api in apis if api.id.startswith(DOMAIN)]
    assert len(sandbox_apis) == 0
