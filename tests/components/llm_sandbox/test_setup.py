"""Tests for LLM Sandbox setup and unload lifecycle."""

from custom_components.llm_sandbox.const import (
    CONF_ASSISTANT,
    CONF_NAME,
    DEFAULT_ASSISTANT,
    DEFAULT_NAME,
    DEFAULT_PROMPT_PROFILE,
    DOMAIN,
)
from custom_components.llm_sandbox.llm_api.prompts import (
    ACTIONS_DISABLED_PROMPT,
    ACTIONS_ENABLED_PROMPT,
    resolve_profile,
)
from homeassistant.core import Context, HomeAssistant
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


async def test_api_prompt_uses_default_profile_and_one_action_section(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: DEFAULT_NAME},
        options={},
        unique_id=f"{DOMAIN}:{DEFAULT_ASSISTANT}",
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    apis = llm.async_get_apis(hass)
    [api] = [registered_api for registered_api in apis if registered_api.id.startswith(DOMAIN)]
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    api_instance = await api.async_get_api_instance(llm_context)

    assert api_instance.api_prompt.startswith(resolve_profile(DEFAULT_PROMPT_PROFILE).base_prompt)
    assert api_instance.api_prompt.startswith("# LLM Sandbox tools")
    assert api_instance.api_prompt.count(ACTIONS_DISABLED_PROMPT) == 1
    assert api_instance.api_prompt.count(ACTIONS_ENABLED_PROMPT) == 0


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
