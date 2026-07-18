"""Tests for LLM Sandbox setup and unload lifecycle."""

from unittest.mock import AsyncMock, patch

from custom_components.llm_sandbox.const import (
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_NAME,
    DEFAULT_ASSISTANT,
    DEFAULT_NAME,
    DOMAIN,
    TOOL_GET_AUTOMATION,
    TOOL_GET_ENERGY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.data.energy import (
    SafeEnergyCatalog,
    SafeEnergyMeasureRef,
    SafeEnergySourceRecord,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from voluptuous_openapi import convert


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


async def test_api_instance_exposes_automation_schema(hass: HomeAssistant) -> None:
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

    assert api_instance.custom_serializer is llm.selector_serializer
    automation_tool = next(tool for tool in api_instance.tools if tool.name == TOOL_GET_AUTOMATION)
    automation_schema = convert(
        automation_tool.parameters,
        custom_serializer=api_instance.custom_serializer,
    )
    automation_properties = automation_schema["properties"]
    assert automation_properties["query"]["type"] == "string"
    assert automation_properties["entity_ids"]["type"] == "array"
    assert automation_properties["entity_ids"]["items"]["type"] == "string"
    assert automation_properties["include"]["type"] == "array"
    assert automation_properties["include"]["items"]["enum"] == ["content", "runs"]
    assert automation_properties["start"] == {"format": "date-time", "type": "string"}
    assert automation_properties["end"] == {"format": "date-time", "type": "string"}


@pytest.mark.parametrize(
    ("catalog_result", "recorder_ok", "energy_registered"),
    [
        pytest.param(
            (
                SafeEnergyCatalog(
                    sources=(
                        SafeEnergySourceRecord(
                            "grid:0",
                            "grid",
                            "Main grid",
                            (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
                        ),
                    ),
                    devices=(),
                    omissions=(),
                    co2_statistic_id=None,
                    configured_record_count=1,
                ),
                {},
            ),
            True,
            True,
            id="visible-source",
        ),
        pytest.param(
            (SafeEnergyCatalog((), (), (), None, 1), {}),
            True,
            True,
            id="configured-all-sources-hidden",
        ),
        pytest.param(
            (SafeEnergyCatalog((), (), (), None, 1), {}),
            False,
            False,
            id="recorder-unavailable",
        ),
        pytest.param(
            None,
            True,
            False,
            id="energy-unconfigured",
        ),
    ],
)
async def test_api_registers_energy_when_dashboard_is_configured(
    hass: HomeAssistant,
    catalog_result: tuple[SafeEnergyCatalog, dict[str, tuple[str, ...]]] | None,
    recorder_ok: bool,
    energy_registered: bool,
) -> None:
    """Configured Energy dashboards register the direct tool only with Recorder."""
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
    [api] = [registered_api for registered_api in llm.async_get_apis(hass) if registered_api.id.startswith(DOMAIN)]
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )

    with (
        patch(
            "custom_components.llm_sandbox.llm_api.api.async_copy_energy_catalog",
            new=AsyncMock(return_value=catalog_result),
        ),
        patch("custom_components.llm_sandbox.llm_api.api.recorder_available", return_value=recorder_ok),
    ):
        api_instance = await api.async_get_api_instance(llm_context)

    assert (TOOL_GET_ENERGY in {tool.name for tool in api_instance.tools}) is energy_registered


async def test_energy_tool_ordered_after_statistics_and_before_logbook(
    hass: HomeAssistant,
) -> None:
    """Energy sits between statistics and logbook in the ordered tool list."""
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
    [api] = [registered_api for registered_api in llm.async_get_apis(hass) if registered_api.id.startswith(DOMAIN)]
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    visible_catalog = (
        SafeEnergyCatalog(
            sources=(
                SafeEnergySourceRecord(
                    "grid:0",
                    "grid",
                    "Main grid",
                    (SafeEnergyMeasureRef("grid_import", "sensor.grid_import"),),
                ),
            ),
            devices=(),
            omissions=(),
            co2_statistic_id=None,
            configured_record_count=1,
        ),
        {},
    )

    with (
        patch(
            "custom_components.llm_sandbox.llm_api.api.async_copy_energy_catalog",
            new=AsyncMock(return_value=visible_catalog),
        ),
        patch("custom_components.llm_sandbox.llm_api.api.recorder_available", return_value=True),
        patch("custom_components.llm_sandbox.llm_api.api.logbook_available", return_value=True),
    ):
        api_instance = await api.async_get_api_instance(llm_context)

    tool_names = [tool.name for tool in api_instance.tools]
    assert tool_names.index(TOOL_GET_ENERGY) == tool_names.index(TOOL_GET_STATISTICS) + 1
    assert tool_names.index(TOOL_GET_LOGBOOK) == tool_names.index(TOOL_GET_ENERGY) + 1


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


async def test_options_update_listener_applies_reloaded_settings(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """An options update reloads the entry and applies the new runtime settings."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.runtime_data.settings.actions_enabled is True

    hass.config_entries.async_update_entry(
        mock_config_entry,
        options=mock_config_entry.options | {CONF_ACTIONS_ENABLED: False},
    )
    await hass.async_block_till_done()

    assert mock_config_entry.runtime_data.settings.actions_enabled is False
