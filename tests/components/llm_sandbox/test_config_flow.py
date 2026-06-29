"""Tests for LLM Sandbox config and options flows."""

from collections.abc import Mapping

import voluptuous as vol
from custom_components.llm_sandbox.const import (
    CONF_ASSISTANT,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_NAME,
    DEFAULT_ASSISTANT,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DOMAIN,
)
from custom_components.llm_sandbox.runtime import SandboxSettings, settings_from_entry
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry


def _schema_default(result: ConfigFlowResult, field: str) -> object:
    data_schema = result["data_schema"]
    assert isinstance(data_schema, vol.Schema)
    schema = data_schema.schema
    assert isinstance(schema, Mapping)
    for marker in schema:
        if isinstance(marker, vol.Marker) and marker.schema == field:
            default = marker.default  # type: ignore[attr-defined]
            assert isinstance(default, object)
            return default() if callable(default) else default
    raise AssertionError(field)


async def test_user_step_shows_form(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})

    assert result["type"] == "form"
    assert result["step_id"] == "user"


async def test_user_step_defaults_assistant_to_conversation(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})

    assert _schema_default(result, CONF_ASSISTANT) == DEFAULT_ASSISTANT


async def test_valid_submit_creates_entry(hass: HomeAssistant) -> None:
    name = "LLM Sandbox Test"
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data={CONF_NAME: name, CONF_ASSISTANT: DEFAULT_ASSISTANT},
    )

    assert result["type"] == "create_entry"
    assert result["data"] == {CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: name}


async def test_duplicate_assistant_aborts(hass: HomeAssistant) -> None:
    data = {CONF_NAME: "LLM Sandbox Test", CONF_ASSISTANT: DEFAULT_ASSISTANT}
    first = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"}, data=data)
    second = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"}, data=data)

    assert first["type"] == "create_entry"
    assert second["type"] == "abort"
    assert second["reason"] == "already_configured"


async def test_options_flow_persists_typed_options(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    init_result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    options = {
        CONF_EXECUTION_TIMEOUT: 17,
        CONF_HELPER_CALL_BUDGET: 48,
    }

    result = await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)

    assert result["type"] == "create_entry"
    assert mock_config_entry.options == options


def test_settings_from_entry_defaults() -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: "LLM Sandbox"},
        options={},
    )

    settings = settings_from_entry(entry)

    assert settings == SandboxSettings(
        execution_timeout_seconds=DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        helper_call_budget=DEFAULT_HELPER_CALL_BUDGET,
    )
