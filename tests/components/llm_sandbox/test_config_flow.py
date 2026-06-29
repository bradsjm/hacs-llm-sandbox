"""Tests for LLM Sandbox config and options flows."""

from collections.abc import Mapping

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.const import (
    CONF_ASSISTANT,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXCLUDED_ENTITY_CATEGORIES,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_NAME,
    CONF_SCOPE_MODE,
    DEFAULT_ASSISTANT,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXCLUDED_ENTITY_CATEGORIES,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_SCOPE_MODE,
    DOMAIN,
)
from custom_components.llm_sandbox.runtime import SandboxSettings, settings_from_entry
from custom_components.llm_sandbox.snapshot import ScopeMode, SnapshotScope
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
    assert mock_config_entry.options[CONF_EXECUTION_TIMEOUT] == 17.0
    assert mock_config_entry.options[CONF_HELPER_CALL_BUDGET] == 48.0
    assert mock_config_entry.options[CONF_SCOPE_MODE] == DEFAULT_SCOPE_MODE
    assert mock_config_entry.options[CONF_EXCLUDED_ENTITY_CATEGORIES] == list(DEFAULT_EXCLUDED_ENTITY_CATEGORIES)
    assert mock_config_entry.options[CONF_EXCLUDE_HIDDEN] is DEFAULT_EXCLUDE_HIDDEN


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
        scope=SnapshotScope(
            mode=ScopeMode.CHARACTERISTICS,
            assistant=DEFAULT_ASSISTANT,
            excluded_entity_categories=frozenset(DEFAULT_EXCLUDED_ENTITY_CATEGORIES),
            exclude_hidden=DEFAULT_EXCLUDE_HIDDEN,
        ),
    )


@pytest.mark.parametrize(
    ("options", "expected_mode", "expected_categories", "expected_exclude_hidden"),
    [
        pytest.param(
            {}, ScopeMode.CHARACTERISTICS, frozenset(DEFAULT_EXCLUDED_ENTITY_CATEGORIES), True, id="defaults"
        ),
        pytest.param(
            {CONF_SCOPE_MODE: "all"}, ScopeMode.ALL, frozenset(DEFAULT_EXCLUDED_ENTITY_CATEGORIES), True, id="mode"
        ),
        pytest.param(
            {CONF_SCOPE_MODE: "characteristics", CONF_EXCLUDED_ENTITY_CATEGORIES: ["diagnostic"]},
            ScopeMode.CHARACTERISTICS,
            frozenset({"diagnostic"}),
            True,
            id="categories",
        ),
        pytest.param(
            {CONF_SCOPE_MODE: "characteristics", CONF_EXCLUDE_HIDDEN: False},
            ScopeMode.CHARACTERISTICS,
            frozenset(DEFAULT_EXCLUDED_ENTITY_CATEGORIES),
            False,
            id="hidden",
        ),
        pytest.param(
            {
                CONF_SCOPE_MODE: "assist_expose",
                CONF_EXCLUDED_ENTITY_CATEGORIES: ["config"],
                CONF_EXCLUDE_HIDDEN: False,
            },
            ScopeMode.ASSIST_EXPOSE,
            frozenset({"config"}),
            False,
            id="full",
        ),
    ],
)
def test_settings_from_entry_scope_options(
    options: dict[str, object],
    expected_mode: ScopeMode,
    expected_categories: frozenset[str],
    expected_exclude_hidden: bool,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: "LLM Sandbox"},
        options=options,
    )

    settings = settings_from_entry(entry)

    assert settings.scope.mode is expected_mode
    assert settings.scope.assistant == DEFAULT_ASSISTANT
    assert settings.scope.excluded_entity_categories == expected_categories
    assert settings.scope.exclude_hidden is expected_exclude_hidden


async def test_options_flow_persists_scope_options(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    init_result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    options = {
        CONF_EXECUTION_TIMEOUT: 17,
        CONF_HELPER_CALL_BUDGET: 48,
        CONF_SCOPE_MODE: "assist_expose",
        CONF_EXCLUDED_ENTITY_CATEGORIES: ["config"],
        CONF_EXCLUDE_HIDDEN: False,
    }

    result = await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)

    assert result["type"] == "create_entry"
    assert mock_config_entry.options == options


async def test_options_flow_scope_defaults(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    assert _schema_default(result, CONF_SCOPE_MODE) == DEFAULT_SCOPE_MODE
    assert _schema_default(result, CONF_EXCLUDE_HIDDEN) is DEFAULT_EXCLUDE_HIDDEN
    assert _schema_default(result, CONF_EXCLUDED_ENTITY_CATEGORIES) == list(DEFAULT_EXCLUDED_ENTITY_CATEGORIES)
