"""Tests for LLM Sandbox config and options flows."""

from collections.abc import Mapping

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.const import (
    CONF_ACTION_DOMAINS,
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_EXCLUDE_CONFIG,
    CONF_EXCLUDE_DIAGNOSTIC,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_NAME,
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_ASSISTANT,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
    DOMAIN,
    SECTION_ACTIONS,
    SECTION_EXECUTION_LIMITS,
    SECTION_PROMPT,
    SECTION_VISIBILITY,
)
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from custom_components.llm_sandbox.runtime import SandboxSettings, settings_from_entry
from custom_components.llm_sandbox.snapshot import SnapshotScope
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
            assistant=DEFAULT_ASSISTANT,
            restrict_to_assist_exposed=DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
            exclude_hidden=DEFAULT_EXCLUDE_HIDDEN,
            excluded_entity_categories=frozenset({"config", "diagnostic"}),
        ),
        actions_enabled=DEFAULT_ACTIONS_ENABLED,
        action_domains=frozenset(),
        prompt_profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )


def test_settings_from_entry_rejects_unknown_prompt_profile() -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: "LLM Sandbox"},
        options={CONF_PROMPT_PROFILE: "nonexistent"},
    )

    with pytest.raises(ValueError, match="unknown prompt profile"):
        settings_from_entry(entry)


@pytest.mark.parametrize(
    ("options", "expected_assist", "expected_hidden", "expected_categories"),
    [
        pytest.param({}, True, True, frozenset({"config", "diagnostic"}), id="defaults"),
        pytest.param(
            {CONF_RESTRICT_TO_ASSIST_EXPOSED: False}, False, True, frozenset({"config", "diagnostic"}), id="assist-off"
        ),
        pytest.param(
            {CONF_EXCLUDE_CONFIG: False},
            True,
            True,
            frozenset({"diagnostic"}),
            id="config-off",
        ),
        pytest.param(
            {CONF_EXCLUDE_DIAGNOSTIC: False},
            True,
            True,
            frozenset({"config"}),
            id="diagnostic-off",
        ),
        pytest.param(
            {CONF_EXCLUDE_HIDDEN: False},
            True,
            False,
            frozenset({"config", "diagnostic"}),
            id="hidden-off",
        ),
    ],
)
def test_settings_from_entry_visibility_options(
    options: dict[str, object],
    expected_assist: bool,
    expected_hidden: bool,
    expected_categories: frozenset[str],
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: "LLM Sandbox"},
        options=options,
    )

    settings = settings_from_entry(entry)

    assert settings.scope.assistant == DEFAULT_ASSISTANT
    assert settings.scope.restrict_to_assist_exposed is expected_assist
    assert settings.scope.exclude_hidden is expected_hidden
    assert settings.scope.excluded_entity_categories == expected_categories


@pytest.mark.parametrize(
    ("options", "expected_enabled", "expected_domains"),
    [
        pytest.param({}, False, frozenset(), id="defaults"),
        pytest.param(
            {CONF_ACTIONS_ENABLED: True, CONF_ACTION_DOMAINS: ["light", "switch"]},
            True,
            frozenset({"light", "switch"}),
            id="configured",
        ),
        pytest.param(
            {CONF_ACTIONS_ENABLED: True, CONF_ACTION_DOMAINS: []},
            True,
            frozenset(),
            id="empty-domains",
        ),
    ],
)
def test_settings_from_entry_action_options(
    options: dict[str, object],
    expected_enabled: bool,
    expected_domains: frozenset[str],
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ASSISTANT: DEFAULT_ASSISTANT, CONF_NAME: "LLM Sandbox"},
        options=options,
    )

    settings = settings_from_entry(entry)

    assert settings.actions_enabled is expected_enabled
    assert settings.action_domains == expected_domains


async def test_options_flow_section_submission_flattens(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    mock_config_entry.add_to_hass(hass)
    init_result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    options = {
        SECTION_PROMPT: {CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE},
        SECTION_EXECUTION_LIMITS: {
            CONF_EXECUTION_TIMEOUT: 17,
            CONF_HELPER_CALL_BUDGET: 48,
        },
        SECTION_VISIBILITY: {
            CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
            CONF_EXCLUDE_HIDDEN: False,
            CONF_EXCLUDE_CONFIG: False,
            CONF_EXCLUDE_DIAGNOSTIC: True,
        },
        SECTION_ACTIONS: {
            CONF_ACTIONS_ENABLED: True,
            CONF_ACTION_DOMAINS: ["light"],
        },
    }

    result = await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)

    assert result["type"] == "create_entry"
    assert mock_config_entry.options == {
        CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE,
        CONF_EXECUTION_TIMEOUT: 17.0,
        CONF_HELPER_CALL_BUDGET: 48.0,
        CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
        CONF_EXCLUDE_HIDDEN: False,
        CONF_EXCLUDE_CONFIG: False,
        CONF_EXCLUDE_DIAGNOSTIC: True,
        CONF_ACTIONS_ENABLED: True,
        CONF_ACTION_DOMAINS: ["light"],
    }
