"""Tests for LLM Sandbox config and options flows."""

from collections.abc import Mapping

from custom_components.llm_sandbox.const import (
    CONF_ACTION_DOMAINS,
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_EXCLUDE_CONFIG,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_INCLUDE_ALL_DIAGNOSTICS,
    CONF_NAME,
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    CONF_SERVICE_CALL_LIMIT,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_ASSISTANT,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_SERVICE_CALL_LIMIT,
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
from homeassistant.data_entry_flow import InvalidData
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol


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


async def test_blank_name_submit_shows_form_error(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": "user"},
        data={CONF_NAME: "   ", CONF_ASSISTANT: DEFAULT_ASSISTANT},
    )

    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert result["errors"] == {CONF_NAME: "name_required"}


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
        service_call_limit=DEFAULT_SERVICE_CALL_LIMIT,
        scope=SnapshotScope(
            assistant=DEFAULT_ASSISTANT,
            restrict_to_assist_exposed=DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
            exclude_hidden=DEFAULT_EXCLUDE_HIDDEN,
            excluded_entity_categories=frozenset({"config"}),
            include_all_diagnostics=False,
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
    ("options", "expected_assist", "expected_hidden", "expected_categories", "expected_all_diagnostics"),
    [
        pytest.param({}, True, True, frozenset({"config"}), False, id="defaults"),
        pytest.param(
            {CONF_RESTRICT_TO_ASSIST_EXPOSED: False}, False, True, frozenset({"config"}), False, id="assist-off"
        ),
        pytest.param(
            {CONF_EXCLUDE_CONFIG: False},
            True,
            True,
            frozenset(),
            False,
            id="config-off",
        ),
        pytest.param(
            {CONF_INCLUDE_ALL_DIAGNOSTICS: True},
            True,
            True,
            frozenset({"config"}),
            True,
            id="include-all-diagnostics",
        ),
        pytest.param(
            {CONF_EXCLUDE_HIDDEN: False},
            True,
            False,
            frozenset({"config"}),
            False,
            id="hidden-off",
        ),
    ],
)
def test_settings_from_entry_visibility_options(
    options: dict[str, object],
    expected_assist: bool,
    expected_hidden: bool,
    expected_categories: frozenset[str],
    expected_all_diagnostics: bool,
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
    assert settings.scope.include_all_diagnostics is expected_all_diagnostics


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
        pytest.param(
            {CONF_ACTIONS_ENABLED: True, CONF_ACTION_DOMAINS: [" Light ", "light", "CUSTOM.Domain", "  "]},
            True,
            frozenset({"light", "custom.domain"}),
            id="normalized-domains",
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
            CONF_EXECUTION_TIMEOUT: 3,
            CONF_SERVICE_CALL_LIMIT: 48,
        },
        SECTION_VISIBILITY: {
            CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
            CONF_EXCLUDE_HIDDEN: False,
            CONF_EXCLUDE_CONFIG: False,
            CONF_INCLUDE_ALL_DIAGNOSTICS: True,
        },
        SECTION_ACTIONS: {
            CONF_ACTIONS_ENABLED: True,
            CONF_ACTION_DOMAINS: [" Light ", "light", "CUSTOM.Domain", "  "],
        },
    }

    result = await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)

    assert result["type"] == "create_entry"
    assert mock_config_entry.options == {
        CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE,
        CONF_EXECUTION_TIMEOUT: 3.0,
        CONF_SERVICE_CALL_LIMIT: 48.0,
        CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
        CONF_EXCLUDE_HIDDEN: False,
        CONF_EXCLUDE_CONFIG: False,
        CONF_INCLUDE_ALL_DIAGNOSTICS: True,
        CONF_ACTIONS_ENABLED: True,
        CONF_ACTION_DOMAINS: ["light", "custom.domain"],
    }


async def test_options_flow_rejects_invalid_action_domain(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Options flow validates custom action-domain syntax before storing options."""
    mock_config_entry.add_to_hass(hass)
    init_result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    options = {
        SECTION_PROMPT: {CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE},
        SECTION_EXECUTION_LIMITS: {
            CONF_EXECUTION_TIMEOUT: 17,
            CONF_SERVICE_CALL_LIMIT: 48,
        },
        SECTION_VISIBILITY: {
            CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
            CONF_EXCLUDE_HIDDEN: False,
            CONF_EXCLUDE_CONFIG: False,
            CONF_INCLUDE_ALL_DIAGNOSTICS: True,
        },
        SECTION_ACTIONS: {
            CONF_ACTIONS_ENABLED: True,
            CONF_ACTION_DOMAINS: ["light", "bad domain"],
        },
    }

    result = await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)

    assert result["type"] == "form"
    assert result["step_id"] == "init"
    assert result["errors"] == {CONF_ACTION_DOMAINS: "invalid_action_domain"}


async def test_options_flow_rejects_timeout_below_three_seconds(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """The execution-timeout boundary rejects values below three seconds."""
    mock_config_entry.add_to_hass(hass)
    init_result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    options = {
        SECTION_PROMPT: {CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE},
        SECTION_EXECUTION_LIMITS: {
            CONF_EXECUTION_TIMEOUT: 2,
            CONF_SERVICE_CALL_LIMIT: 48,
        },
        SECTION_VISIBILITY: {
            CONF_RESTRICT_TO_ASSIST_EXPOSED: False,
            CONF_EXCLUDE_HIDDEN: False,
            CONF_EXCLUDE_CONFIG: False,
            CONF_INCLUDE_ALL_DIAGNOSTICS: True,
        },
        SECTION_ACTIONS: {
            CONF_ACTIONS_ENABLED: True,
            CONF_ACTION_DOMAINS: ["light"],
        },
    }

    with pytest.raises(InvalidData):
        await hass.config_entries.options.async_configure(init_result["flow_id"], user_input=options)
