"""Config flow for LLM Sandbox."""

from collections.abc import Iterable
from typing import Any, cast, final, override

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_ACTION_DOMAINS,
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_EXCLUDE_CONFIG,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_INCLUDE_ALL_DIAGNOSTICS,
    CONF_NAME,
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ASSISTANT,
    DEFAULT_NAME,
    DOMAIN,
    MAX_EXECUTION_TIMEOUT_SECONDS,
    MAX_HELPER_CALL_BUDGET,
    MIN_EXECUTION_TIMEOUT_SECONDS,
    MIN_HELPER_CALL_BUDGET,
    SECTION_ACTIONS,
    SECTION_EXECUTION_LIMITS,
    SECTION_PROMPT,
    SECTION_VISIBILITY,
)
from .llm_api.prompts import PROFILE_OPTIONS
from .runtime import normalize_action_domains, option_value
from .schema_helpers import flatten_section_data, section_schema_key

type UserInput = dict[str, Any]


class LlmSandboxOptionsFlow(OptionsFlow):
    """Handle LLM Sandbox options."""

    async def async_step_init(self, user_input: UserInput | None = None) -> ConfigFlowResult:
        """Manage LLM Sandbox entry options."""
        if user_input is not None:
            # HA sections namespace submitted values; options are stored flat.
            data = flatten_section_data(
                user_input,
                [SECTION_PROMPT, SECTION_VISIBILITY, SECTION_ACTIONS, SECTION_EXECUTION_LIMITS],
            )
            data[CONF_ACTION_DOMAINS] = normalize_action_domains(cast(Iterable[str], data[CONF_ACTION_DOMAINS]))
            return self.async_create_entry(data=data)

        options = self.config_entry.options
        live_domains = sorted({eid.split(".", 1)[0] for eid in er.async_get(self.hass).entities})
        live_domain_set = set(live_domains)
        selected_action_domains = list(cast(Iterable[str], option_value(options, CONF_ACTION_DOMAINS)))
        action_domain_options = [SelectOptionDict(value=d, label=d) for d in live_domains]
        for domain in reversed(selected_action_domains):
            # Preserve previously selected custom domains even when no entity currently exposes them.
            if domain not in live_domain_set:
                action_domain_options.insert(0, SelectOptionDict(value=domain, label=domain))

        prompt_fields: VolDictType = {
            vol.Required(
                CONF_PROMPT_PROFILE,
                default=option_value(options, CONF_PROMPT_PROFILE),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[SelectOptionDict(value=p.id, label=p.label) for p in PROFILE_OPTIONS],
                    multiple=False,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=False,
                )
            ),
        }

        visibility_fields: VolDictType = {
            vol.Required(
                CONF_RESTRICT_TO_ASSIST_EXPOSED,
                default=option_value(options, CONF_RESTRICT_TO_ASSIST_EXPOSED),
            ): BooleanSelector(),
            vol.Required(
                CONF_EXCLUDE_HIDDEN,
                default=option_value(options, CONF_EXCLUDE_HIDDEN),
            ): BooleanSelector(),
            vol.Required(
                CONF_EXCLUDE_CONFIG,
                default=option_value(options, CONF_EXCLUDE_CONFIG),
            ): BooleanSelector(),
            vol.Required(
                CONF_INCLUDE_ALL_DIAGNOSTICS,
                default=option_value(options, CONF_INCLUDE_ALL_DIAGNOSTICS),
            ): BooleanSelector(),
        }
        actions_fields: VolDictType = {
            vol.Required(
                CONF_ACTIONS_ENABLED,
                default=option_value(options, CONF_ACTIONS_ENABLED),
            ): BooleanSelector(),
            vol.Required(
                CONF_ACTION_DOMAINS,
                default=selected_action_domains,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=action_domain_options,
                    multiple=True,
                    mode=SelectSelectorMode.LIST,
                    custom_value=True,
                )
            ),
        }
        execution_limits_fields: VolDictType = {
            vol.Required(
                CONF_EXECUTION_TIMEOUT,
                default=option_value(options, CONF_EXECUTION_TIMEOUT),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=MIN_EXECUTION_TIMEOUT_SECONDS,
                    max=MAX_EXECUTION_TIMEOUT_SECONDS,
                    mode=NumberSelectorMode.BOX,
                    step=1,
                    unit_of_measurement="s",
                )
            ),
            vol.Required(
                CONF_HELPER_CALL_BUDGET,
                default=option_value(options, CONF_HELPER_CALL_BUDGET),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=MIN_HELPER_CALL_BUDGET,
                    max=MAX_HELPER_CALL_BUDGET,
                    mode=NumberSelectorMode.BOX,
                    step=1,
                )
            ),
        }
        schema = {
            section_schema_key(SECTION_PROMPT, prompt_fields): section(
                vol.Schema(prompt_fields),
                {"collapsed": True},
            ),
            section_schema_key(SECTION_VISIBILITY, visibility_fields): section(
                vol.Schema(visibility_fields),
                {"collapsed": True},
            ),
            section_schema_key(SECTION_ACTIONS, actions_fields): section(
                vol.Schema(actions_fields),
                {"collapsed": True},
            ),
            section_schema_key(SECTION_EXECUTION_LIMITS, execution_limits_fields): section(
                vol.Schema(execution_limits_fields),
                {"collapsed": True},
            ),
        }
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


@final
class LlmSandboxConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle an LLM Sandbox config flow."""

    VERSION: int = 1

    @staticmethod
    @callback
    def async_get_options_flow(_config_entry: ConfigEntry) -> LlmSandboxOptionsFlow:
        """Return the options flow for this config entry."""
        return LlmSandboxOptionsFlow()

    @override
    async def async_step_user(self, user_input: UserInput | None = None) -> ConfigFlowResult:
        """Create a config entry for one assistant exposure scope."""
        if user_input is not None:
            name = str(user_input[CONF_NAME])
            if not name.strip():
                return self.async_show_form(
                    step_id="user",
                    data_schema=_user_schema(),
                    errors={CONF_NAME: "name_required"},
                )
            assistant = user_input[CONF_ASSISTANT]
            _ = await self.async_set_unique_id(f"{DOMAIN}:{assistant}")
            _ = self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=name,
                data={CONF_ASSISTANT: assistant, CONF_NAME: name},
            )
        return self.async_show_form(
            step_id="user",
            data_schema=_user_schema(),
        )


def _user_schema() -> vol.Schema:
    """Return the initial config-flow schema."""
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
            vol.Required(CONF_ASSISTANT, default=DEFAULT_ASSISTANT): vol.In([DEFAULT_ASSISTANT]),
        }
    )
