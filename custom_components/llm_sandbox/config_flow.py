"""Config flow for LLM Sandbox."""

from typing import Any, final, override

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
    CONF_EXCLUDE_DIAGNOSTIC,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_NAME,
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ACTION_DOMAINS,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_ASSISTANT,
    DEFAULT_EXCLUDE_CONFIG,
    DEFAULT_EXCLUDE_DIAGNOSTIC,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_NAME,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
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
            return self.async_create_entry(data=data)

        options = self.config_entry.options
        live_domains = sorted({eid.split(".", 1)[0] for eid in er.async_get(self.hass).entities})
        live_domain_set = set(live_domains)
        selected_action_domains = list(options.get(CONF_ACTION_DOMAINS, DEFAULT_ACTION_DOMAINS))
        action_domain_options = [SelectOptionDict(value=d, label=d) for d in live_domains]
        for domain in reversed(selected_action_domains):
            # Preserve previously selected custom domains even when no entity currently exposes them.
            if domain not in live_domain_set:
                action_domain_options.insert(0, SelectOptionDict(value=domain, label=domain))

        prompt_fields: VolDictType = {
            vol.Required(
                CONF_PROMPT_PROFILE,
                default=options.get(CONF_PROMPT_PROFILE, DEFAULT_PROMPT_PROFILE),
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
                default=options.get(CONF_RESTRICT_TO_ASSIST_EXPOSED, DEFAULT_RESTRICT_TO_ASSIST_EXPOSED),
            ): BooleanSelector(),
            vol.Required(
                CONF_EXCLUDE_HIDDEN,
                default=options.get(CONF_EXCLUDE_HIDDEN, DEFAULT_EXCLUDE_HIDDEN),
            ): BooleanSelector(),
            vol.Required(
                CONF_EXCLUDE_CONFIG,
                default=options.get(CONF_EXCLUDE_CONFIG, DEFAULT_EXCLUDE_CONFIG),
            ): BooleanSelector(),
            vol.Required(
                CONF_EXCLUDE_DIAGNOSTIC,
                default=options.get(CONF_EXCLUDE_DIAGNOSTIC, DEFAULT_EXCLUDE_DIAGNOSTIC),
            ): BooleanSelector(),
        }
        actions_fields: VolDictType = {
            vol.Required(
                CONF_ACTIONS_ENABLED,
                default=options.get(CONF_ACTIONS_ENABLED, DEFAULT_ACTIONS_ENABLED),
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
                default=options.get(CONF_EXECUTION_TIMEOUT, DEFAULT_EXECUTION_TIMEOUT_SECONDS),
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
                default=options.get(CONF_HELPER_CALL_BUDGET, DEFAULT_HELPER_CALL_BUDGET),
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
            assistant = user_input[CONF_ASSISTANT]
            _ = await self.async_set_unique_id(f"{DOMAIN}:{assistant}")
            _ = self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=user_input[CONF_NAME],
                data={CONF_ASSISTANT: assistant, CONF_NAME: user_input[CONF_NAME]},
            )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
                    vol.Required(CONF_ASSISTANT, default=DEFAULT_ASSISTANT): vol.In([DEFAULT_ASSISTANT]),
                }
            ),
        )
