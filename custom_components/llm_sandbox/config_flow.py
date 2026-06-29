"""Config flow for LLM Sandbox."""

from typing import Any, final, override

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers.selector import NumberSelector, NumberSelectorConfig, NumberSelectorMode

from .const import (
    CONF_ASSISTANT,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_NAME,
    DEFAULT_ASSISTANT,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_NAME,
    DOMAIN,
    MAX_EXECUTION_TIMEOUT_SECONDS,
    MAX_HELPER_CALL_BUDGET,
    MIN_EXECUTION_TIMEOUT_SECONDS,
    MIN_HELPER_CALL_BUDGET,
)

type UserInput = dict[str, Any]


class LlmSandboxOptionsFlow(OptionsFlow):
    """Handle LLM Sandbox options."""

    async def async_step_init(self, user_input: UserInput | None = None) -> ConfigFlowResult:
        """Manage LLM Sandbox entry options."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
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
            ),
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
