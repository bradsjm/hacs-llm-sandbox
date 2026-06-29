"""Home Assistant LLM API for LLM Sandbox."""

import logging
from typing import cast, final, override

import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from ..const import DOMAIN, TOOL_EXECUTE_HOME_CODE
from ..runtime import SandboxConfigEntry, settings_from_entry
from ..snapshot import build_snapshot
from .errors import setup_error_payload, tool_error_from_exception
from .executor import MAX_MONTY_CODE_CHARS, async_execute_home_code
from .executor_support import ExecutionState
from .facade_views import build_llm_context
from .prompts import BASE_API_PROMPT, build_execute_home_code_description
from .runtime import RuntimeContext

_LOGGER = logging.getLogger(__name__)

type ToolArgs = dict[str, object]


def async_register_llm_api(hass: HomeAssistant, entry: SandboxConfigEntry) -> None:
    """Register the entry-scoped LLM Sandbox LLM API."""
    api = LlmSandboxAPI(hass, entry.entry_id, entry.title)
    entry.async_on_unload(llm.async_register_api(hass, api))


@final
class LlmSandboxAPI(llm.API):
    """Entry-scoped LLM Sandbox LLM API."""

    def __init__(self, hass: HomeAssistant, entry_id: str, entry_title: str) -> None:
        """Initialize one entry-scoped LLM Sandbox API."""
        super().__init__(
            hass=hass,
            id=f"{DOMAIN}_{entry_id}",
            name=f"{entry_title} LLM Sandbox",
        )
        self.entry_id = entry_id

    @override
    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt=BASE_API_PROMPT,
            llm_context=llm_context,
            tools=[ExecuteHomeCodeTool(self.entry_id)],
        )


@final
class ExecuteHomeCodeTool(llm.Tool):
    """Run bounded Home Assistant code in the Monty sandbox."""

    name = TOOL_EXECUTE_HOME_CODE
    description = build_execute_home_code_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required(
                "code",
                description="Python-like Monty code. Assign final JSON-safe data to result.",
            ): vol.All(str, vol.Length(min=1, max=MAX_MONTY_CODE_CHARS)),
        }
    )

    def __init__(self, entry_id: str) -> None:
        """Initialize the code execution tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        try:
            data = cast(ToolArgs, self.parameters(tool_input.tool_args))
            return await _execute(hass, self.entry_id, data, llm_context)
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            key, placeholders = mapped
            return cast(JsonObjectType, setup_error_payload(key, placeholders))


async def _execute(
    hass: HomeAssistant,
    entry_id: str,
    data: ToolArgs,
    llm_context: llm.LLMContext,
) -> JsonObjectType:
    """Build a snapshot, construct facades, and run the executor."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        return cast(JsonObjectType, setup_error_payload("unknown_config_entry", {"config_entry_id": entry_id}))
    typed_entry = cast(SandboxConfigEntry, entry)
    if typed_entry.state is not ConfigEntryState.LOADED or typed_entry.runtime_data is None:
        return cast(
            JsonObjectType,
            setup_error_payload("config_entry_not_loaded", {"config_entry_id": entry_id}),
        )

    settings = settings_from_entry(typed_entry)
    code = cast(str, data["code"])

    # Build a fresh snapshot on the event loop before execution.
    snapshot = build_snapshot(hass)

    # Build the LLM context view from the live request metadata.
    context = llm_context.context if llm_context.context is not None else Context()
    safe_context = build_llm_context(
        platform=llm_context.platform,
        context_id=context.id,
        parent_id=context.parent_id,
        user_id=context.user_id,
        language=llm_context.language,
        assistant=llm_context.assistant,
        device_id=llm_context.device_id,
    )

    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=settings.helper_call_budget),
        settings=settings,
    )

    result = await async_execute_home_code(
        code,
        snapshot=snapshot,
        llm_context=safe_context,
        runtime=runtime,
    )
    if isinstance(result, dict):
        execution = result.get("execution", {})
        _LOGGER.debug(
            "execute_home_code: status=%s helper_calls=%s/%s proposed_actions=%d",
            execution.get("status") if isinstance(execution, dict) else "n/a",
            runtime.state.helper_calls,
            runtime.state.helper_call_limit,
            len(runtime.state.proposed_actions),
        )
    return cast(JsonObjectType, result)
