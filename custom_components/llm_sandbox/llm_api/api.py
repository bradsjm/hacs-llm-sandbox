"""Home Assistant LLM API for LLM Sandbox."""

import logging
import time
from typing import Any, cast, final, override

import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from ..const import DOMAIN, TOOL_EXECUTE_HOME_CODE
from ..runtime import SandboxConfigEntry, settings_from_entry
from ..snapshot import build_snapshot
from ..snapshot.models import HomeSnapshot
from ..types import ProposedAction
from .errors import setup_error_payload, tool_error_from_exception
from .executor import MAX_MONTY_CODE_CHARS, async_execute_home_code
from .executor_support import ExecutionState
from .facade_views import build_llm_context
from .prompts import ACTIONS_DISABLED_PROMPT, BASE_API_PROMPT, build_execute_home_code_description
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
            name=entry_title,
        )
        self._hass = hass
        self.entry_id = entry_id

    @override
    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        entry = self._hass.config_entries.async_get_entry(self.entry_id)
        actions_enabled = False
        # Missing or wrong-domain entries get a conservative prompt.
        if entry is not None and entry.domain == DOMAIN:
            actions_enabled = settings_from_entry(cast(SandboxConfigEntry, entry)).actions_enabled
        return llm.APIInstance(
            api=self,
            api_prompt=_build_api_prompt(self._hass, llm_context, actions_enabled),
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
    """Build a Monty view, construct facades, and run the executor."""
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
    deadline = time.monotonic() + settings.execution_timeout_seconds

    # Build a fresh Monty view on the event loop before execution.
    snapshot = build_snapshot(
        hass,
        scope=settings.scope,
        anchor_device_id=llm_context.device_id,
    )

    # Build the LLM context view from the live request metadata.
    real_context = llm_context.context if llm_context.context is not None else Context()
    area_id, area_name, floor_id, floor_name = _snapshot_location(snapshot, llm_context.device_id)
    safe_context = build_llm_context(
        platform=llm_context.platform,
        context_id=real_context.id,
        parent_id=real_context.parent_id,
        user_id=real_context.user_id,
        language=llm_context.language,
        assistant=llm_context.assistant,
        device_id=llm_context.device_id,
        area_id=area_id,
        area_name=area_name,
        floor_id=floor_id,
        floor_name=floor_name,
    )

    async def _invoke(action: ProposedAction) -> object:
        return await hass.services.async_call(
            cast(str, action["domain"]),
            cast(str, action["service"]),
            service_data=cast(dict[str, Any] | None, action["service_data"] or None),
            target=cast(dict[str, Any] | None, action["target"]),
            blocking=cast(bool, action["blocking"]),
            return_response=cast(bool, action["return_response"]),
            context=real_context,
        )

    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=settings.helper_call_budget),
        settings=settings,
        invoke=_invoke,
        deadline=deadline,
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
            "execute_home_code: status=%s helper_calls=%s/%s actions=%d",
            execution.get("status") if isinstance(execution, dict) else "n/a",
            runtime.state.helper_calls,
            runtime.state.helper_call_limit,
            len(runtime.state.actions),
        )
    return cast(JsonObjectType, result)


def _build_api_prompt(hass: HomeAssistant, llm_context: llm.LLMContext, actions_enabled: bool) -> str:
    """Return the base API prompt plus concise initiating-location context."""
    prompt = BASE_API_PROMPT
    # Disabled actions add an explicit override section after the general API docs.
    if not actions_enabled:
        prompt = f"{prompt}\n\n{ACTIONS_DISABLED_PROMPT}"
    location_prompt = _request_location_prompt(hass, llm_context.device_id)
    if location_prompt is None:
        return prompt
    return f"{prompt}\n\n{location_prompt}"


def _request_location_prompt(hass: HomeAssistant, device_id: str | None) -> str | None:
    """Render a compact prompt section for the initiating device location."""
    if device_id is None:
        return None

    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)
    floor_registry = fr.async_get(hass)

    device = device_registry.async_get(device_id)
    area_id = device.area_id if device is not None else None
    area = area_registry.async_get_area(area_id) if area_id is not None else None
    floor_id = area.floor_id if area is not None else None
    floor = floor_registry.async_get_floor(floor_id) if floor_id is not None else None

    lines = [
        "## Request location",
        f"- device_id: {device_id}",
    ]
    if area is not None:
        lines.append(f"- area_id: {area.id} ({area.name})")
    if floor is not None:
        lines.append(f"- floor_id: {floor.floor_id} ({floor.name})")
    if area is not None:
        lines.append(
            "For underspecified local questions, use this area as the default scope. "
            "If the user asks for the whole home or names another area/floor, follow that explicit scope."
        )
    return "\n".join(lines)


def _snapshot_location(
    snapshot: HomeSnapshot,
    device_id: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve initiating-device location from the fresh Monty view."""
    if device_id is None:
        return None, None, None, None

    device = snapshot.devices.get(device_id)
    area_id = device.area_id if device is not None else None
    area = snapshot.areas.get(area_id) if area_id is not None else None
    floor_id = area.floor_id if area is not None else None
    floor = snapshot.floors.get(floor_id) if floor_id is not None else None
    return (
        area_id,
        area.name if area is not None else None,
        floor_id,
        floor.name if floor is not None else None,
    )
