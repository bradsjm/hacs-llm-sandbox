"""Home Assistant LLM API for LLM Sandbox."""

from typing import cast, final, override

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import llm

from ..const import DEFAULT_PROMPT_PROFILE, DOMAIN
from ..runtime import SandboxConfigEntry
from .prompts import compose_system_prompt, render_request_location, resolve_profile
from .tools import ExecuteHomeCodeTool, GetCameraImageTool, GetHistoryTool, GetLogbookTool, GetStatisticsTool


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
        base_prompt = resolve_profile(DEFAULT_PROMPT_PROFILE).base_prompt
        # Missing, wrong-domain, unloaded, or uninitialized entries get a conservative prompt.
        if (
            entry is not None
            and entry.domain == DOMAIN
            and entry.state is ConfigEntryState.LOADED
            and entry.runtime_data is not None
        ):
            typed_entry = cast(SandboxConfigEntry, entry)
            runtime_data = typed_entry.runtime_data
            assert runtime_data is not None
            actions_enabled = runtime_data.settings.actions_enabled
            base_prompt = runtime_data.settings.prompt_profile.base_prompt
        return llm.APIInstance(
            api=self,
            api_prompt=_build_api_prompt(self._hass, llm_context, base_prompt, actions_enabled),
            llm_context=llm_context,
            tools=[
                ExecuteHomeCodeTool(self.entry_id),
                GetHistoryTool(self.entry_id),
                GetStatisticsTool(self.entry_id),
                GetLogbookTool(self.entry_id),
                GetCameraImageTool(self.entry_id),
            ],
        )


def _build_api_prompt(
    hass: HomeAssistant, llm_context: llm.LLMContext, base_prompt: str, actions_enabled: bool
) -> str:
    """Return the base API prompt plus concise initiating-location context."""
    location_prompt = _request_location_prompt(hass, llm_context.device_id)
    return compose_system_prompt(base_prompt, actions_enabled, location_prompt)


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

    return render_request_location(
        device_id,
        area.id if area is not None else None,
        area.name if area is not None else None,
        floor.floor_id if floor is not None else None,
        floor.name if floor is not None else None,
    )
