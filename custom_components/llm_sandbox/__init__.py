"""LLM Sandbox integration lifecycle."""

from collections.abc import Mapping

from homeassistant.core import HomeAssistant

from .llm_api.api import async_register_llm_api
from .runtime import SandboxConfigEntry, SandboxRuntime, settings_from_entry


async def async_setup(hass: HomeAssistant, _config: Mapping[str, object]) -> bool:
    """Set up LLM Sandbox.

    No domain services are registered. The integration exposes its tools through
    one LLM API registered per config entry.
    """
    _ = hass
    return True


async def async_setup_entry(hass: HomeAssistant, entry: SandboxConfigEntry) -> bool:
    """Set up an LLM Sandbox config entry.

    The integration builds a fresh snapshot on each tool call, so setup only
    resolves the per-entry settings and registers the entry-scoped LLM API.
    """
    settings = settings_from_entry(entry)
    entry.runtime_data = SandboxRuntime(settings=settings)
    async_register_llm_api(hass, entry)
    entry.async_on_unload(entry.add_update_listener(_async_update_entry))
    return True


async def async_unload_entry(_hass: HomeAssistant, _entry: SandboxConfigEntry) -> bool:
    """Unload an LLM Sandbox config entry.

    There is no background manager, coordinator, or scheduler to tear down.
    The LLM API unregister hook attached via ``entry.async_on_unload`` handles
    cleanup, so unload always succeeds.
    """
    return True


async def _async_update_entry(hass: HomeAssistant, entry: SandboxConfigEntry) -> None:
    """Reload the entry after options are updated."""
    _ = await hass.config_entries.async_reload(entry.entry_id)
