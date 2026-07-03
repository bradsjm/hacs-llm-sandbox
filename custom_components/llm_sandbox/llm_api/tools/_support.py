"""Shared setup helpers for LLM Sandbox tools.

These resolve the loaded LLM Sandbox config entry (and its settings) that every
tool call needs. Kept in a neutral module so neither ``code.py`` nor
``recorder.py`` has to import from ``api.py`` (which would recreate a cycle).
"""

from typing import cast

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from ...const import DOMAIN
from ...runtime import SandboxConfigEntry, SandboxRuntime
from ..errors import setup_error_payload


def _require_loaded_entry(hass: HomeAssistant, entry_id: str) -> SandboxConfigEntry:
    """Return a loaded LLM Sandbox config entry."""
    setup_error = _require_loaded_entry_error(hass, entry_id)
    if setup_error is not None:
        key, placeholders = setup_error
        msg = setup_error_payload(key, placeholders)["execution"]["message"]
        raise RuntimeError(msg)
    return cast(SandboxConfigEntry, hass.config_entries.async_get_entry(entry_id))


def _require_sandbox_runtime(hass: HomeAssistant, entry_id: str) -> SandboxRuntime:
    """Return typed runtime data for a loaded LLM Sandbox config entry."""
    entry = _require_loaded_entry(hass, entry_id)
    runtime_data = entry.runtime_data
    assert runtime_data is not None
    return runtime_data


def _require_loaded_entry_error(
    hass: HomeAssistant,
    entry_id: str,
) -> tuple[str, dict[str, str]] | None:
    """Return setup error metadata when the entry cannot be used."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        return "unknown_config_entry", {"config_entry_id": entry_id}
    typed_entry = cast(SandboxConfigEntry, entry)
    if typed_entry.state is not ConfigEntryState.LOADED or typed_entry.runtime_data is None:
        return "config_entry_not_loaded", {"config_entry_id": entry_id}
    return None
