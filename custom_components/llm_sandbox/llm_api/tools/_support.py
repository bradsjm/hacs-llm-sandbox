"""Shared setup helpers for LLM Sandbox tools.

These resolve the loaded LLM Sandbox config entry (and its settings) that every
tool call needs, and provide input normalization shared across tool modules.
Kept in a neutral module so neither ``code.py`` nor ``recorder.py`` has to
import from ``api.py`` (which would recreate a cycle).
"""

from collections.abc import Callable, Mapping
from typing import cast

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
import voluptuous as vol

from ...const import DOMAIN
from ...runtime import SandboxConfigEntry, SandboxRuntime
from ..errors import setup_error_payload


def _bounded_list(
    field: str,
    *,
    min_items: int = 0,
    max_items: int | None = None,
) -> Callable[[object], list[object]]:
    """Return a converter-transparent whole-list bounds validator."""

    def validate(value: object) -> list[object]:
        if not isinstance(value, list):
            raise vol.Invalid(f"{field} must be a list")
        if len(value) < min_items:
            raise vol.Invalid(f"{field} must contain at least {min_items} item(s)")
        if max_items is not None and len(value) > max_items:
            raise vol.Invalid(f"{field} must contain at most {max_items} item(s)")
        return value

    return validate


def _omit_empty_optional_args(
    args: Mapping[str, object],
    *,
    null_keys: frozenset[str],
    empty_string_keys: frozenset[str] = frozenset(),
    empty_list_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Omit empty/null optional arguments before voluptuous validation.

    Implements Postel's law for LLM tool inputs: a null, empty-string, or
    empty-list optional value is dropped as if never sent, so the schema
    applies its default (or treats the key as absent). Required keys and
    non-empty values are preserved unchanged; malformed non-empty values still
    surface the natural schema error.
    """
    result: dict[str, object] = {}
    for key, value in args.items():
        # A null optional value is dropped for every key in null_keys.
        if value is None and key in null_keys:
            continue
        # An empty-string optional value is dropped for scalar-string keys.
        if isinstance(value, str) and value == "" and key in empty_string_keys:
            continue
        # An empty-list optional value is dropped for list keys.
        if isinstance(value, list) and not value and key in empty_list_keys:
            continue
        result[key] = value
    return result


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
