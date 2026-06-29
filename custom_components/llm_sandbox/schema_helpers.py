"""Config-flow schema helpers for collapsible sections."""

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.helpers.typing import VolDictType


def section_defaults(section_schema: VolDictType) -> dict[str, Any]:
    """Return defaults for a section from its nested field markers."""
    defaults: dict[str, Any] = {}
    for key in section_schema:
        key_default = getattr(key, "default", vol.Undefined)
        # Only string-backed voluptuous markers map cleanly to HA section data.
        if isinstance(key, vol.Marker) and isinstance(key.schema, str) and not isinstance(key_default, vol.Undefined):
            defaults[key.schema] = key_default()
    return defaults


def section_schema_key(section_name: str, section_schema: VolDictType) -> vol.Optional:
    """Return a section marker whose default mirrors its nested schema values."""
    return vol.Optional(section_name, default=section_defaults(section_schema))


def flatten_section_data(data: Mapping[str, Any], section_keys: list[str]) -> dict[str, Any]:
    """Return form data with HA section namespaces flattened to top-level keys."""
    flattened = dict(data)
    for key in section_keys:
        value = flattened.pop(key, None)
        # Home Assistant sections submit nested mappings keyed by section name.
        if isinstance(value, Mapping):
            flattened.update(value)
        elif value is not None:
            flattened[key] = value
    return flattened
