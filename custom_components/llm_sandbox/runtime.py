"""Typed runtime data for LLM Sandbox config entries."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_ASSISTANT,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXCLUDED_ENTITY_CATEGORIES,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_SCOPE_MODE,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXCLUDED_ENTITY_CATEGORIES,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_SCOPE_MODE,
)
from .snapshot import ScopeMode, SnapshotScope

type SandboxConfigEntry = ConfigEntry[SandboxRuntime]


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Per-entry configurable settings read from entry.options."""

    execution_timeout_seconds: int
    helper_call_budget: int
    scope: SnapshotScope


def settings_from_entry(entry: SandboxConfigEntry) -> SandboxSettings:
    """Read typed settings from entry options, applying defaults."""
    options = entry.options
    assistant = entry.data[CONF_ASSISTANT]
    scope = SnapshotScope(
        mode=ScopeMode(options.get(CONF_SCOPE_MODE, DEFAULT_SCOPE_MODE)),
        assistant=assistant,
        excluded_entity_categories=frozenset(
            options.get(CONF_EXCLUDED_ENTITY_CATEGORIES, DEFAULT_EXCLUDED_ENTITY_CATEGORIES)
        ),
        exclude_hidden=bool(options.get(CONF_EXCLUDE_HIDDEN, DEFAULT_EXCLUDE_HIDDEN)),
    )
    return SandboxSettings(
        execution_timeout_seconds=int(options.get(CONF_EXECUTION_TIMEOUT, DEFAULT_EXECUTION_TIMEOUT_SECONDS)),
        helper_call_budget=int(options.get(CONF_HELPER_CALL_BUDGET, DEFAULT_HELPER_CALL_BUDGET)),
        scope=scope,
    )


@dataclass(slots=True)
class SandboxRuntime:
    """Per-entry runtime data stored on the config entry.

    The LLM Sandbox integration builds a fresh snapshot per tool call, so the
    only per-entry state is the resolved settings; there is no background
    manager, coordinator, or scheduler.
    """

    settings: SandboxSettings
