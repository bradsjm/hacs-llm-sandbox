"""Typed runtime data for LLM Sandbox config entries."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_ACTION_DOMAINS,
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_EXCLUDE_CONFIG,
    CONF_EXCLUDE_DIAGNOSTIC,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ACTION_DOMAINS,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_EXCLUDE_CONFIG,
    DEFAULT_EXCLUDE_DIAGNOSTIC,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
)
from .snapshot import SnapshotScope

type SandboxConfigEntry = ConfigEntry[SandboxRuntime]


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Per-entry configurable settings read from entry.options."""

    execution_timeout_seconds: int
    helper_call_budget: int
    scope: SnapshotScope
    actions_enabled: bool
    action_domains: frozenset[str]


def settings_from_entry(entry: SandboxConfigEntry) -> SandboxSettings:
    """Read typed settings from entry options, applying defaults."""
    options = entry.options
    assistant = entry.data[CONF_ASSISTANT]
    exclude_config = bool(options.get(CONF_EXCLUDE_CONFIG, DEFAULT_EXCLUDE_CONFIG))
    exclude_diagnostic = bool(options.get(CONF_EXCLUDE_DIAGNOSTIC, DEFAULT_EXCLUDE_DIAGNOSTIC))
    excluded_categories: set[str] = set()
    # Visibility category restrictions are assembled from independent toggles.
    if exclude_config:
        excluded_categories.add("config")
    # Visibility category restrictions are assembled from independent toggles.
    if exclude_diagnostic:
        excluded_categories.add("diagnostic")
    scope = SnapshotScope(
        assistant=assistant,
        restrict_to_assist_exposed=bool(
            options.get(CONF_RESTRICT_TO_ASSIST_EXPOSED, DEFAULT_RESTRICT_TO_ASSIST_EXPOSED)
        ),
        exclude_hidden=bool(options.get(CONF_EXCLUDE_HIDDEN, DEFAULT_EXCLUDE_HIDDEN)),
        excluded_entity_categories=frozenset(excluded_categories),
    )
    return SandboxSettings(
        execution_timeout_seconds=int(options.get(CONF_EXECUTION_TIMEOUT, DEFAULT_EXECUTION_TIMEOUT_SECONDS)),
        helper_call_budget=int(options.get(CONF_HELPER_CALL_BUDGET, DEFAULT_HELPER_CALL_BUDGET)),
        scope=scope,
        actions_enabled=bool(options.get(CONF_ACTIONS_ENABLED, DEFAULT_ACTIONS_ENABLED)),
        action_domains=frozenset(options.get(CONF_ACTION_DOMAINS, DEFAULT_ACTION_DOMAINS)),
    )


@dataclass(slots=True)
class SandboxRuntime:
    """Per-entry runtime data stored on the config entry.

    The LLM Sandbox integration builds a fresh snapshot per tool call, so the
    only per-entry state is the resolved settings; there is no background
    manager, coordinator, or scheduler.
    """

    settings: SandboxSettings
