"""Typed runtime data for LLM Sandbox config entries."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

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
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ACTION_DOMAINS,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_EXCLUDE_CONFIG,
    DEFAULT_EXCLUDE_DIAGNOSTIC,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
)
from .llm_api.prompts import PromptProfile, resolve_profile
from .snapshot import SnapshotScope

type SandboxConfigEntry = ConfigEntry[SandboxRuntime]

OPTION_DEFAULTS: Mapping[str, object] = {
    CONF_RESTRICT_TO_ASSIST_EXPOSED: DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
    CONF_EXCLUDE_HIDDEN: DEFAULT_EXCLUDE_HIDDEN,
    CONF_EXCLUDE_CONFIG: DEFAULT_EXCLUDE_CONFIG,
    CONF_EXCLUDE_DIAGNOSTIC: DEFAULT_EXCLUDE_DIAGNOSTIC,
    CONF_ACTIONS_ENABLED: DEFAULT_ACTIONS_ENABLED,
    CONF_ACTION_DOMAINS: DEFAULT_ACTION_DOMAINS,
    CONF_EXECUTION_TIMEOUT: DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    CONF_HELPER_CALL_BUDGET: DEFAULT_HELPER_CALL_BUDGET,
    CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE,
}


def option_value(options: Mapping[str, object], key: str) -> object:
    """Return an entry option value, falling back to the single default table."""
    return options.get(key, OPTION_DEFAULTS[key])


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Per-entry configurable settings read from entry.options."""

    execution_timeout_seconds: int
    helper_call_budget: int
    scope: SnapshotScope
    actions_enabled: bool
    action_domains: frozenset[str]
    prompt_profile: PromptProfile


def settings_from_entry(entry: SandboxConfigEntry) -> SandboxSettings:
    """Read typed settings from entry options, applying defaults."""
    options = entry.options
    assistant = entry.data[CONF_ASSISTANT]
    exclude_config = bool(option_value(options, CONF_EXCLUDE_CONFIG))
    exclude_diagnostic = bool(option_value(options, CONF_EXCLUDE_DIAGNOSTIC))
    excluded_categories: set[str] = set()
    # Visibility category restrictions are assembled from independent toggles.
    if exclude_config:
        excluded_categories.add("config")
    # Visibility category restrictions are assembled from independent toggles.
    if exclude_diagnostic:
        excluded_categories.add("diagnostic")
    scope = SnapshotScope(
        assistant=assistant,
        restrict_to_assist_exposed=bool(option_value(options, CONF_RESTRICT_TO_ASSIST_EXPOSED)),
        exclude_hidden=bool(option_value(options, CONF_EXCLUDE_HIDDEN)),
        excluded_entity_categories=frozenset(excluded_categories),
    )
    prompt_profile = resolve_profile(str(option_value(options, CONF_PROMPT_PROFILE)))
    return SandboxSettings(
        execution_timeout_seconds=int(cast(int, option_value(options, CONF_EXECUTION_TIMEOUT))),
        helper_call_budget=int(cast(int, option_value(options, CONF_HELPER_CALL_BUDGET))),
        scope=scope,
        actions_enabled=bool(option_value(options, CONF_ACTIONS_ENABLED)),
        action_domains=frozenset(cast(Iterable[str], option_value(options, CONF_ACTION_DOMAINS))),
        prompt_profile=prompt_profile,
    )


@dataclass(slots=True)
class SandboxRuntime:
    """Per-entry runtime data stored on the config entry.

    The LLM Sandbox integration builds a fresh snapshot per tool call, so the
    only per-entry state is the resolved settings; there is no background
    manager, coordinator, or scheduler.
    """

    settings: SandboxSettings
