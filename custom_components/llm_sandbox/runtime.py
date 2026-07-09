"""Typed runtime data for LLM Sandbox config entries."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_ACTION_DOMAINS,
    CONF_ACTIONS_ENABLED,
    CONF_ASSISTANT,
    CONF_EXCLUDE_CONFIG,
    CONF_EXCLUDE_HIDDEN,
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    CONF_INCLUDE_ALL_DIAGNOSTICS,
    CONF_PROMPT_PROFILE,
    CONF_RESTRICT_TO_ASSIST_EXPOSED,
    DEFAULT_ACTION_DOMAINS,
    DEFAULT_ACTIONS_ENABLED,
    DEFAULT_EXCLUDE_CONFIG,
    DEFAULT_EXCLUDE_HIDDEN,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
    DEFAULT_INCLUDE_ALL_DIAGNOSTICS,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
)
from .llm_api.prompts import PromptProfile, resolve_profile
from .llm_api.resolution_memory import ResolutionMemoryStore
from .snapshot import SnapshotScope

type SandboxConfigEntry = ConfigEntry[SandboxRuntime]

OPTION_DEFAULTS: Mapping[str, object] = {
    CONF_RESTRICT_TO_ASSIST_EXPOSED: DEFAULT_RESTRICT_TO_ASSIST_EXPOSED,
    CONF_EXCLUDE_HIDDEN: DEFAULT_EXCLUDE_HIDDEN,
    CONF_EXCLUDE_CONFIG: DEFAULT_EXCLUDE_CONFIG,
    CONF_INCLUDE_ALL_DIAGNOSTICS: DEFAULT_INCLUDE_ALL_DIAGNOSTICS,
    CONF_ACTIONS_ENABLED: DEFAULT_ACTIONS_ENABLED,
    CONF_ACTION_DOMAINS: DEFAULT_ACTION_DOMAINS,
    CONF_EXECUTION_TIMEOUT: DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    CONF_HELPER_CALL_BUDGET: DEFAULT_HELPER_CALL_BUDGET,
    CONF_PROMPT_PROFILE: DEFAULT_PROMPT_PROFILE,
}


def option_value(options: Mapping[str, object], key: str) -> object:
    """Return an entry option value, falling back to the single default table."""
    return options.get(key, OPTION_DEFAULTS[key])


def normalize_action_domains(domains: Iterable[str]) -> list[str]:
    """Trim, lowercase, drop blank, and dedupe configured action domains."""
    normalized: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        value = domain.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


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
    excluded_categories: set[str] = set()
    # Visibility category restrictions are assembled from independent toggles.
    if exclude_config:
        excluded_categories.add("config")
    scope = SnapshotScope(
        assistant=assistant,
        restrict_to_assist_exposed=bool(option_value(options, CONF_RESTRICT_TO_ASSIST_EXPOSED)),
        exclude_hidden=bool(option_value(options, CONF_EXCLUDE_HIDDEN)),
        excluded_entity_categories=frozenset(excluded_categories),
        include_all_diagnostics=bool(option_value(options, CONF_INCLUDE_ALL_DIAGNOSTICS)),
    )
    prompt_profile = resolve_profile(str(option_value(options, CONF_PROMPT_PROFILE)))
    return SandboxSettings(
        execution_timeout_seconds=int(cast(int, option_value(options, CONF_EXECUTION_TIMEOUT))),
        helper_call_budget=int(cast(int, option_value(options, CONF_HELPER_CALL_BUDGET))),
        scope=scope,
        actions_enabled=bool(option_value(options, CONF_ACTIONS_ENABLED)),
        action_domains=frozenset(
            normalize_action_domains(cast(Iterable[str], option_value(options, CONF_ACTION_DOMAINS)))
        ),
        prompt_profile=prompt_profile,
    )


@dataclass(slots=True)
class SandboxRuntime:
    """Per-entry runtime data stored on the config entry.

    The LLM Sandbox integration builds a fresh snapshot per tool call, so the
    only cross-call state is the resolved settings plus a small advisory
    conversation resolution cache; there is no background manager, coordinator,
    or scheduler.
    """

    settings: SandboxSettings
    memory_store: ResolutionMemoryStore = field(default_factory=ResolutionMemoryStore)
