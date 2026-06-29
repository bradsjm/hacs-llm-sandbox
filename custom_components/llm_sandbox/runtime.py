"""Typed runtime data for LLM Sandbox config entries."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from .const import (
    CONF_EXECUTION_TIMEOUT,
    CONF_HELPER_CALL_BUDGET,
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_HELPER_CALL_BUDGET,
)

type SandboxConfigEntry = ConfigEntry[SandboxRuntime]


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Per-entry configurable settings read from entry.options."""

    execution_timeout_seconds: int
    helper_call_budget: int


def settings_from_entry(entry: SandboxConfigEntry) -> SandboxSettings:
    """Read typed settings from entry options, applying defaults."""
    options = entry.options
    return SandboxSettings(
        execution_timeout_seconds=int(options.get(CONF_EXECUTION_TIMEOUT, DEFAULT_EXECUTION_TIMEOUT_SECONDS)),
        helper_call_budget=int(options.get(CONF_HELPER_CALL_BUDGET, DEFAULT_HELPER_CALL_BUDGET)),
    )


@dataclass(slots=True)
class SandboxRuntime:
    """Per-entry runtime data stored on the config entry.

    The LLM Sandbox integration builds a fresh snapshot per tool call, so the
    only per-entry state is the resolved settings; there is no background
    manager, coordinator, or scheduler.
    """

    settings: SandboxSettings
