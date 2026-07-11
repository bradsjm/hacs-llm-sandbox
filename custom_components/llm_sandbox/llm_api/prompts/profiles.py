"""Source-controlled prompt profile registry for base API prompt selection."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ...const import DEFAULT_PROMPT_PROFILE
from .catalog import render_capability_catalog


class PromptDetail(StrEnum):
    """Composition detail level for profile-aware shared prompt sections."""

    GUIDED = "guided"
    BALANCED = "balanced"
    FRONTIER = "frontier"


@dataclass(frozen=True, slots=True)
class PromptProfile:
    """One selectable base API prompt profile shipped with the integration."""

    id: str
    label: str
    detail: PromptDetail
    base_prompt: str


def _guided_guidance() -> str:
    """Return explicit routing and compact examples for weaker models."""
    return """## Working route
- Automation lookup, complete content, or recent-run questions use the standalone get_automation tool; direct history, statistics, or logbook retrieval uses the matching standalone tool. Current state, registry joins, computation, conditions, composed recorder data, or actions use one execute_home_code call. Independent direct reads may run in parallel. Stop after sufficient evidence; do not refetch it.
- Await hass.history(...), hass.query(...), hass.logbook(...), and enabled hass.services.async_call(...). State, registry, config, repairs, notification, and service-catalog reads are synchronous despite async_-style names. Put the JSON-safe final answer in result or use a final bare expression.

### Current state
```python
state = states.get("<entity_id>")
result = {"entity_id": state.entity_id, "state": state.state} if state else None
```

### Registry join
```python
entry = entity_registry.async_get("<entity_id>")
device = device_registry.async_get(entry.device_id) if entry and entry.device_id else None
result = entry.area_id or (device.area_id if device else None)
```

### Composed recorder read
```python
current = states.get("<entity_id>")
history = await hass.history(entity_ids=["<entity_id>"], hours=6)
result = {"current": current.state if current else None, "history": history}
```

### Enabled action
```python
await hass.services.async_call("<domain>", "<service>", {"<field>": "<value>"}, target={"entity_id": "<entity_id>"}, blocking=True)
result = "requested"
```
Use the action form only when the later service-call section says actions are enabled."""


def _balanced_guidance() -> str:
    """Return the readable default guidance without tutorial examples."""
    return """## Working route
Use the standalone get_automation tool for automation lookup, complete content, or recent-run questions, and the matching standalone tool for direct history, statistics, or logbook retrieval; use one execute_home_code call when current state, registry joins, computation, conditions, composed recorder data, or actions depend on each other. Independent direct reads may run in parallel; stop after sufficient evidence and do not refetch it.

## Execution and output
Await hass.history(...), hass.query(...), hass.logbook(...), and enabled hass.services.async_call(...); state, registry, config, repairs, notification, and service-catalog reads are synchronous despite async_-style names. Return the useful JSON-safe answer in result or a final bare expression."""


def _frontier_guidance() -> str:
    """Return the compact outcome contract for capable models."""
    return """## Outcome contract
Use the least evidence needed for a grounded answer. Choose direct get_automation and recorder tools for independent automation or recorder retrieval, and one composed code call when dependencies require current snapshot data, joins, computation, conditions, recorder access, or actions. Return the useful JSON-safe result."""


def _base_prompt(detail: PromptDetail) -> str:
    """Compose profile-specific guidance with the full canonical catalog."""
    if detail is PromptDetail.GUIDED:
        guidance = _guided_guidance()
    elif detail is PromptDetail.BALANCED:
        guidance = _balanced_guidance()
    else:
        guidance = _frontier_guidance()
    return f"{guidance}\n\n{render_capability_catalog(compact=detail is PromptDetail.FRONTIER)}"


_GUIDED = PromptProfile("guided", "Guided", PromptDetail.GUIDED, _base_prompt(PromptDetail.GUIDED))
_BALANCED = PromptProfile(
    DEFAULT_PROMPT_PROFILE, "Balanced", PromptDetail.BALANCED, _base_prompt(PromptDetail.BALANCED)
)
_FRONTIER = PromptProfile("frontier", "Frontier", PromptDetail.FRONTIER, _base_prompt(PromptDetail.FRONTIER))

PROFILE_OPTIONS: tuple[PromptProfile, ...] = (_GUIDED, _BALANCED, _FRONTIER)
PROFILE_REGISTRY: MappingProxyType[str, PromptProfile] = MappingProxyType(
    {profile.id: profile for profile in PROFILE_OPTIONS}
)


def resolve_profile(profile_id: str) -> PromptProfile:
    """Return the prompt profile for ``profile_id`` or raise if unknown."""
    try:
        return PROFILE_REGISTRY[profile_id]
    except KeyError:
        raise ValueError(f"unknown prompt profile: {profile_id}") from None
