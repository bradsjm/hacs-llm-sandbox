---
title: Service Actions
description: Let Assist control devices only when action gates are enabled.
---

# Service Actions

Service actions are disabled by default. When enabled, the model can call Home Assistant services through `hass.services.async_call`, but every call is checked first.

## Example prompts

```text
Turn off all visible lights in the living room.
```

```text
If any basement lights are on, turn them off and tell me which ones changed.
```

## Runtime behavior

When a service call succeeds, the response includes an action record. If the model reads `hass.states` again in the same code run, it still sees the frozen snapshot from the start of the run. The executor adds a note when actions succeed so the model knows to call a tool again if it needs fresh state.

## Blocked actions

Policy-blocked service calls are recorded as errored actions and do not crash the whole code run. That makes policy failures visible without treating them as Python bugs.

Service-call behavior is implemented in [`facades/services.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/services.py).
