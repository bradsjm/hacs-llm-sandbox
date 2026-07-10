---
title: Actions
description: Configure whether the assistant may call Home Assistant services.
---

# Actions

Actions control whether code running inside the sandbox can call Home Assistant services through `hass.services.async_call`.

| Option | Default | Effect |
| --- | --- | --- |
| Enable actions on visible entities | Off | Master switch for live service calls. |
| Allowed service domains | Empty | When actions are enabled, limits calls to domains such as `light` or `switch`. Empty means all domains are allowed. |

Service calls are implemented by [`facades/services.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/services.py) and dispatched through a private runtime callback created in [`tools/code.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/code.py). The live `hass` object and live service callable are not exposed to Monty.

## Policy gates

Every call is checked against the fresh snapshot and current settings:

- Actions must be enabled.
- The service domain must be allowed.
- The service must exist in the snapshot service catalog.
- The target must resolve to visible entities when a target is used.
- Response-mode requirements must match the service catalog.
- The validated call must fit inside the service-call limit before it is dispatched.

## Recommendation

Start read-only. If you enable actions, begin with a small domain allowlist such as `light` or `switch` and expand only when you understand the model's behavior.
