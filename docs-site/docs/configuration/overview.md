---
title: Configuration Overview
description: The configuration sections exposed by the integration options flow.
---

# Configuration Overview

Open the integration's `Configure` dialog in Home Assistant. Options are grouped into four areas:

| Section | Purpose |
| --- | --- |
| Prompt | Selects the base prompt profile sent to the model. |
| Visibility | Controls which snapshot records are exposed to the tools. |
| Actions | Controls whether service calls are allowed and which domains may be used. |
| Execution Limits | Bounds code runtime and service-call count. |

The options flow and defaults are implemented in [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py), [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py), and [`const.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/const.py).

## Defaults

The default configuration is conservative:

- Assist-exposed entity filtering is on.
- Hidden entities are excluded.
- Config-category entities are excluded.
- Only selected useful diagnostic entities are included.
- Actions are disabled.
- Execution timeout is 12 seconds.
- Service-call limit is 32 dispatched calls.
- Prompt profile is `balanced`.
