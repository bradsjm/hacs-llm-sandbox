---
title: Execution Limits
description: Runtime limits for code execution and service calls.
---

# Execution Limits

Execution limits bound each `execute_home_code` call.

| Option | Default | Range |
| --- | --- | --- |
| Maximum execution time | 12 seconds | 3 to 30 seconds |
| Maximum service calls per request | 32 | 1 to 100 calls |

The option ranges are defined in [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py). The executor also applies Monty resource limits in [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py), including bounded memory and allocation settings.

## How to choose values

- Keep the timeout near the default unless your model regularly runs legitimate longer summaries.
- Lower the service-call budget if you enable actions and want stricter operational control.
- Raise limits only after observing repeated valid failures, not as a first troubleshooting step.
