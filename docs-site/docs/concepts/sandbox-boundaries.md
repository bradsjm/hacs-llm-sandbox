---
title: Sandbox Boundaries
description: What code inside Monty can access and what stays outside.
---

# Sandbox Boundaries

`execute_home_code` runs model-written Python in Monty with frozen facade objects. The live Home Assistant object does not enter the sandbox.

## Not exposed to Monty

The sandbox does not receive:

- Live `hass`.
- Live registries.
- Event bus.
- Auth objects.
- Config files.
- Filesystem APIs.
- Network APIs.
- OS or process APIs.
- Live service callables.

## Exposed to Monty

The sandbox receives safe facades built from snapshot records and a small allowed import surface. The README documents imports as limited to `json`, `math`, and `re`.

The executor and facade construction live in [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py). Runtime-only dependencies are carried privately in [`sandbox_context.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/sandbox_context.py).
