---
title: Integration Lifecycle
description: Setup, runtime data, update handling, and unload behavior.
---

# Integration Lifecycle

The integration lifecycle is implemented in [`custom_components/llm_sandbox/__init__.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/__init__.py).

## Setup

`async_setup` registers no Home Assistant services. `async_setup_entry` resolves settings, stores a typed runtime object on `entry.runtime_data`, registers the LLM API, and adds an update listener.

## Runtime data

Per-entry runtime state is stored on `entry.runtime_data`. That keeps the loaded integration state typed and tied to the Home Assistant config entry lifecycle.

## Updates

When options change, the update listener reloads the config entry so runtime settings and the registered LLM API are refreshed.

## Unload

Unload delegates cleanup to the callback registered during setup. The unload path returns success because the lifecycle helper owns the unregister behavior.
