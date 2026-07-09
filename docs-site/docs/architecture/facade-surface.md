---
title: Facade Surface
description: The Home Assistant-like API exposed inside Monty.
---

# Facade Surface

Facades are safe objects built from snapshot records. They preserve useful Home Assistant read idioms without exposing live Home Assistant internals.

## State facade

[`facades/state.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/state.py) provides `hass.states` methods such as `get`, `async_all`, `is_state`, `async_entity_ids`, and `entity_ids`. `SafeHass` exposes only `states`, `services`, and `config`, plus async helpers such as `hass.history(...)` and `hass.query(...)`.

## Registry facades

[`facades/registries.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/registries.py) mirrors common Home Assistant registry patterns, including `er.async_get(hass)` and instance methods over entity, device, area, floor, label, category, issue, notification, and config-entry records.

## Service facade

[`facades/services.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/services.py) exposes the service catalog and the gated `async_call` path.

## Boundary

Facades are not live registries or live Home Assistant objects. They are read interfaces over frozen snapshot data, with service calls routed through a private runtime path only when enabled.
