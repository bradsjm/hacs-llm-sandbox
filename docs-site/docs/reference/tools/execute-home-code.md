---
title: execute_home_code
description: Reference for bounded Python/Monty execution over Home Assistant snapshot facades.
---

# `execute_home_code`

Runs a short Python/Monty snippet over a fresh, frozen, visibility-filtered Home Assistant snapshot.

## Primary input

| Field | Meaning |
| --- | --- |
| `code` | Python code to run in Monty. The current maximum is 8000 characters. |

## Available data

Inside code, the model can use Home Assistant-like facades such as `hass.states`, registry helpers, `await hass.history(...)`, `await hass.logbook(...)`, `await hass.query(...)`, and gated `await hass.services.async_call(...)`. State, registry, config, and service-catalog reads are synchronous.

## Recorder composition

Use one `execute_home_code` call when recorder data must be combined with current state or registries, computed or compared across sources, used for conditional reasoning, or used to decide or perform an action. Use the matching standalone recorder tool for a direct history, statistics, or logbook answer; independent direct reads can run in parallel.

`await hass.logbook(entity_ids=None, hours=None)` accepts up to 20 visible entities, defaults to and caps at 24 hours, and returns at most the newest 200 chronological JSON-safe entries. It has no cursor and requires recorder plus logbook runtime support.

## Result shape

Successful calls return an object with `execution.status == "ok"` and `output` containing the JSON-safe result. The payload can also include top-level `printed`, `notes`, `actions`, normalization adjustments, and `resolutions`.

Errors are structured as `setup_error`, `code_error`, or `helper_error`.

## Source

- Tool wrapper: [`tools/code.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/code.py)
- Executor: [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py)
- Facades: [`llm_api/facades/`](https://github.com/bradsjm/hacs-llm-sandbox/tree/main/custom_components/llm_sandbox/llm_api/facades)
