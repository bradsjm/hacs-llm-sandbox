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

Inside code, the model can use Home Assistant-like facades such as `hass.states`, registry helpers, `hass.history(...)`, `hass.query(...)`, and gated `hass.services.async_call(...)`.

## Result shape

Successful calls return an object with `execution.status == "ok"` and `output` containing the JSON-safe result. The payload can also include printed lines, action records, normalization adjustments, entity resolutions, and notes.

Errors are structured as `setup_error`, `code_error`, or `helper_error`.

## Source

- Tool wrapper: [`tools/code.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/code.py)
- Executor: [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py)
- Facades: [`llm_api/facades/`](https://github.com/bradsjm/hacs-llm-sandbox/tree/main/custom_components/llm_sandbox/llm_api/facades)
