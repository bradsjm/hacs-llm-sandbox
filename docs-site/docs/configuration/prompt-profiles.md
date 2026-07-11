---
title: Prompt Profiles
description: Choose the model-guidance strategy used by the LLM API.
---

# Prompt Profiles

Prompt profiles tune how the LLM API explains its tools and sandbox surface. All profiles expose the same capabilities and safety boundaries; they differ only in guidance, examples, and presentation density.

Profiles are selected in the options flow and resolved through runtime settings in [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) and [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py).

## Choosing a profile

| Profile | Use when |
| --- | --- |
| `guided` | A weaker or more literal model benefits from explicit routing and compact code examples. |
| `balanced` | You want the readable default: complete capability guidance without tutorial examples or repeated coaching. |
| `frontier` | You are using a GPT-5.6-class model and want a compact, complete capability contract. |

The profile never changes the runtime safety boundary. Fresh frozen snapshots, Monty execution, visibility filtering, recorder limits, and action validation continue to apply regardless of selection.
