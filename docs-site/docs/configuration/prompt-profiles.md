---
title: Prompt Profiles
description: Choose the base prompt profile used by the LLM API.
---

# Prompt Profiles

Prompt profiles tune the base instructions exposed through the LLM API. The current options are `standard`, `terse`, and `minimal`.

Profiles are selected by the options flow in [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) and resolved through runtime settings in [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py).

## Choosing a profile

| Profile | Use when |
| --- | --- |
| `standard` | You want the richest guidance and the safest default for setup and exploration. |
| `terse` | You want fewer prompt tokens but still want meaningful tool guidance. |
| `minimal` | You are using a strong model and want the shortest base instructions. |

The prompt profile does not change the runtime safety boundary. Snapshots, Monty execution, visibility filtering, and action gates still apply.
