---
title: Assist Tool Calling
description: How the integration plugs tools into Home Assistant Assist.
---

# Assist Tool Calling

Home Assistant conversation agents can expose LLM APIs and tools to a model. Assist Agent Sandbox registers an entry-scoped LLM API that describes the tools and lets the agent call them during a conversation.

The registration path is in [`__init__.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/__init__.py) and [`llm_api/api.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/api.py).

## Why tool calling matters

Without tools, Assist can only answer from the model's prompt context and the built-in capabilities of the selected agent. With these tools enabled, the model can request fresh Home Assistant data and use bounded computation to answer questions that are not pre-scripted.

## Tool availability

Tool availability depends on runtime support:

- `execute_home_code` is the main tool.
- Recorder tools require recorder support.
- Logbook requires logbook support.
- Camera capture requires visible camera or image entities at call time.

## Round-aware routing

The model should use `get_history`, `get_statistics`, or `get_logbook` for a direct answer from that one recorder source. It can call independent direct reads in parallel. When recorder data depends on current snapshot state or registries, needs computation or other evidence, or drives a condition or action, it should use one `execute_home_code` call with `await hass.history(...)`, `await hass.query(...)`, or `await hass.logbook(...)`.
