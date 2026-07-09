---
title: Architecture Overview
description: Main runtime components and how a tool call flows through the integration.
---

# Architecture Overview

Assist Agent Sandbox is a Home Assistant custom integration with an LLM API layer, snapshot builder, Monty executor, facade surface, recorder helpers, vision helper, and action-gating path.

## Main components

| Component | Source | Responsibility |
| --- | --- | --- |
| Integration lifecycle | [`__init__.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/__init__.py) | Set up config entries, runtime data, unload callbacks, and LLM API registration. |
| Options flow | [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) | Expose prompt, visibility, action, and limit settings. |
| LLM API | [`llm_api/api.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/api.py) | Build prompts and available tools for the conversation agent. |
| Snapshot pipeline | [`snapshot/builder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/builder.py) | Build fresh filtered snapshots from live Home Assistant data. |
| Executor | [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py) | Normalize, type-check, run, and format Monty code execution. |
| Facades | [`llm_api/facades/`](https://github.com/bradsjm/hacs-llm-sandbox/tree/main/custom_components/llm_sandbox/llm_api/facades) | Provide HA-like read APIs and gated service calls over snapshot data. |
| Recorder tools | [`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py) | Expose bounded history, statistics, and logbook helpers. |
| Vision tool | [`tools/vision.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/vision.py) | Capture visible camera or image entities for multimodal models. |

## Tool-call flow

1. A conversation agent decides to call a tool.
2. The tool validates setup and input.
3. The integration builds a fresh snapshot for that tool call.
4. The tool either queries recorder data, captures an image, or runs Monty code over facades.
5. Results are converted to JSON-compatible payloads for the model.

No tool call reuses a prior live object reference inside Monty.
