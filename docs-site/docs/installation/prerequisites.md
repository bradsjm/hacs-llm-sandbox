---
title: Prerequisites
description: Required Home Assistant, HACS, model, recorder, logbook, and camera capabilities.
---

# Prerequisites

## Home Assistant

Assist Agent Sandbox currently targets Home Assistant `2026.6.4` or newer and Python `3.14.2` or newer. These requirements are declared in [`hacs.json`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/hacs.json), [`manifest.json`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/manifest.json), and [`pyproject.toml`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/pyproject.toml).

## Installation path

Use HACS 2.0 or newer and add this repository as a custom integration repository. The integration is not a core Home Assistant integration.

## Conversation agent

You need a conversation agent that supports Home Assistant LLM tools and is configured as an Assist agent. The integration provides the tools; the conversation agent and model decide when and how to call them.

## Model quality

Use a strong tool-calling model. The main tool asks the model to write short Python snippets, pick APIs, interpret results, and recover from structured guidance. Weak models may fail by writing unsupported code or using the wrong tool.

## Recorder, logbook, and camera requirements

- `get_history` and `get_statistics` require Home Assistant recorder runtime support.
- `get_logbook` requires logbook runtime data in addition to recorder support.
- `get_automation` summaries do not require recorder or logbook; its optional `runs` projection requires both.
- `get_camera_image` requires a visible `camera.*` or `image.*` entity and a multimodal model that can interpret the returned image.

## Before enabling actions

Actions are off by default. Before enabling them, decide which service domains the assistant may control and keep the allowlist tight.
