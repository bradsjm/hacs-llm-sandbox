---
title: Introduction
description: What Assist Agent Sandbox does, who it is for, and where to start.
slug: /
sidebar_position: 1
---

# Assist Agent Sandbox

Assist Agent Sandbox is a Home Assistant custom integration that gives Assist conversation agents a bounded tool set for reasoning over your home. It lets a capable tool-calling model inspect current state, ask bounded recorder questions, read the logbook, capture a camera frame, and optionally call services through policy gates.

The integration is intentionally built around snapshots and an isolated Monty Python runtime. The model never receives the live Home Assistant object, live registries, event bus, auth objects, config files, network access, filesystem access, or OS/process APIs.

## Start here

1. Check [prerequisites](installation/prerequisites.md).
2. [Install with HACS](installation/install-with-hacs.md).
3. Complete the critical [Enable in Assist](installation/enable-in-assist.md) step.
4. Try the [quickstart prompts](usage/quickstart.md).
5. Review [privacy and security](operations/privacy-and-security.md) before exposing sensitive entities.

## What the tools provide

| Tool | Purpose |
| --- | --- |
| `execute_home_code` | Runs bounded Python/Monty over frozen Home Assistant snapshot facades. |
| `get_history` | Reads bounded state history from recorder. |
| `get_statistics` | Reads bounded long-term statistics. |
| `get_logbook` | Reads bounded activity timeline entries. |
| `get_camera_image` | Captures a visible camera or image entity for a multimodal model. |

## Source grounding

Important behavior in this site is grounded in the checked-in integration source, especially [`llm_api/api.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/api.py), [`tools/code.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/code.py), [`snapshot/builder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/builder.py), and [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py).
