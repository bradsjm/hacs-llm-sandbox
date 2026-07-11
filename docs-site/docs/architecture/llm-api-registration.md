---
title: LLM API Registration
description: How the integration exposes tools to Home Assistant conversation agents.
---

# LLM API Registration

The LLM API layer lives in [`llm_api/api.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/api.py).

## API instance

`LlmSandboxAPI` describes the integration to Home Assistant's LLM API system. If an entry is missing or unloaded at prompt time, the code uses a conservative prompt path. Actual tool calls still require a loaded entry.

## Tool list

The API always exposes `execute_home_code` and `get_automation` for a loaded entry. Recorder and logbook tools are added only when their runtime dependencies are available. `get_automation` is available without recorder support; only its optional `runs` projection requires recorder and logbook runtime data. `get_camera_image` is part of the tool set and performs its own visibility and entity-domain checks when called.

## Fresh data rule

The prompt-time snapshot is advisory. Every tool call builds its own fresh snapshot before doing work.
