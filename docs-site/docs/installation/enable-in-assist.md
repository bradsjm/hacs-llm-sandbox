---
title: Enable In Assist
description: The required conversation-agent step after installation.
---

# Enable In Assist

After installing the integration, open the settings for the conversation agent you use with Assist and enable the `Assist Agent Sandbox` tool set.

This is the most common missing step. If the tools are not enabled in the agent, the integration can be installed correctly but the model will never call `execute_home_code`, `get_history`, `get_statistics`, `get_logbook`, or `get_camera_image`.

## What should be enabled

Enable the tool set exposed by the integration entry. Tool availability is built by [`LlmSandboxAPI._build_tools`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/api.py):

- `execute_home_code` is always registered for a loaded entry.
- `get_history` and `get_statistics` are added when recorder support is available.
- `get_logbook` is added when logbook support is available.
- `get_camera_image` is always added, but it can only capture visible `camera.*` or `image.*` entities.

## Quick verification

Ask Assist a question that clearly needs the tools, such as:

```text
Which lights are currently on, grouped by area?
```

If your agent UI shows tool calls, you should see the model call the sandbox tool set.
