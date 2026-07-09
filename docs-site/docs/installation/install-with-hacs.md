---
title: Install With HACS
description: Install Assist Agent Sandbox as a custom HACS integration.
---

# Install With HACS

1. Open HACS in Home Assistant.
2. Add `https://github.com/bradsjm/hacs-llm-sandbox` as a custom repository.
3. Select repository type `Integration`.
4. Find `Assist Agent Sandbox` in HACS and install it.
5. Restart Home Assistant.
6. Go to `Settings -> Devices & services -> Add integration`.
7. Search for `Assist Agent Sandbox`.
8. Confirm the integration name and assistant scope.

The current implementation supports one assistant scope and defaults to the `conversation` assistant. That setup path is defined in [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py).

## After installation

Installation only creates the integration entry. You must also enable the tool set in your conversation agent settings before the model can call these tools.
