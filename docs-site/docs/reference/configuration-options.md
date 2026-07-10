---
title: Configuration Options
description: Current options, defaults, and ranges.
---

# Configuration Options

| Option | Default | Range or values | Source |
| --- | --- | --- | --- |
| Prompt profile | `standard` | `standard`, `terse`, `minimal` | [`const.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/const.py) |
| Restrict to Assist-exposed entities | On | Boolean | [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py) |
| Exclude hidden entities | On | Boolean | [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py) |
| Exclude configuration entities | On | Boolean | [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py) |
| Include all diagnostic entities | Off | Boolean | [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py) |
| Enable actions | Off | Boolean | [`runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/runtime.py) |
| Allowed service domains | Empty | Domain IDs | [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) |
| Maximum execution time | 12 seconds | 3 to 30 seconds | [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) |
| Maximum service calls per request | 32 | 1 to 100 validated calls dispatched to Home Assistant | [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py) |
