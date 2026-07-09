---
title: Recorder Tools
description: Runtime design for history, statistics, logbook, and SQL queries.
---

# Recorder Tools

Recorder-backed behavior is split between public tool classes and runtime helpers.

## Public tools

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py) defines the LLM-facing `get_history`, `get_statistics`, and `get_logbook` tools.

## Runtime helpers

[`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py) implements query execution, validation, scoping, aggregation, pagination, and JSON-safe row formatting.

## SQL helper

[`llm_api/data/home_db.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/home_db.py) builds the bounded in-memory SQLite database used by `hass.query`.

## Selector behavior

Explicit entity IDs are capped and checked for visibility. Selectors such as area, device, floor, label, and domain resolve through snapshot indexes.
