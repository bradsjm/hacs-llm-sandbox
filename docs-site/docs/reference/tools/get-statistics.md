---
title: get_statistics
description: Reference for bounded long-term statistics queries.
---

# `get_statistics`

Reads visible Home Assistant long-term statistics.

## Inputs

Common inputs include:

- `hours`, or ISO `start` and `end`.
- Explicit statistic or entity IDs, depending on model/tool usage.
- Selectors such as `area_id`, `device_id`, `floor_id`, `label_id`, and `domain`.
- Statistic period: `5minute`, `hour`, or `day`.
- Statistic types such as `mean`, `min`, `max`, `state`, and `sum`.
- Pagination cursor for older rows. A `next_cursor` can only be passed back to `get_statistics` with the same resolved scope; omit `start`, `end`, and `hours` when using it.

## Bounds

Statistics default to a 24-hour window and are capped at 30 days. Output rows are bounded.

## Source

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py) and [`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py).
