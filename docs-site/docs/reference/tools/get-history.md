---
title: get_history
description: Reference for bounded recorder state-history queries.
---

# `get_history`

Reads visible recorder state history.

## Inputs

Common inputs include:

- `hours`, or ISO `start` and `end`.
- Explicit `entity_ids`, capped at 20.
- Selectors such as `area_id`, `device_id`, `floor_id`, `label_id`, and `domain`.
- Analytics options such as `aggregate`, `group_by`, `bucket`, `where`, `order_by`, and `limit`.
- Pagination cursor for raw rows.

## Bounds

Raw history defaults to a one-hour window and is capped at 24 hours. Aggregate history can look back up to 30 days. Output rows are bounded.

## Source

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py) and [`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py).
