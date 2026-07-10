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
- Pagination cursor for raw rows. A `next_cursor` can only be passed back to `get_history` with the same resolved scope; omit `start`, `end`, and `hours` when using it.

## Bounds

Raw history defaults to a one-hour window and is capped at 24 hours. Aggregate history can look back up to 30 days. Raw cursor pages normally fit their complete compact UTF-8 JSON response within 16 KiB; 1000 rows remains an emergency ceiling. Rows are never split, and a single oversized row is returned intact on its own so the existing cursor can make progress.

## Source

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py) and [`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py).
