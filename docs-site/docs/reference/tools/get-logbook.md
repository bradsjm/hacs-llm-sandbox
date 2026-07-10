---
title: get_logbook
description: Reference for bounded Home Assistant logbook activity queries.
---

# `get_logbook`

Reads visible Home Assistant logbook activity entries.

## Inputs

Common inputs include:

- `hours`, or ISO `start` and `end`.
- Explicit `entity_ids`, capped at 20.
- Selectors such as `area_id`, `device_id`, `floor_id`, `label_id`, and `domain`.
- Pagination cursor for older entries. A `next_cursor` can only be passed back to `get_logbook` with the same resolved scope; omit `start`, `end`, and `hours` when using it.

## Bounds

Logbook defaults to a bounded recent window and is capped at 24 hours. Cursor pages normally fit their complete compact UTF-8 JSON response within 16 KiB; 200 entries remains an emergency ceiling. Entries are never split, and a single oversized entry is returned intact on its own so the existing cursor can make progress.

## Runtime requirement

The tool is registered only when logbook runtime support is available.

## Source

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py).
