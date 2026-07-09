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

## Bounds

Logbook defaults to a bounded recent window and is capped at 24 hours, with at most 200 returned entries.

## Runtime requirement

The tool is registered only when logbook runtime support is available.

## Source

[`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py).
