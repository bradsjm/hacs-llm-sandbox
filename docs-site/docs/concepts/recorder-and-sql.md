---
title: Recorder And SQL
description: Recorder-backed tools and the read-only in-memory SQL surface.
---

# Recorder And SQL

Assist Agent Sandbox provides two ways to ask historical questions:

- Public recorder tools: `get_history`, `get_statistics`, and `get_logbook`.
- `await hass.query(sql, ...)` inside `execute_home_code`.

## SQL is not the live recorder database

`hass.query` runs against a per-run in-memory SQLite database populated from visible snapshot states and bounded recorder rows. It does not expose Home Assistant's live recorder database.

The SQL implementation is in [`llm_api/data/home_db.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/home_db.py).

## Scoping

History and statistics can be narrowed with entity IDs or Home Assistant selectors such as `area_id`, `device_id`, `floor_id`, `label_id`, and `domain`. Selector resolution uses snapshot indexes, so invisible entities are not silently included.

## Read-only guard

The in-memory SQL path allows read-only operations and approved pragmas. It is intended for analysis, not mutation.
