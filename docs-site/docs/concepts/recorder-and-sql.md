---
title: Recorder And SQL
description: Recorder-backed tools and the read-only in-memory SQL surface.
---

# Recorder And SQL

Assist Agent Sandbox provides two ways to ask historical questions:

- Public recorder tools: `get_history`, `get_statistics`, and `get_logbook`.
- Composition helpers inside `execute_home_code`: `await hass.history(...)`, `await hass.logbook(...)`, and `await hass.query(sql, ...)`.

Use the matching public tool for a direct retrieval or summary of one recorder data type. Use one code call when recorder data must be joined with current state or registries, computed or compared across sources, used in a condition, or used to decide or perform an action. Independent direct recorder reads can run in parallel.

## SQL is not the live recorder database

`hass.query` runs against a per-run in-memory SQLite database populated from visible snapshot states and bounded recorder rows. It does not expose Home Assistant's live recorder database.

The SQL implementation is in [`llm_api/data/home_db.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/home_db.py).

## Scoping

History and statistics can be narrowed with entity IDs or Home Assistant selectors such as `area_id`, `device_id`, `floor_id`, `label_id`, and `domain`. Use selectors for direct tool reads instead of a separate discovery call. Selector resolution uses snapshot indexes, so invisible entities are not silently included.

`hass.logbook(entity_ids=None, hours=None)` is limited to 20 visible entities, 24 hours, and the newest 200 chronological entries. It is uncursored and requires recorder plus logbook runtime support.

## Read-only guard

The in-memory SQL path allows read-only operations and approved pragmas. It is intended for analysis, not mutation.
