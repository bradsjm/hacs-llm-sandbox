---
title: SQL Schema
description: Tables and compatibility views exposed by hass.query.
---

# SQL Schema

`await hass.query(sql, ...)` runs against a per-run in-memory SQLite database. The implementation is in [`llm_api/data/home_db.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/home_db.py).

## Base tables

### `states`

Columns include `entity_id`, `domain`, `object_id`, `name`, `state`, `value`, `attributes`, `area_id`, `floor_id`, `device_id`, `platform`, `unique_id`, `last_changed`, `last_changed_ts`, `last_updated`, and `last_updated_ts`.

### `history`

Columns include `entity_id`, `domain`, `area_id`, `floor_id`, `device_id`, `when_iso`, `when_ts`, `state`, and `value`.

### `statistics`

Columns include `statistic_id`, `entity_id`, `when_iso`, `when_ts`, `mean`, `min`, `max`, `state`, and `sum`.

## Compatibility views

The in-memory database also exposes recorder-compatible views: `state_history`, `long_term_statistics`, `states_meta`, `statistics_meta`, and `statistics_short_term`.

## Limits

Queries are length-limited, read-only, row-limited, and populated with bounded recorder rows.
