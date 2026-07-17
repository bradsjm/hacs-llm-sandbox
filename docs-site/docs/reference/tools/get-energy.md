---
title: get_energy
description: Reference for bounded Home Assistant Energy dashboard queries.
---

# `get_energy`

Reads dashboard-configured Energy data through a fresh, visibility-filtered snapshot. The same query contract is available inside `execute_home_code` as `await hass.energy(...)` for requests that combine Energy with current state, registries, computation, conditions, or actions.

The tool is registered only when recorder is available and the Energy dashboard is configured. If every configured source is outside the assistant's current visibility scope, the call returns the structured `no_visible_energy_sources` error.

## Inputs

- `hours`, or explicit `start` and `end` ISO datetimes. `hours` may use `end` as its anchor but cannot be combined with `start`.
- `period`: `auto`, `5minute`, `hour`, `day`, `week`, `month`, or `year`. `auto` chooses the finest period that fits both query and response point budgets.
- `source_types`: any of `grid`, `solar`, `battery`, `gas`, `water`, `device`, and `device_water`.
- `device_statistic_ids`: optional visible, dashboard-configured tracked-device sensors.
- `include`: any of `summary`, `series`, `current`, `forecast`, `carbon`, and `validation`. Requesting `series` also includes `summary`.
- `compare`: `previous` or `year_over_year`; comparison requires `summary`.

## Output

Successful output reports the effective query `window`, selected `period` and `scope`, plus the requested projections:

- source measures and whole-home electricity flow totals;
- tracked-device inclusive and child-subtracted exclusive usage;
- cost and compensation totals;
- current prices, normalized power rates, and state of charge;
- bounded historical series, solar forecasts, carbon data, validation issues, and comparisons.

Unaligned requests report both the effective bucket boundaries and the original requested window. Calendar periods use the configured Home Assistant time zone.

## Bounds and privacy

One primary or comparison query is limited to 100 selected Energy source/device records, 40 statistic IDs, and 12,000 queried points. When `validation` is requested, the 100-record ceiling applies to all configured Energy source/device records rather than only the selected scope. Returned historical series are limited to 500 points in total; forecasts are limited to 96 points across at most eight solar sources. Complete compact output is fitted to the integration's 16 KiB recorder response budget without changing reported totals.

Only entity-backed statistic, rate, price, state-of-charge, and tracked-device IDs present in the fresh visible snapshot are copied into the safe catalog. External statistic IDs, hidden entity IDs, raw cost-sensor mappings, and solar forecast config-entry IDs are never returned. Rejected references appear only as omission role/reason counts.

## Source

The LLM-facing tool and host adapters are implemented in [`llm_api/tools/energy.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/energy.py). Safe records, time-window alignment, flow calculation, comparisons, and response fitting live in [`llm_api/data/energy.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/energy.py).
