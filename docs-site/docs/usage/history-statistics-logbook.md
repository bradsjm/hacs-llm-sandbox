---
title: History, Statistics, And Logbook
description: Ask questions over recorder-backed state history, long-term statistics, and activity timeline entries.
---

# History, Statistics, And Logbook

Recorder-backed tools answer questions about what happened before the current snapshot.

## Choose the fewest rounds

For a direct history, statistics, or activity answer, use `get_history`, `get_statistics`, or `get_logbook` respectively. Run independent direct reads in parallel and scope them with selectors instead of making a discovery call first.

Use one `execute_home_code` call when recorder evidence must be combined with current states or registries, computed or compared across sources, used to decide a condition, or used to perform an action. Do not retrieve the same evidence through both paths.

## State history

Use history for state changes, time-in-state, transitions, and declarative analytics.

```text
How many times did the garage door open in the last 24 hours?
```

`get_history` can return raw rows, legacy aggregates, or analytics grouped by fields such as domain, area, device, floor, label, and time bucket.

## Long-term statistics

Use statistics for sensor values that Home Assistant records as statistics.

```text
Compare hourly living room temperature means over the last day.
```

`get_statistics` supports `5minute`, `hour`, and `day` periods and statistic fields such as `mean`, `min`, `max`, `state`, and `sum`.

## Logbook

Use logbook for activity timeline questions.

```text
What happened around the time the alarm armed last night?
```

`get_logbook` returns bounded activity entries when logbook runtime data is available.

Inside `execute_home_code`, `await hass.logbook(entity_ids=None, hours=None)` supports that composed path. It accepts at most 20 visible entities over a 24-hour maximum and returns at most the newest 200 chronological JSON-safe entries. It has no cursor and requires recorder plus logbook runtime support.

## Raw-page pagination

Raw `get_history`, `get_statistics`, and `get_logbook` pages are assembled newest-first and emitted in their normal ascending stream order. Each complete compact UTF-8 response, including its cursor metadata, normally fits within 16 KiB. Row ceilings remain emergency limits (1000 history/statistics rows and 200 logbook entries). Records are never split; if one record alone is larger, it is returned intact by itself and the existing cursor continues with older records. This standalone 16 KiB cursor-page limit does not apply to `hass.logbook`; that in-code helper is instead capped at 200 entries with no continuation.

## Source

The public recorder tools live in [`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py), with runtime query helpers in [`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py).
