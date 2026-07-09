---
title: History, Statistics, And Logbook
description: Ask questions over recorder-backed state history, long-term statistics, and activity timeline entries.
---

# History, Statistics, And Logbook

Recorder-backed tools answer questions about what happened before the current snapshot.

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

## Source

The public recorder tools live in [`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py), with runtime query helpers in [`tools/_recorder_runtime.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/_recorder_runtime.py).
