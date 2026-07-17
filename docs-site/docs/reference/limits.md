---
title: Limits
description: Runtime, recorder, SQL, service-call, and image limits.
---

# Limits

The current limits are defined mainly in [`const.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/const.py), [`config_flow.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/config_flow.py), [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py), and [`llm_api/data/home_db.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/data/home_db.py).

| Area | Limit |
| --- | --- |
| Monty code length | 8000 characters. |
| Execution timeout option | 3 to 30 seconds, default 12. |
| Service-call limit option | 1 to 100 validated calls dispatched to Home Assistant, default 32. |
| Raw history lookback | Up to 24 hours. |
| History aggregate lookback | Up to 30 days. |
| Statistics lookback | Up to 30 days. |
| Raw recorder response | Compact UTF-8 JSON is normally limited to 16 KiB, including the window and continuation metadata. |
| History rows | Emergency ceiling of 1000 rows per raw page. |
| Statistics rows | Emergency ceiling of 1000 rows per raw page. |
| Standalone logbook entries | Emergency ceiling of 200 entries per raw 16 KiB cursor page. |
| `hass.logbook` | At most 20 visible entities, a 24-hour window, and the newest 200 chronological entries; no cursor. |
| `hass.history` | At most 1000 raw rows per call; capped results report truncation in the top-level overflow field. |
| Explicit recorder entity IDs | Up to 20. |
| SQL length | Up to 4000 characters. |
| SQL result rows | Up to 500 rows. |
| Loaded history/statistics rows for SQL | Up to 20000 rows. |
| Camera target width | 384 to 1920 pixels, default 1280. |
| Image attachment size | Up to 5 MiB. |
| Energy source/device records | Up to 100 selected records per query; when validation is requested, the ceiling applies to all configured records. |
| Energy statistic IDs | Up to 40 per primary or comparison query. |
| Energy query points | Up to 12,000 per primary or comparison window. |
| Energy returned series points | Up to 500 historical series points across one response. |
| Energy forecast points | Up to 96 across at most 8 solar sources. |
| Energy response | Compact output is fitted to the 16 KiB recorder response budget, preserving reported totals. |

Only dispatched service calls consume the service-call limit, including failures and timeouts after dispatch; reads and pre-dispatch blocks do not. Raw standalone recorder pages preserve whole records and use their existing `next_cursor` to continue with older data. The `hass.logbook` helper is not a standalone page and does not use the 16 KiB cursor-page budget.
