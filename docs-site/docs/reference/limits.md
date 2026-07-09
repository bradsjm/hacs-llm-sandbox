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
| Service-call budget option | 1 to 100 calls, default 32. |
| Raw history lookback | Up to 24 hours. |
| History aggregate lookback | Up to 30 days. |
| Statistics lookback | Up to 30 days. |
| History rows | Up to 1000 returned rows. |
| Statistics rows | Up to 1000 returned rows. |
| Logbook entries | Up to 200 returned entries. |
| Explicit recorder entity IDs | Up to 20. |
| SQL length | Up to 4000 characters. |
| SQL result rows | Up to 500 rows. |
| Loaded history/statistics rows for SQL | Up to 20000 rows. |
| Camera target width | 512 to 1920 pixels, default 1280. |
| Image attachment size | Up to 5 MiB. |
