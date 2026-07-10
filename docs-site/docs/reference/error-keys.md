---
title: Error Keys
description: Stable error categories exposed by tools and execution payloads.
---

# Error Keys

Errors are intended to be recoverable by the model. They usually include a stable key, a concise message, and sometimes structured guidance.

## Execution statuses

| Status | Meaning |
| --- | --- |
| `ok` | Code ran successfully, though individual service actions can still be blocked and recorded. |
| `setup_error` | The tool could not start, often because the integration entry is unavailable. |
| `code_error` | Monty could not run the submitted code. |
| `helper_error` | A facade operation failed during execution. |

## Common tool keys

Stable keys include `invalid_tool_input`, setup-related keys, selector resolution keys, target visibility keys, recorder availability keys, and vision capture keys. The exact mappings live in [`llm_api/errors.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/errors.py), [`tools/recorder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/recorder.py), and [`tools/vision.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/vision.py).

## Guidance object

When available, guidance can include confidence, candidates, reason, next step, and cross-kind details. Use it as the model-facing remediation surface rather than parsing prose.
