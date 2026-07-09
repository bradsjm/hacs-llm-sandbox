---
title: Monty Execution
description: How execute_home_code normalizes and runs model-written Python.
---

# Monty Execution

`execute_home_code` runs short model-written Python in Monty. The execution path is implemented in [`llm_api/executor.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor.py).

## Pipeline

1. Validate code size.
2. Build a fresh full snapshot.
3. Build safe facades and runtime context.
4. Apply fail-open normalization passes.
5. Run Monty with resource limits and timeout.
6. Convert output, printed lines, actions, notes, and errors to JSON-compatible payloads.

## Forgiveness layer

The normalization pipeline handles common LLM mistakes such as harmless datetime imports, builtin-surface mismatches, missing `await`, and missing `result =` binding. Each pass is fail-open: if normalization fails, the original code continues to Monty so the natural error can surface.

## Error shape

Successful runs return `execution.status == "ok"`. Code failures return `code_error`; helper/facade failures return `helper_error`; setup failures return `setup_error`.
