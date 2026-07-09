---
title: Guidance And Recovery
description: How errors and ambiguous entity references become actionable model feedback.
---

# Guidance And Recovery

The integration aims for success in one call and recovery in one follow-up call. It does that with normalization, structured error payloads, and entity guidance.

## Error payloads

[`llm_api/errors.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/errors.py) defines compact execution and tool error payloads. Recoverable errors carry stable keys, messages, and optional structured guidance.

## Code refinement

[`executor_support/refinement.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/executor_support/refinement.py) maps Monty and type-check failures into more familiar Python-style guidance where possible.

## Entity guidance

The guidance engine ranks candidates when a requested entity ID or service target does not resolve cleanly. Guidance can include confidence, candidate IDs, names, match reasons, and next steps.

## Optional memory

When guidance confidently resolves a model's mistaken entity literal, later code in the same conversation can reuse that remembered resolution if it still exists in the fresh snapshot.
