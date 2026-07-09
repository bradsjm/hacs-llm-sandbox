---
title: Eval Harness
description: Development-only evaluation package for tool behavior.
---

# Eval Harness

The repository includes a development-only `llm_sandbox_evals` package for evaluating production tools against frozen `HomeSnapshot` fixtures. It is not part of the Home Assistant runtime integration.

The eval README is [`llm_sandbox_evals/README.md`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/llm_sandbox_evals/README.md).

## Common commands

```bash
scripts/setup-evals
scripts/check-evals
```

Eval runs write report artifacts under `eval_data/runs/<run_id>/`. That output directory is gitignored.

## Purpose

Use the harness to compare candidate prompts, models, and tool behavior against repeatable fixtures without requiring a live Home Assistant instance.
