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

Eval runs write artifacts under `eval_data/runs/<run_id>/`. That output directory is gitignored.
Every run creates an atomic `manifest.json` before model calls. Completed runs
contain native `report.json` and `report.html`; cancelled or failed runs contain
the typed `partial.json` journal instead. A partial journal is explicitly not a
report, cannot be rendered as HTML, and cannot be resumed.

The harness uses scoring v6. It preserves provider `model_id` and records the
resolved run-wide reasoning effort and temperature separately, deriving labels
such as `model(high)` for presentation. `quality_rate` is correct/scored and
`coverage_rate` is scored/total; incomplete cells carry an operational cause,
not an action mismatch.

TTY runs render one transient Rich view and a durable stderr final with the
artifact location once. Redirected runs, or `--machine`, emit stable KV on
stdout; failed and cancelled runs leave stdout empty. Lanes show request,
variant, elapsed/timeout, and tools/cap only—there is no phase, Activity, or
`Waiting` column.

## Purpose

Use the harness to compare candidate prompts, models, and tool behavior against repeatable fixtures without requiring a live Home Assistant instance.
