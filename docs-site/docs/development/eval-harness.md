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
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini,stub
```

Eval runs write artifacts under `eval_data/runs/<run_id>/`. That output directory is gitignored.
For `eval --models` and `optimize --cross-eval-models`, bare Pydantic AI model
IDs resolve to the `openai-chat` provider (for example, `gpt-4o-mini` becomes
`openai-chat:gpt-4o-mini`). `stub` and IDs containing `:` are preserved.
Optimizer `--target-model` and `--proposer-model` remain DSPy/LiteLLM IDs.
Every run creates an atomic `manifest.json` before model calls. Completed runs
contain native `report.json` and `report.html`; cancelled or failed runs contain
the typed `partial.json` journal instead. A partial journal is explicitly not a
report, cannot be rendered as HTML, and cannot be resumed.

The harness uses scoring v7. It matches successful actions exactly first. If
exact matching leaves exactly one unmatched authored multi-target action, the
remaining successful concrete entity-ID calls may score as
`equivalent_target_partition` only when at least two calls form a complete,
disjoint, duplicate-free partition of the authored target set with matching
domain, service, and comparable service data. Missing, extra, duplicate,
wrong-service, and different-data calls remain failures; raw calls remain
diagnostics. Version 6 and older artifacts are rejected without compatibility.
It preserves provider `model_id` and records the resolved run-wide reasoning
effort and temperature separately, deriving labels such as `model(high)` for
presentation. `quality_rate` is correct/scored and `coverage_rate` is
scored/total; incomplete cells carry an operational cause, not an action
mismatch.

TTY runs render one transient Rich view and a durable stderr final with the
artifact location once. Redirected runs, or `--machine`, emit stable KV on
stdout; failed and cancelled runs leave stdout empty. Every agent run consumes
Pydantic AI's native `run_stream_events`, with no streaming flag or
non-streaming fallback. Lanes keep their existing five-column layout unless a
real `thinking` event is observed for an active lane; only then does a sticky,
structured Activity column appear for the run. Providers without `ThinkingPart`
keep the five-column layout, without synthesized reasoning or `Waiting`.

Activity labels are payload-free. Actual runtime `running` and `processing`
tool phases include the validated tool name; provider/model-supplied
`preparing` names are not retained or rendered. The transient phase/activity
channel is label/tool-name only and is not persisted in reports or artifacts.
Interactive Activity and machine output do not render reasoning content, model
responses, tool arguments, or tool results. Existing durable reports retain
their established `CaseTrace` answer and tool-diagnostics contract.

## Purpose

Use the harness to compare candidate prompts, models, and tool behavior against repeatable fixtures without requiring a live Home Assistant instance.
