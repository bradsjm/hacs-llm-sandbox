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
# With --judge-model, judging is limited to the five opted-in complex-code cases.
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini --judge-model gpt-5.4
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id> --markdown
```

Eval runs write artifacts under `eval_data/runs/<run_id>/`. That output directory is gitignored.
For `eval --models`, `eval --judge-model`, and `optimize --cross-eval-models`,
bare Pydantic AI model IDs resolve to the `openai-chat` provider (for example,
`gpt-4o-mini` becomes `openai-chat:gpt-4o-mini`). `stub` and IDs containing `:`
are preserved for candidate models; the normalized `stub` is rejected for the
judge model. `--judge-model MODEL` is singular and optional.
Optimizer `--target-model` and `--proposer-model` remain DSPy/LiteLLM IDs.
Every run creates an atomic `manifest.json` before model calls. Completed runs
contain native `report.json`, `errors.log`, and `report.html`; cancelled or
failed runs contain the typed `partial.json` journal instead. A partial journal
is explicitly not a report, cannot be rendered as HTML, and cannot be resumed.
For completed reports, `errors.log` is UTF-8 NDJSON with one record per
incomplete execution error in report order, including repeated incidents and
full error/provider detail. It is zero bytes when no execution errors occur.
The shared completed-report writer atomically replaces `errors.log` before
atomically replacing `report.json`, so a newly completed `report.json` has its
companion log without making the two files a single transaction.

The harness uses scoring v9 over 17 canonical tasks in the `direct`,
`discovery`, `service_data`, `conditional`, `ambiguity`, `tool_contract`, and
`read_answer` categories. The declared oracle is `effect`, `tool_calls`, or
`answer`: effect scoring uses sparse final-state predicates with exact action
fallback, tool-contract scoring compares normalized successful tool events, and
read-answer scoring uses deterministic typed predicates. Version 8 and older
artifacts are rejected without compatibility.
It preserves provider `model_id` and records the resolved run-wide reasoning
effort and temperature separately, deriving labels such as `model(high)` for
presentation. `quality_rate` is correct/scored and `coverage_rate` is
scored/total; incomplete cells carry an operational cause, not an action
mismatch. Provider HTTP 429 responses and provider bodies containing
`token_quota_exceeded` classify as `rate_limit`; structured execution metadata
is additive and does not change scoring or action semantics.
Canonical quality is the primary leaderboard and includes Wilson 95% intervals
over scored canonical cells. Paraphrases remain distinct utterance-level cells,
while task robustness aggregates all request variants. HTML and deterministic
Markdown reports share the same immutable saved-report projection;
`report <run_id> --markdown` writes `report.md` without model calls.

TTY runs render one transient Rich view and a durable stderr final with the
artifact location once. Redirected runs, or `--machine`, emit stable KV on
stdout; failed and cancelled runs leave stdout empty. Every agent run consumes
Pydantic AI's native `run_stream_events`, with no streaming flag or
non-streaming fallback. Rich Activity is always visible from lane creation as
`queued`, is phase-colored, and remains payload-free. Judged lanes show
transient `judging` and remain active until judge termination. Narrow layouts
drop only `Variant`; Activity remains visible. There is no synthesized
`Waiting`.
Human live and persistent durable final output render `Operational issues` as a
full-width actionable table. Exact duplicate issues group for display with
affected cells, while `errors.log` remains per-trace and machine output remains
payload-free. After a completed interactive judged run, the durable Rich final
conditionally appends a separate `Code judge · advisory` panel sourced from the
completed native report. It shows judge model/rubric identity, overall
requested/available counts, pass rate, mean score, evaluator-failure and
unavailable counts, per-candidate/model-variant aggregates, and a bounded
five-item needs-attention preview with overflow. The preview uses fixed
classifications and only the safe evaluator error type; it never renders judge
reasons, provider messages, stacktraces, or request/code/tool payloads. Full
judge reasons remain in HTML/Markdown/report.json. The panel does not affect
deterministic quality, ranking, coverage, or verdict; machine KV remains
judge-free.

Activity labels are payload-free. Actual runtime `running` and `processing`
tool phases include the validated tool name; provider/model-supplied
`preparing` names are not retained or rendered. The transient phase/activity
channel is label/tool-name only and is not persisted in reports or artifacts.
Interactive Activity and machine output do not render reasoning content, model
responses, tool arguments, or tool results. Existing durable reports retain
their established `CaseTrace` answer and tool-diagnostics contract.

## Optional code-quality judge

Judging requires both gates: a separately authored case sets `judge_code: true`
and the run supplies `--judge-model MODEL`. The default is false. Five current
complex-code cases are opted in: `discover_utility_room_lights` (area discovery
and coordinated multi-target action), `discover_basement_ceiling_lights` (large
inventory area/name filtering and twelve targets),
`condition_history_change_turn_off` (history processing and conditional
action), `no_action_history_no_recent_change` (history processing and
conditional no-op), and `ambiguous_logic_living_room_recent` (comparing recent
histories across candidates). All other current cases omit `judge_code` and
remain false. When `--judge-model MODEL` is supplied, it invokes the judge only
for cells from those five selected cases; without the model, no cells are
judged. The case oracle does not matter to this opt-in; the judge remains an
advisory, oracle-agnostic assessment of `execute_home_code` efficiency and
appropriateness rather than final-answer or read-answer correctness.

The judge treats Monty as ephemeral request-scoped glue code. It prioritizes
effective task contribution, minimal model/tool round trips, scoped reads,
in-sandbox computation, and compact useful output rather than Ruff, formatting,
comments, abstractions, typing, tests, or maintainability. Its bounded context
contains the request, trusted deterministic outcome, every ordered code call
with execution status and bounded output/action/resolution/note evidence, and
compact summaries of relevant interleaved non-code tools. It excludes the
answer, expected evidence, and live objects. If complete code source or action
evidence cannot fit, no provider call is made and the result is unavailable
rather than partially judged. Zero-submission cells still receive one call.
The existing model timeout applies, with no judge retries or fallbacks.
Provider, validation, and timeout failures are native `EvaluatorFailure`
records and do not alter deterministic scoring.

Native outputs remain `code_quality_score` and `code_quality_pass`, using the
stable evaluator name `code_quality_judge`, rubric ID
`llm_sandbox_code_quality`, version `2`. The judge model and rubric identity
are persisted in descriptor/metadata. `CaseTrace` has no judge field and
remains scoring v9; judge results do not affect correctness, quality,
coverage, Wilson intervals, ranking, rescoring, `partial.json`, `errors.log`,
machine phase lines, or the deterministic CSV score. Cancellation during
judging emits no false `cell_finished` or partial record; ordinary judge
failure releases/completes the lane and persists the native evaluator failure.
Completed HTML and Markdown reports conditionally show separate advisory Code
Judge sections when judging was requested; full judge reasons remain available
there and in `report.json`. The interactive terminal preview is the surface
that uses fixed classifications and safe evaluator error types only.

## Purpose

Use the harness to compare candidate prompts, models, and tool behavior against repeatable fixtures without requiring a live Home Assistant instance.
