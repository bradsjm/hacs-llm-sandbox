# LLM Sandbox Evals

Development-only eval harness for the `llm_sandbox` Home Assistant integration. It runs a real `pydantic_ai.Agent` against **frozen Home Assistant fixtures**, executes the production `execute_home_code` / recorder tool cores, scores structured final outcomes with a global tool-call cutoff plus successful-call efficiency, and ranks **prompt candidates** across a **matrix of language models** through native `pydantic_evals` `Dataset` / `EvaluationReport` reporting.

This package is not part of the integration runtime. It never adds dependencies to `custom_components/` or `manifest.json`, and it never touches a live Home Assistant instance.

## Quick start (offline, no API key)

```bash
scripts/setup-evals                      # uv sync --group dev --group evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
```

The `stub` FunctionModel is deterministic and keyless — it validates the full pipeline (Pydantic AI agent -> production tool core -> terminal answer -> scoring -> native report) end to end. It prints the run directory and compact native analysis lines:

```
eval_data/runs/20260630-164326-318981

overall_mean: 0.858
baseline/stub: mean=0.858 tool_calls=1.000
```

Interactive runs also print the native `pydantic_evals` report summary on stderr. Every run writes `report.json` containing native analyses plus per-cell traces and auto-emits `report.html` for browser-based visual navigation; regenerate the HTML with `report <run_id> --html`.

## Running real models

Any Pydantic AI provider-prefixed model id works, such as `openai:gpt-4o-mini`, `anthropic:claude-haiku-4-5`, or `openrouter:openai/gpt-4o-mini`; API keys are read from the environment (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY`). A `.env` file in the repo root is auto-loaded by the CLI (shell exports take precedence over `.env`), and `.env` is gitignored.

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EOF
uv run --group dev --group evals python -m llm_sandbox_evals eval --models openai:gpt-4o-mini,anthropic:claude-haiku-4-5,stub
```

Every candidate is evaluated against every model. The native `Candidate x model means` analysis shows the matrix; the native `Candidate ranking` analysis ranks candidates by mean score with a small prompt-size tie-break. A model call that fails (bad key, bad model id, network, or per-generation timeout) is captured as a `model_error` trace, scores `0.0`, and is excluded from candidate/model mean-score denominators (it is reported separately as an `Incomplete cells` count) so a provider outage does not read as a candidate-quality failure. The provider exception type/message, timeout details, and cause chain are included in the trace.

## Optimizing the prompt (DSPy COPRO)

The `optimize` command uses [DSPy](https://dspy.ai/)'s COPRO instruction optimizer to rewrite the API instruction (`api_prompt`, seeded from the selected production prompt profile) and **keeps the real agent harness as the metric**: COPRO proposes instruction variants, and each is scored by `run_case(...)` against one target model using production tool calls and structured outcome checks. The winner is then optionally cross-evaluated against the model matrix.

`optimize` uses DSPy, whose `dspy.LM` path accepts litellm-style ids such as `openrouter/...`; `eval` uses Pydantic AI provider-prefixed ids such as `openrouter:...`.

```bash
uv run --group dev --group evals python -m llm_sandbox_evals optimize \
  --target-model openrouter/deepseek/deepseek-v4-flash \
  --breadth 5 --depth 2 \
  --cases state_read,registry_read,complex \
  --cross-eval-models openrouter/deepseek/deepseek-v4-flash,stub
```

Flags:
- `--target-model` — model to optimize against (required; must be a real model, not `stub`).
- `--proposer-model` — model COPRO uses to propose rewrites (defaults to the target model).
- `--breadth` / `--depth` — COPRO search breadth/depth (defaults 5/2). COPRO requires breadth greater than 1. Cost scales as `breadth × depth × trainset`.
- `--length-penalty` — penalty coefficient applied when selecting COPRO candidates to tie-break toward smaller prompts at equal quality (default: `0.02`; `0` disables the size tie-break).
- `--cases` — case ids/categories used as the optimization trainset (keep small to bound cost).
- `--cross-eval-models` — models for the baseline-vs-optimized leaderboard.
- `--prompt-profile PROFILE_ID` — selects one production prompt profile for the baseline candidate and runtime settings (default: `standard`). This is separate from `--candidates`, which selects eval prompt candidates.
- `--target-reasoning` — reasoning effort stored for DSPy target-model configuration; the eval harness itself uses `--reasoning` / `config.reasoning_effort` when running `run_case(...)`.
- `--proposer-reasoning` — reasoning effort for the proposer model during DSPy.
- `--reasoning` — reasoning effort forwarded to the cross-eval harness models.

It writes `optimized_candidate.json` + `optimized_prompt.md` and prints a baseline-vs-optimized summary plus the cross-eval run dir when requested. Cross-eval runs use the same native eval artifact: `report.json`. **Production `prompts.py` is never auto-patched** — the optimized text is exported for human review. Re-evaluate a saved candidate against any models:

```bash
uv run --group dev --group evals python -m llm_sandbox_evals eval \
  --prompt-profile standard \
  --candidates baseline,optimized:eval_data/runs/<run_id>/optimized_candidate.json \
  --models openrouter:deepseek/deepseek-v4-flash,stub
```

## Commands

```
python -m llm_sandbox_evals eval [--models id,...] [--candidates id,...] [--prompt-profile ID] [--cases id,...|category,...] [--concurrency N] [--max-tool-calls N] [--model-timeout SECONDS] [--reasoning LEVEL] [--logfire] [--runs-dir PATH]
python -m llm_sandbox_evals optimize --target-model ID [--proposer-model ID] [--prompt-profile ID] [--breadth N] [--depth N] [--length-penalty COEFF] [--cases ...] [--cross-eval-models ...] [--target-reasoning LEVEL] [--proposer-reasoning LEVEL] [--reasoning LEVEL] [--runs-dir PATH]
python -m llm_sandbox_evals report <run_id> [--html] [--runs-dir PATH]
```

- `eval` builds a native `pydantic_evals.Dataset`, runs the matrix, prints the native `EvaluationReport` summary, and writes `report.json` plus interactive `report.html` artifacts under `eval_data/runs/<run_id>/`.
- `optimize` runs DSPy COPRO against one target model and cross-evaluates the winner (see *Optimizing the prompt* above).
- `report <run_id>` re-renders saved native analyses and cell scores from `report.json` and makes no model calls; add `--html` to regenerate `report.html` only.
- `--cases` accepts case ids **or** category names (`state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`, `recovery`).
- `--candidates` accepts `baseline`, `profile:<id>` production profiles, and `optimized:<path>` (a saved `optimized_candidate.json`).
- `--prompt-profile PROFILE_ID` selects one production base prompt profile for the whole run (default: `standard`); it is not comma-separated and is separate from `--candidates`.
- `terse` and `minimal` are condensed production profiles for capability-vs-size analysis; compare them with `--candidates baseline,profile:terse,profile:minimal` or select one via `--prompt-profile`.
- `--reasoning LEVEL` forwards a reasoning effort (e.g. `medium`/`high`, or `none` to disable a reasoning model) to real models via Pydantic AI provider settings (OpenRouter/OpenAI reasoning effort). `optimize` adds `--target-reasoning` and `--proposer-reasoning` to control the target and proposer models independently (e.g. `--target-reasoning none --proposer-reasoning high`).
- `--model-timeout SECONDS` bounds one model generation before recording `model_error` (default `75`). Slow free models may need a higher value or lower `--concurrency`.
- `--logfire` enables optional native Pydantic Logfire instrumentation when `LOGFIRE_TOKEN` is available.
- Defaults: `--models stub`, `--candidates baseline`, `--prompt-profile standard`, all cases.

## Checks

```bash
scripts/setup-evals        # uv sync --group dev --group evals
scripts/check-evals        # ruff + mypy + eval pytest + offline stub eval (no API key, no dspy call)
scripts/format-evals       # ruff format
scripts/optimize-evals     # DSPy COPRO run — needs API keys, costs model calls
```

The integration's `scripts/check` is unaffected by this package. `optimize-evals` is intentionally **not** part of `check-evals` (it requires a real model and spends budget).

## How scoring works

Each `(candidate, model, case)` runs until the Pydantic AI agent emits a terminal natural-language answer with no tool calls, or until the global `config.max_tool_calls` cutoff is exceeded and the harness records `tool_calls_exceeded`. It then produces a score in `[0.0, 1.0]`:

- **Required outcome gates** (any failure caps the case at `0.0`): the case has a meaningful structured oracle; `provenance_values` appear in structured payloads when specified; each `tool_result_check_*` finds a successful, relevant tool result shape that is non-empty unless the expected outcome explicitly allows no data with `min_results: 0`; the final tool call did not fail; allowed actions match exact expected side effects, or blocked actions satisfy `blocked_outcome`; and tool calls stay within the global cap.
- **Successful-call efficiency:** successful cases score `1.0` at or below their tool-call par, then linearly decay to `0.5` at the `10`-call runaway cap. Par is derived from structured case requirements by default (`tool_result_checks` plus one action/blocked-action step) and can be overridden with `tool_call_par` when a case has a known better calibrated expectation.
- **Allowed actions are exact successful side effects:** split calls may satisfy the exact expected target union for the same domain/service/service-data effect, but supersets, duplicate successful calls, and unrelated successful side effects fail. Failed intermediate action attempts do not fail scoring by themselves; they only count against the global tool-call cap.
- **Blocked actions are structured side-effect checks:** they require no successful disallowed action, bounded blocked attempts, and expected error keys when an action is attempted. Sandbox/tool enforcement belongs in unit or integration tests, not LLM evals.
- **Answer text checks are diagnostic only:** `answer_evidence_present` and `answer_excludes_absent` are reported for analysis but do not determine the score.
- **Tool-call counts are reported and efficiency-scored:** every trace records `tool_call_count`; the default hard cutoff is `10` and remains a runaway stop, while below-cap calls affect successful-case score through `tool_call_efficiency`.

## Adding cases

Edit `data/cases.yaml`. It is a native `pydantic_evals.Dataset` file with one authored `Case` per eval case (`name` mirrors `inputs.id`); `cases.py` loads it with `Dataset.from_file()` and exposes the stable `CASES: list[EvalCase]` surface. Each case must be a realistic human Home Assistant task, not a tool-contract, sandbox-enforcement, or malformed-input test. Do not reintroduce removed tool-contract/recovery/malformed-input cases as LLM evals; they belong in unit/integration tests. Prompts should not ask for entity IDs, tool names, raw service names, selectors, or implementation details unless a real user would naturally ask that way. Each case references a fixture by `home` name and pins the action setting, the initiating context, and deterministic expectations:

```yaml
name: llm_sandbox_cases
cases:
- name: my_case
  inputs:
    id: my_case
    category: state_read
    home: home_default
    user_request: What is the living room temperature?
    actions_enabled: false
    llm_context:
      platform: test
      device_id: null
      language: en
    expected:
      answer_values:
      - '25.2'
      provenance_values:
      - sensor.living_temp
      tool_result_checks:
      - tool_name: execute_home_code
        entity_ids:
        - sensor.living_temp
        entry_values:
        - '25.2'
      answer_excludes: []
      actions: []
      max_tool_calls: 10
```

Categories: `state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`, `recovery`. New cases should use the current outcome fields:

- `answer_values` — diagnostic-only expected final-answer facts for report analysis. They do not make an oracle meaningful and never determine the score.
- `provenance_values` — entity IDs, default IDs, selector-expansion facts, or other hidden/tool-payload evidence that should not be required in final prose.
- `tool_result_checks` — structured evidence for `execute_home_code`, `get_history`, `get_statistics`, or `get_logbook`; `execute_home_code` checks may combine top-level `output` values across successful snippets, but never score `printed` lines or envelope metadata. Recorder/history/statistics/logbook checks prove one successful, relevant result shape that is non-empty unless the expected outcome explicitly allows no data with `min_results: 0`. Use `entry_values_by_entity` when a multi-entity result needs different expected values per entity.
- `blocked_outcome` — structured expectations for deliberately blocked actions: allowed error keys and maximum attempts.
- `actions` — exact allowed successful side effects, with split calls allowed only when they satisfy the expected target union and no extra or duplicate successful side effects occur.
- `tool_call_par` — optional successful-call efficiency par. Leave unset unless there is a concrete calibrated reason; the scorer derives a par from the case's structured requirements.
- `answer_excludes` and `max_tool_calls` — diagnostic final-answer exclusions and the hard call cap.

`expected_values` is a legacy diagnostic bucket; do not use it for new cases. Scoring also requires `execution_ok` and excludes provider/infra `model_error` cells from means while counting them as incomplete. Recorder evals may be solved by supported selectors through native function calling, but the prompt should only mention selectors when that is natural user phrasing.

Recorder eval execution calls the production `GetHistoryTool` / `GetStatisticsTool` / `GetLogbookTool.run_query(...)` cores with a fixture-backed `RecorderSource`; result envelopes are the production envelopes, including cursor and recoverable-error shapes.

## Adding fixtures

Add a module under `homes/` exposing `snapshot() -> HomeSnapshot` and `recorder() -> dict`, then register it in `homes/__init__.py`. Fixtures are **Python data modules** (there is no JSON deserializer for `HomeSnapshot`); model the helpers on the existing `home_default.py` / `home_minimal.py`, which mirror the production builder's effective-area rule and sorted tuple indexes. Keep fixtures frozen and deterministic.

`homes/home_real.py` is a data-driven fixture baked from a frozen snapshot of a real Home Assistant instance (`home_real_data.json`, Assist-exposed entities only), demonstrating the pattern at scale (3 floors, 19 areas, 24 entities). The real `execute_home_code` runs genuine facade queries against it.

## Adding prompt candidates

`baseline` (auto-built from production `prompts.py`), `profile:<id>` production profiles such as `profile:terse` and `profile:minimal`, and any `optimized:<path>` candidate (a saved `optimized_candidate.json`) are loadable via `--candidates`. `load_candidates` rejects unknown ids. To evaluate a hand-authored alternative prompt, add a `PromptCandidate` and expose it through `prompts.load_candidates`. The `optimize` command emits optimized candidates through this same seam.

## Artifacts

`eval_data/` is gitignored. Each eval run writes these artifacts under `eval_data/runs/<run_id>/`:

- `report.json` — native analyses from the `EvaluationReport`, run metadata, per-cell scores/checks, and outcome traces (`output`, `tool_call_count`, recorded actions, checks, and error label/detail). It does not store API keys.
- `report.html` — interactive self-contained dashboard for visual navigation of the candidate × model × case matrix. Open it in a browser; `eval` writes it automatically, and `python -m llm_sandbox_evals report <run_id> --html` regenerates it from `report.json` without model calls.

DSPy optimization runs write `optimized_candidate.json` and `optimized_prompt.md`; if `--cross-eval-models` is set, the cross-eval run directory contains its own `report.json` and `report.html`.

## Scope

**In:** deterministic structured-outcome scoring, successful-call efficiency scoring, tool-call count reporting, multi-model matrix, native `pydantic_evals` `Dataset` / `EvaluationReport` integration, optional Logfire export via `--logfire`, Pydantic AI agent tool-calling, offline FunctionModel stub validation, production `execute_home_code` and recorder cores against frozen snapshots, `report.json` artifacts, and DSPy COPRO instruction optimization (export-only; never auto-patches production `prompts.py`).

**Out of scope:** LLM-as-judge scoring, live Home Assistant or recorder DB, CI jobs that call paid models, mutable cross-turn fixture state, and auto-editing production `prompts.py`. GEPA/MIPROv2 (richer feedback-driven or joint demo+instruction search) are not yet wired; COPRO is the implemented optimizer.
