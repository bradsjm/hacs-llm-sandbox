# LLM Sandbox Evals

Development-only eval harness for the `llm_sandbox` Home Assistant integration. It runs a bounded, multi-turn native tool-calling agent loop against **frozen Home Assistant fixtures**, scores final task outcomes plus turn efficiency, and ranks **prompt candidates** across a **matrix of language models** (the production model is unknown).

This package is not part of the integration runtime. It never adds dependencies to `custom_components/` or `manifest.json`, and it never touches a live Home Assistant instance.

## Quick start (offline, no API key)

```bash
scripts/setup-evals                      # uv sync --group dev --group evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
```

The `stub` adapter is deterministic and keyless — it validates the full pipeline (message render -> native tool call -> tool result -> terminal answer -> scoring -> report) end to end. It prints the run directory and a leaderboard:

```
eval_data/runs/20260630-164326-318981

| Candidate | Mean | MinModel | Turns | Eff | state_read | registry_read | recorder_read | action_allowed | action_blocked | complex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.858 | 0.858 | 1.000 | 0.858 | 1.000 | 1.000 | 0.889 | 0.500 | 1.000 | 0.750 |
```

## Running real models

Any [LiteLLM](https://github.com/BerriAI/litellm) model id works; API keys are read from the environment (e.g. `OPENAI_API_KEY`). A `.env` file in the repo root is auto-loaded by the CLI (shell exports take precedence over `.env`), and `.env` is gitignored.

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
EOF
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini,claude-haiku-4-5,stub
```

Every candidate is evaluated against every model. The `Candidate x model means` table shows the matrix; candidates rank by mean score, tie-broken by the best **minimum-across-models** score (robustness to the worst model wins ties). A model call that fails (bad key, bad model id, network, or per-generation timeout) is captured as a `model_error` trace and scores `0.0`; the provider exception type/message, common LiteLLM metadata, response status/body when available, timeout details, and cause chain are printed to stderr. Remaining cases for that candidate/model pair are marked the same way without continuing to call the failing provider.

## Optimizing the prompt (DSPy COPRO)

The `optimize` command uses [DSPy](https://dspy.ai/)'s COPRO instruction optimizer to rewrite the API instruction (`api_prompt`, seeded from the selected production prompt profile) and **keeps the real multi-turn harness as the metric**: COPRO proposes instruction variants, and each is scored by `run_case(...)` against one target model using native tool calls, fixture tool results, final-answer outcome checks, and turn-efficiency scoring. The winner is then optionally cross-evaluated against the model matrix.

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
- `--cases` — case ids/categories used as the optimization trainset (keep small to bound cost).
- `--cross-eval-models` — models for the baseline-vs-optimized leaderboard.
- `--prompt-profile PROFILE_ID` — selects one production prompt profile for the baseline candidate and runtime settings (default: `standard`). This is separate from `--candidates`, which selects eval prompt candidates.
- `--target-reasoning` — reasoning effort for the target model during DSPy scoring and the baseline/optimized eval (e.g. `none` to disable a reasoning model that defaults to high).
- `--proposer-reasoning` — reasoning effort for the proposer model during DSPy.
- `--reasoning` — reasoning effort forwarded to the cross-eval harness models.

It writes `optimized_candidate.json` + `optimized_prompt.md` and prints a baseline-vs-optimized summary plus the cross-eval run dir. **Production `prompts.py` is never auto-patched** — the optimized text is exported for human review. Re-evaluate a saved candidate against any models:

```bash
uv run --group dev --group evals python -m llm_sandbox_evals eval \
  --prompt-profile standard \
  --candidates baseline,optimized:eval_data/runs/<run_id>/optimized_candidate.json \
  --models openrouter/deepseek/deepseek-v4-flash,stub
```

## Commands

```
python -m llm_sandbox_evals eval [--models id,...] [--candidates id,...] [--prompt-profile ID] [--cases id,...|category,...] [--concurrency N] [--model-timeout SECONDS] [--reasoning LEVEL] [--runs-dir PATH]
python -m llm_sandbox_evals optimize --target-model ID [--proposer-model ID] [--prompt-profile ID] [--breadth N] [--depth N] [--cases ...] [--cross-eval-models ...] [--target-reasoning LEVEL] [--proposer-reasoning LEVEL] [--reasoning LEVEL] [--runs-dir PATH]
python -m llm_sandbox_evals report <run_id> [--runs-dir PATH]
```

- `eval` runs the matrix and writes artifacts under `eval_data/runs/<run_id>/`.
- `optimize` runs DSPy COPRO against one target model and cross-evaluates the winner (see *Optimizing the prompt* above).
- `report <run_id>` re-renders a saved run's leaderboard from its `run.json` without re-running.
- `--cases` accepts case ids **or** category names (`state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`).
- `--candidates` accepts `baseline` and `optimized:<path>` (a saved `optimized_candidate.json`).
- `--prompt-profile PROFILE_ID` selects one production base prompt profile for the whole run (default: `standard`); it is not comma-separated and is separate from `--candidates`.
- `--reasoning LEVEL` forwards a reasoning effort (e.g. `medium`/`high`, or `none` to disable a reasoning model) to real models via litellm. `optimize` adds `--target-reasoning` and `--proposer-reasoning` to control the target and proposer models independently (e.g. `--target-reasoning none --proposer-reasoning high`).
- `--model-timeout SECONDS` bounds one model generation before recording `model_error` (default `75`). Slow free models may need a higher value or lower `--concurrency`.
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

Each `(candidate, model, case)` runs until the assistant emits a terminal natural-language answer with no tool calls, or until `case.max_turns or config.max_turns` is reached and the harness forces a final answer. It then produces a score in `[0.0, 1.0]`:

- **Required outcome gates** (any failure caps the case at `0.0`): expected tools were used, required multi-tool sequences were followed, recorder windows covered the requested period, intermediate tool evidence was present/absent, expected final-answer entity references are present/absent for read/report cases, expected actions were recorded, disabled-action cases did not execute actions, expected execution status was observed, and invisible targets were not referenced.
- **Efficiency** applies only after required gates pass: `1.0` when tool-turns are at or below `par_turns`, otherwise `max(efficiency_floor, 1 - efficiency_k * (turns - par_turns))`.
- Default efficiency settings are `efficiency_k=0.25`, `efficiency_floor=0.1`, `max_turns=5`.

## Adding cases

Edit `cases.py` and append to `CASES: list[EvalCase]`. Each case references a fixture by `home` name and pins the action setting, the initiating context, and deterministic expectations:

```python
EvalCase(
    id="my_case",
    category="state_read",
    home="home_default",
    user_request="What is the living room temperature?",
    actions_enabled=False,
    llm_context=CaseContext(),
    expected=Expected(
        tool_name="execute_home_code",
        output_contains_entities=("sensor.living_temp",),
    ),
    par_turns=1,
),
```

Categories: `state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`, `recovery`. The `expected.tool_name` must match a production tool constant (`execute_home_code`, `get_history`, `get_statistics`, `get_logbook`) and is enforced as the primary expected tool. Use `required_tool_names` and `required_tool_sequence` for multi-tool cases, `recorder_window` for bounded recorder coverage, `required_error_keys` and `required_result_paths` for recovery metadata, `max_tool_turns` / `max_successful_actions` for no-retry gates, and `evidence_contains_entities` / `evidence_excludes_entities` for tool-call/tool-result evidence. Final-answer entity checks are for read/report cases; action cases may finish with a simple acknowledgement and should be scored through recorded actions plus intermediate evidence. Set `par_turns` to the efficient tool-turn target for the case. Recorder cases can be solved with explicit ids or supported selectors (`entity_ids`/`statistic_ids`, `area_id`, `device_id`, `floor_id`, `label_id`, `domain`, bounded time window args) through native function calling.

Recorder emulator results mirror production payload shapes: history returns `{"window": {...}, "entities": {"sensor.id": {"unit": "°C", "rows": [[t, state]]}}}`, statistics returns `{"window": {...}, "period": "hour", "statistics": {"sensor.id": {"fields": ["sum"], "rows": [[t, {"sum": value}]]}}}` with optional `types` selecting statistic fields, and logbook returns `{"window": {...}, "entries": [{"entity_id": "light.id", "when": t, "name": name, "message": message}]}`. Successful recorder results omit `status`; `next_cursor` appears only when more rows remain; recorder errors are `{"status":"error","error":{"key": str, "message": str, "fix"?: list[str]}}`.

## Adding fixtures

Add a module under `homes/` exposing `snapshot() -> HomeSnapshot` and `recorder() -> dict`, then register it in `homes/__init__.py`. Fixtures are **Python data modules** (there is no JSON deserializer for `HomeSnapshot`); model the helpers on the existing `home_default.py` / `home_minimal.py`, which mirror the production builder's effective-area rule and sorted tuple indexes. Keep fixtures frozen and deterministic.

`homes/home_real.py` is a data-driven fixture baked from a frozen snapshot of a real Home Assistant instance (`home_real_data.json`, Assist-exposed entities only), demonstrating the pattern at scale (3 floors, 19 areas, 24 entities). The real `execute_home_code` runs genuine facade queries against it.

## Adding prompt candidates

`baseline` (auto-built from production `prompts.py`) and any `optimized:<path>` candidate (a saved `optimized_candidate.json`) are loadable via `--candidates`. `load_candidates` rejects unknown ids. To evaluate a hand-authored alternative prompt, add a `PromptCandidate` and expose it through `prompts.load_candidates`. The `optimize` command emits optimized candidates through this same seam.

## Artifacts

`eval_data/` is gitignored. Each run writes, under `eval_data/runs/<run_id>/`:

- `run.json` — run metadata and per-(candidate, model) scores (no API keys).
- `leaderboard.md` — candidate ranking + candidate-by-model matrix.
- `results.jsonl` — one line per case/candidate/model with scores and check outcomes.
- `failures.jsonl` — cases that scored `0.0` or failed a required gate.
- `traces/<case>.<model>.<candidate>.json` — full messages, raw model output, final answer, turns/par/efficiency, per-turn tool calls/results, recorded actions, and per-check results.

## Scope

**In:** deterministic outcome + efficiency scoring, multi-model matrix, native provider tool-calling, offline stub validation, real `execute_home_code` against frozen snapshots, fixture-backed recorder emulation, artifact reports, and DSPy COPRO instruction optimization (export-only; never auto-patches production `prompts.py`).

**Out of scope:** LLM-as-judge scoring, live Home Assistant or recorder DB, CI jobs that call paid models, mutable cross-turn fixture state, and auto-editing production `prompts.py`. GEPA/MIPROv2 (richer feedback-driven or joint demo+instruction search) are not yet wired; COPRO is the implemented optimizer.
