# LLM Sandbox Evals

Development-only eval harness for the `llm_sandbox` Home Assistant integration. It runs the integration's real LLM tools against a set of **frozen Home Assistant fixtures**, scores each operation deterministically, and ranks **prompt candidates** across a **matrix of language models** (the production model is unknown).

This package is not part of the integration runtime. It never adds dependencies to `custom_components/` or `manifest.json`, and it never touches a live Home Assistant instance.

## Quick start (offline, no API key)

```bash
scripts/setup-evals                      # uv sync --group dev --group evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
```

The `stub` adapter is deterministic and keyless — it validates the full pipeline (prompt render -> tool call -> real execution -> scoring -> report) end to end. It prints the run directory and a leaderboard:

```
eval_data/runs/20260630-164326-318981

| Candidate | Mean | MinModel | state_read | registry_read | recorder_read | action_allowed | action_blocked | complex |
| --------- | ---- | -------- | ---------- | ------------- | ------------- | -------------- | -------------- | ------- |
| baseline  | 0.858| 0.858    | 1.000      | 1.000         | 0.889         | 0.500          | 1.000          | 0.750   |
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

Every candidate is evaluated against every model. The `Candidate x model means` table shows the matrix; candidates rank by mean score, tie-broken by the best **minimum-across-models** score (robustness to the worst model wins ties). A model call that fails (bad key, network) is captured as an error in that case's trace and scores `0.0` — it never aborts the run.

## Optimizing the prompt (DSPy COPRO)

The `optimize` command uses [DSPy](https://dspy.ai/)'s COPRO instruction optimizer to rewrite the `execute_home_code` instruction (`api_prompt`, seeded from production `BASE_API_PROMPT`) and **keeps the real harness as the metric**: COPRO proposes instruction variants, and each is scored by the existing pipeline (`parse_tool_call` → `run_tool` → `check_case` → `score_case`) against one target model. The winner is then cross-evaluated against the model matrix.

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
- `--breadth` / `--depth` — COPRO search breadth/depth (defaults 5/2). Cost scales as `breadth × depth × trainset`.
- `--cases` — case ids/categories used as the optimization trainset (keep small to bound cost).
- `--cross-eval-models` — models for the baseline-vs-optimized leaderboard.
- `--target-reasoning` — reasoning effort for the target model during DSPy scoring and the baseline/optimized eval (e.g. `none` to disable a reasoning model that defaults to high).
- `--proposer-reasoning` — reasoning effort for the proposer model during DSPy.
- `--reasoning` — reasoning effort forwarded to the cross-eval harness models.

It writes `optimized_candidate.json` + `optimized_prompt.md` and prints a baseline-vs-optimized summary plus the cross-eval run dir. **Production `prompts.py` is never auto-patched** — the optimized text is exported for human review. Re-evaluate a saved candidate against any models:

```bash
uv run --group dev --group evals python -m llm_sandbox_evals eval \
  --candidates baseline,optimized:eval_data/runs/<run_id>/optimized_candidate.json \
  --models openrouter/deepseek/deepseek-v4-flash,stub
```

## Commands

```
python -m llm_sandbox_evals eval [--models id,...] [--candidates id,...] [--cases id,...|category,...] [--concurrency N] [--reasoning LEVEL] [--runs-dir PATH]
python -m llm_sandbox_evals optimize --target-model ID [--proposer-model ID] [--breadth N] [--depth N] [--cases ...] [--cross-eval-models ...] [--target-reasoning LEVEL] [--proposer-reasoning LEVEL] [--reasoning LEVEL] [--runs-dir PATH]
python -m llm_sandbox_evals report <run_id> [--runs-dir PATH]
```

- `eval` runs the matrix and writes artifacts under `eval_data/runs/<run_id>/`.
- `optimize` runs DSPy COPRO against one target model and cross-evaluates the winner (see *Optimizing the prompt* above).
- `report <run_id>` re-renders a saved run's leaderboard from its `run.json` without re-running.
- `--cases` accepts case ids **or** category names (`state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`).
- `--candidates` accepts `baseline` and `optimized:<path>` (a saved `optimized_candidate.json`).
- `--reasoning LEVEL` forwards a reasoning effort (e.g. `medium`/`high`, or `none` to disable a reasoning model) to real models via litellm. `optimize` adds `--target-reasoning` and `--proposer-reasoning` to control the target and proposer models independently (e.g. `--target-reasoning none --proposer-reasoning high`).
- Defaults: `--models stub`, `--candidates baseline`, all cases.

## Checks

```bash
scripts/setup-evals        # uv sync --group dev --group evals
scripts/check-evals        # ruff + mypy + offline stub eval (no API key, no dspy call)
scripts/format-evals       # ruff format
scripts/optimize-evals     # DSPy COPRO run — needs API keys, costs model calls
```

The integration's `scripts/check` is unaffected by this package. `optimize-evals` is intentionally **not** part of `check-evals` (it requires a real model and spends budget).

## How scoring works

Each `(candidate, model, case)` produces a score in `[0.0, 1.0]`:

- **Required gates** (any failure caps the case at `0.0`): valid JSON tool call, correct tool name, expected execution status, no action when actions are disabled, no invisible target, recorder tool produced an `ok` result.
- **Optional checks** (contribute to the ratio): output contains/excludes the expected entities, actions match expectations, a bounded recorder window was supplied, concise output.
- Case score = `0.0` if any required gate fails, else `passed_optional / total_optional`.

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
),
```

Categories: `state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`. The `expected.tool_name` must match a production tool constant (`execute_home_code`, `get_history`, `get_statistics`, `get_logbook`). For recorder cases, include the resolved entity id in `user_request` (e.g. `"...(sensor.living_temp)..."`) since models do not receive an entity list in the prompt.

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
- `traces/<case>.<model>.<candidate>.json` — full prompt, raw model output, parsed tool call, tool result, recorded actions, and per-check results.

## Scope

**In:** deterministic scoring, multi-model matrix, offline stub validation, real `execute_home_code` against frozen snapshots, fixture-backed recorder emulation, artifact reports, and DSPy COPRO instruction optimization (export-only; never auto-patches production `prompts.py`).

**Out of scope:** native provider function-calling, LLM-as-judge scoring, live Home Assistant or recorder DB, CI jobs that call paid models, and auto-editing production `prompts.py`. GEPA/MIPROv2 (richer feedback-driven or joint demo+instruction search) are not yet wired; COPRO is the implemented optimizer.
