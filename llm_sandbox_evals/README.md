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

## Commands

```
python -m llm_sandbox_evals eval [--models id,...] [--candidates id,...] [--cases id,...|category,...] [--runs-dir PATH]
python -m llm_sandbox_evals report <run_id> [--runs-dir PATH]
```

- `eval` runs the matrix and writes artifacts under `eval_data/runs/<run_id>/`.
- `report <run_id>` re-renders a saved run's leaderboard from its `run.json` without re-running.
- `--cases` accepts case ids **or** category names (`state_read`, `registry_read`, `recorder_read`, `action_allowed`, `action_blocked`, `complex`).
- Defaults: `--models stub`, `--candidates baseline`, all cases.

## Checks

```bash
scripts/check-evals      # ruff + mypy + offline stub eval
scripts/format-evals     # ruff format
```

The integration's `scripts/check` is unaffected by this package.

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

Only `baseline` ships (auto-built from production `prompts.py`). `load_candidates` rejects unknown ids. To evaluate an alternative prompt, add a `PromptCandidate` and expose it through `prompts.load_candidates`. The follow-up DSPy optimizer (see scope) will emit optimized candidates through this same seam.

## Artifacts

`eval_data/` is gitignored. Each run writes, under `eval_data/runs/<run_id>/`:

- `run.json` — run metadata and per-(candidate, model) scores (no API keys).
- `leaderboard.md` — candidate ranking + candidate-by-model matrix.
- `results.jsonl` — one line per case/candidate/model with scores and check outcomes.
- `failures.jsonl` — cases that scored `0.0` or failed a required gate.
- `traces/<case>.<model>.<candidate>.json` — full prompt, raw model output, parsed tool call, tool result, recorded actions, and per-check results.

## Scope

**In:** deterministic scoring, multi-model matrix, offline stub validation, real `execute_home_code` against frozen snapshots, fixture-backed recorder emulation, artifact reports.

**Out of scope (v1):** native provider function-calling, LLM-as-judge scoring, live Home Assistant or recorder DB, CI jobs that call paid models, and auto-editing production `prompts.py`.

**Deferred follow-up:** the DSPy/GEPA prompt optimizer. The `ModelAdapter` protocol and `PromptCandidate` seam are in place so an optimizer can propose candidates, cross-evaluate them through this same harness, and export the winner for human review — without auto-patching production prompts.
