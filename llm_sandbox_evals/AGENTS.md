# LLM Sandbox Evals

## Project Identity

`llm_sandbox_evals/` is a **development-only** package at repo root that evaluates the `llm_sandbox` LLM tools (`execute_home_code`, `get_history`, `get_statistics`, `get_logbook`). It runs the **real** `async_execute_home_code` against **frozen** `HomeSnapshot` fixtures (hand-built in Python — no JSON deserializer exists), scores each operation deterministically, and ranks prompt candidates across a matrix of language models.

It is **not** part of the integration runtime. See `README.md` for usage.

## Tool Purpose and Alignment

`execute_home_code` should help an LLM complete the user's Home Assistant task, not force the LLM to write perfect Python.

Treat the submitted code as short-lived task glue: interpret reasonable intent, accept common LLM coding patterns, and prefer "do what the user likely meant" over strict rejection when it is safe to do so.

Design for success in one call, and recovery in no more than one follow-up call.

On success, return the useful result directly. On failure, return actionable feedback that tells the next LLM call exactly what went wrong, what names or APIs are available, and what concrete change is likely to work.

Do not require the LLM to learn integration-specific details when normal Home Assistant knowledge can be adapted safely inside the tool.

Prioritize improving accomdating reasonable intent over increasing prompting length.

## Non-Negotiables

- **Never** pass a live `HomeAssistant` object, live registries, event bus, auth, config, filesystem, network, or OS/process API into the tool runner. The only service seam is `RecordingInvoker` (`tools.py`), which records the `ProposedAction` and returns `None` — it never calls `hass.services.async_call`.
- Build a **fresh** `HomeSnapshot` per case evaluation; never cache or mutate fixtures.
- The recorder tools are **emulated** from fixture `recorder()` data, never a live database.
- Keep eval dependencies **isolated**: `litellm` and `dspy` live only in `[dependency-groups] evals`. Never add them to `[project] dependencies`, `manifest.json`, or any `custom_components/**` import. `custom_components/**` is read-only.
- Keep `scripts/check` (the integration check) untouched; this package has its own `scripts/*-evals`.
- The DSPy optimizer is dev-only. Keep `dspy` imports inside `optimize_dspy.py` and the lazy CLI optimize handler so eval/report/stub paths import without DSPy.
- No fallbacks unless explicitly approved.

## Commands

- Setup: `scripts/setup-evals` (`uv sync --group dev --group evals`)
- Check: `scripts/check-evals` (ruff + mypy + offline stub eval)
- Format: `scripts/format-evals`
- Run: `uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub`
- Optimize: `uv run --group dev --group evals python -m llm_sandbox_evals optimize --target-model <real-model>`
- Report: `uv run --group dev --group evals python -m llm_sandbox_evals report <run_id>`

Note: eval runs need **both** groups (`dev` provides `homeassistant`, `evals` provides `litellm`). Artifacts go to the gitignored `eval_data/runs/`.

## Architecture And Data Flow

The harness (`harness.run_matrix`) runs, per `(candidate, model, case)`:

```
snapshot = homes.get_home(case.home).snapshot()          # fresh frozen snapshot
prompt   = prompts.render_prompt(candidate, case, snapshot)
result   = await models.get_adapter(model).complete(model, prompt)
outcome  = await tools.run_tool(result.tool_call, case, snapshot)
checks   = scoring.check_case(case, result.tool_call, outcome, snapshot)
score    = scoring.score_case(checks)
```

The harness owns the snapshot lifecycle (build once per evaluation, pass to render/run/score). The real executor activates/clears its own runtime contextvars internally — the harness does **not** call `activate_runtime`/`clear_runtime`.

### Module map

- `schema.py` — **stable shared contracts** (`PromptCandidate`, `EvalCase`, `Expected`, `ExpectedAction`, `CaseContext`, `ModelResult`, `ToolOutcome`, `CheckResult`, `ToolSpec`). Do not rename fields without updating all consumers.
- `config.py` — `EvalConfig` + `load_config()` (defaults: `models=["stub"]`, `candidates=["baseline"]`).
- `cases.py` — `CASES: list[EvalCase]`, the predefined suite (simple -> complex, all categories).
- `homes/` — frozen fixture modules (`snapshot() -> HomeSnapshot`, `recorder() -> dict`) + `get_home(name)` registry.
- `prompts.py` — `baseline_candidate()` (from production builders), `load_candidates(ids)`, `render_prompt(...)`, `tool_specs(...)`. Reuses production `BASE_API_PROMPT` / `ACTIONS_*_PROMPT` / tool-description builders; derives the request-location section from the frozen snapshot.
- `models.py` — `ModelAdapter` protocol, `StubAdapter` (offline, deterministic), `LiteLLMAdapter` (any provider, lazy import), `get_adapter(id)`, `parse_tool_call(text)`.
- `tools.py` — `run_tool(tool_call, case, snapshot) -> ToolOutcome`. Real executor path + fixture-backed recorder emulators matching production response shapes + `RecordingInvoker`.
- `scoring.py` — `check_case(...)`, `score_case(...)`, `mean_score(...)`. Required gates + optional checks.
- `harness.py` — `run_matrix(config) -> RunResult`; `CaseTrace`, `CandidateModelScore`, `RunResult`.
- `reports.py` — `write_run(...)`, `render_leaderboard(...)`, `load_run_json(...)` (for `report`).
- `optimize_dspy.py` — DSPy COPRO prompt optimizer that exports optimized `PromptCandidate` artifacts and reuses the real harness metric path.
- `cli.py` / `__main__.py` — `eval`, `report`, and `optimize` subcommands.

### Key contracts

- Tool call dict shape: `{"tool_name": str, "tool_args": dict}`. Tool names are the production constants from `custom_components.llm_sandbox.const`.
- `execute_home_code` `tool_args` = `{"code": str}`; the result dict carries `execution.status` (`ok|code_error|helper_error|setup_error`), `output`, `printed`, `actions`.
- Recorder `ToolOutcome.result` matches production: history `"entities": {id: [rows]}`, statistics `"statistics": {id: [rows]}`, logbook `"entries": [rows]`. Empty/missing ids -> `{"status":"error","error":{"key":"invalid_tool_input"}}`; invisible ids -> `entity_not_visible`.
- `ToolOutcome.recorded_actions` for the execute path comes from `result["actions"]` (the facade's `ActionRecord` list), not the invoker's captured calls.

## How To Extend

- **Add a case:** append to `cases.CASES`. Reference a real fixture `home`; keep `expected` deterministic. For recorder cases, put the resolved entity id in `user_request` (models get no entity list).
- **Add a fixture:** add `homes/<name>.py` with `snapshot()`/`recorder()` and `NAME`, then register in `homes/__init__.py`. Mirror `home_default.py`'s helpers (effective-area rule, sorted tuple `SnapshotIndexes`, nested `SafeContext`).
- **Add a candidate:** add a `PromptCandidate` and expose via `prompts.load_candidates`. `baseline` is auto-built; unknown ids currently raise.
- **Add a model:** no code needed — pass any litellm id to `--models`. To add a non-litellm backend, implement the `ModelAdapter` protocol and branch in `get_adapter`.

## The Stub Adapter

`StubAdapter` is a **pipeline validator**, not a scoring benchmark. It keyword-detects the tool from the **user request only** (not the whole prompt, which lists every tool name) and emits runnable, minimally-valid calls. Recorder cases score meaningfully only when the entity id appears in the request. A low stub score in a category is expected and honest; use real models for prompt-quality signal.

## Safety Verification

When changing `tools.py` or `homes/`, confirm: no `HomeAssistant` instantiation, no `hass.services.async_call`, no recorder DB imports, no `subprocess`/network/OS APIs. The only live seam must be `RecordingInvoker`. Run `scripts/check-evals` and the integration `scripts/check` (must stay green and must show no `custom_components/` changes).

## Code Style

- Python >=3.14.2, ruff `py314`, `line-length=119`, mypy strict. Concrete annotations; `from datetime import UTC, datetime` (use `UTC`, not `timezone.utc`).
- Comments at branch boundaries and safety constraints. Type-annotate all helper params.
- Keep `__init__.py` a stable surface. KISS/YAGNI.
- Do not write tests that pass only because mocks return expected values; assert observable behavior. No regression tests unless requested.
