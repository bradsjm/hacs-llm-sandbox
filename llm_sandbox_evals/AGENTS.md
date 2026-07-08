# LLM Sandbox Evals

## Project Identity

`llm_sandbox_evals/` is a **development-only** package at repo root that evaluates the `llm_sandbox` LLM tools (`execute_home_code`, `get_history`, `get_statistics`, `get_logbook`). It runs a real `pydantic_ai.Agent` against **frozen** `HomeSnapshot` fixtures (hand-built in Python — no JSON deserializer exists), executes the production `run_execute` / `run_query` tool cores, scores final outcomes with tool-call efficiency, and ranks prompt candidates across a matrix of language models through native `pydantic_evals` `Dataset` / `EvaluationReport` reporting.

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
- The recorder tools run production `run_query(...)` cores against fixture-backed `RecorderSource` data, never a live database.
- Keep eval dependencies **isolated**: `pydantic-ai-slim[...]`, `pydantic-evals[logfire]`, `dspy`, and `rich` live only in `[dependency-groups] evals`. `litellm` remains only as a transitive dep of `dspy`; the eval-matrix adapter never imports it. Never add them to `[project] dependencies`, `manifest.json`, or any `custom_components/**` import. `custom_components/**` is read-only.
- Keep `scripts/check` (the integration check) untouched; this package has its own `scripts/*-evals`.
- The DSPy optimizer is dev-only. Keep `dspy` imports inside `optimize_dspy.py` and the lazy CLI optimize handler so eval/report/stub paths import without DSPy.
- No fallbacks unless explicitly approved.

## Commands

- Setup: `scripts/setup-evals` (`uv sync --group dev --group evals`)
- Check: `scripts/check-evals` (ruff + mypy + offline stub eval)
- Format: `scripts/format-evals`
- Run: `uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub --prompt-profile standard`
- Optimize: `uv run --group dev --group evals python -m llm_sandbox_evals optimize --target-model <real-model>`
- Report: `uv run --group dev --group evals python -m llm_sandbox_evals report <run_id>`

Note: eval runs need **both** groups (`dev` provides `homeassistant`, `evals` provides `pydantic-ai-slim` and `pydantic-evals`). Artifacts go to the gitignored `eval_data/runs/`; each native eval run writes `report.json`. Pass `--logfire` to enable optional Pydantic Logfire export when `LOGFIRE_TOKEN` is available.

## Architecture And Data Flow

The native experiment builds a `pydantic_evals.Dataset` with one case per `(candidate, model, case)` matrix cell, then evaluates each cell through `harness.run_case`:

```
profile  = resolve_profile(config.prompt_profile)         # selected production prompt profile
snapshot = apply_scope(homes.get_home(case.home).snapshot(), EVAL_SCOPE)
runtime  = build_eval_runtime(case, candidate, profile, snapshot, fixture)
agent    = build_agent(runtime, model_id)                 # production schemas via convert(tool.parameters)
result   = await agent.run(case.user_request, deps=runtime, usage_limits=UsageLimits(tool_calls_limit=...))
checks   = scoring.check_case(case, result.output, recorded_actions, tool_call_count)
score    = scoring.score_case(checks)
report = dataset.evaluate(task, name="matrix", max_concurrency=config.concurrency)
```

The harness owns the snapshot lifecycle and builds a fresh scoped snapshot per case. Correct outcomes score required gates plus optional `reference_tool_calls / actual_tool_calls`; failed required outcome gates score `0.0`. The native `EvaluationReport` carries the per-cell scores, deterministic check labels, candidate ranking, candidate x model means, and overall mean. The real executor activates/clears its own runtime contextvars internally — the harness does **not** call `activate_runtime`/`clear_runtime`.

### Module map

- `schema.py` — **stable shared contracts** (`PromptCandidate`, `EvalCase`, `Expected`, `ExpectedAction`, `CaseContext`, `CheckResult`, `CaseTrace`). Do not rename fields without updating all consumers.
- `config.py` — `EvalConfig` + `load_config()` (defaults: `models=["stub"]`, `candidates=["baseline"]`, `max_tool_calls=8`).
- `data/cases.yaml` — native `pydantic_evals.Dataset` authoring file for the predefined suite (simple -> complex, all categories), with `data/cases_schema.json` generated by `Dataset.to_file()`.
- `cases.py` — loads `data/cases.yaml` with `Dataset.from_file()` and exposes stable `CASES: list[EvalCase]`.
- `homes/` — frozen fixture modules (`snapshot() -> HomeSnapshot`, `recorder() -> dict`) + `get_home(name)` registry.
- `prompts.py` — `baseline_candidate()` (from production builders), `load_candidates(ids, prompt_profile_id)`, and prompt-size helpers. `terse`/`minimal` are condensed production profiles for size-axis evals.
- `agent_runner.py` — Pydantic AI `Agent`/`Tool.from_schema` wiring, production schema conversion, real-model inference, reasoning settings, and the offline `FunctionModel` stub.
- `runtime.py` — `EvalRuntime`, fixture-backed `RecorderSource`, runtime context factory, and SQL/history/statistics fixture seams for `execute_home_code`.
- `tools.py` — `EVAL_SCOPE`, `apply_scope`, `RecordingInvoker`, and action normalization helpers only; no tool emulators.
- `scoring.py` — `check_case(...)`, `score_case(...)`, `mean_score(...)`. Outcome gates + turn-efficiency scoring.
- `harness.py` — `run_case(...) -> CaseTrace`; the bounded per-cell task body reused by native experiments and DSPy.
- `experiment.py` — native `pydantic_evals` `Dataset` construction, deterministic `SandboxOutcome` evaluator, report-level candidate/model analyses, and `run_matrix(...) -> EvaluationReport`.
- `reports.py` — `write_report_json(...)`, `load_report_payload(...)`, and `render_report_summary(...)` for the single saved `report.json` artifact.
- `logfire_config.py` — optional Pydantic Logfire configuration used only when `eval --logfire` is passed.
- `optimize_dspy.py` — DSPy COPRO prompt optimizer that exports optimized `PromptCandidate` artifacts and reuses the real harness metric path.
- `cli.py` / `__main__.py` — `eval`, `report`, and `optimize` subcommands.

### Key contracts

- Tool schemas are `convert(tool.parameters)` from production tool instances. Tool execution returns production result envelopes directly; recoverable errors are not raised as `ModelRetry`.
- `execute_home_code` `tool_args` = `{"code": str}`; the result dict carries `execution.status` (`ok|code_error|helper_error|setup_error`), `output`, and optional `printed`, `actions`, `note`, and `fix` fields on the relevant success/error payloads.
- Recorder `ToolOutcome.result` matches production: history `{"window": {...}, "entities": {id: {"unit"?: str, "rows": [[t, state]]}}}`, statistics `{"window": {...}, "period": str, "statistics": {id: {"fields": [field], "rows": [[t, {field: value}]]}}}` with optional `types` selecting statistic fields, and logbook `{"window": {...}, "entries": [{"entity_id", "when", "name", "message", ...}]}`. Success omits `status`; `next_cursor` appears only when more rows remain. Errors are `{"status":"error","error":{"key": str, "message": str, "fix"?: list[str]}}`, including `entity_not_visible` with concrete visible candidates when available.
- `ToolOutcome.recorded_actions` for the execute path comes from `result["actions"]` (the facade's `ActionRecord` list), not the invoker's captured calls.

## How To Extend

- **Add a case:** append one native Dataset `Case` to `data/cases.yaml` with `name` matching `inputs.id`; do not hand-parse the file in code. Reference a real fixture `home`; keep `expected` outcome-only (`answer_facts`, `answer_excludes`, `actions`, `max_tool_calls`, `reference_tool_calls`).
- **Add a fixture:** add `homes/<name>.py` with `snapshot()`/`recorder()` and `NAME`, then register in `homes/__init__.py`. Mirror `home_default.py`'s helpers (effective-area rule, sorted tuple `SnapshotIndexes`, nested `SafeContext`).
- **Add a candidate:** add a `PromptCandidate` and expose via `prompts.load_candidates`. `baseline` is auto-built; unknown ids currently raise.
- **Add a model:** no code needed — pass any Pydantic AI provider-prefixed id (`openai:...`, `anthropic:...`, `openrouter:...`) to `--models`.

## The Stub Adapter

The `stub` FunctionModel is a **pipeline validator**, not a scoring benchmark. It keyword-detects the tool from the **user request only** (not the whole prompt, which lists every tool name), emits runnable Pydantic AI `ToolCallPart`s, then returns a terminal answer echoing the latest tool result. Recorder cases without explicit ids use broad selectors to exercise resolver support. A low stub score in a category is expected and honest; use real models for prompt-quality signal.

## Safety Verification

When changing `tools.py` or `homes/`, confirm: no `HomeAssistant` instantiation, no `hass.services.async_call`, no recorder DB imports, no `subprocess`/network/OS APIs. The only live seam must be `RecordingInvoker`. Run `scripts/check-evals` and the integration `scripts/check` (must stay green and must show no `custom_components/` changes).

## Code Style

- Python >=3.14.2, ruff `py314`, `line-length=119`, mypy strict. Concrete annotations; `from datetime import UTC, datetime` (use `UTC`, not `timezone.utc`).
- Comments at branch boundaries and safety constraints. Type-annotate all helper params.
- Keep `__init__.py` a stable surface. KISS/YAGNI.
- Do not write tests that pass only because mocks return expected values; assert observable behavior. No regression tests unless requested.
