# LLM Sandbox Evals

## Project Identity

`llm_sandbox_evals/` is a **development-only** package at repo root that evaluates the `llm_sandbox` LLM tools (`execute_home_code`, `get_history`, `get_statistics`, `get_logbook`). It runs a bounded, multi-turn native tool-calling agent loop against **frozen** `HomeSnapshot` fixtures (hand-built in Python — no JSON deserializer exists), executes the **real** `async_execute_home_code` for code cases, scores final outcomes with turn efficiency, and ranks prompt candidates across a matrix of language models.

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
- Run: `uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub --prompt-profile standard`
- Optimize: `uv run --group dev --group evals python -m llm_sandbox_evals optimize --target-model <real-model>`
- Report: `uv run --group dev --group evals python -m llm_sandbox_evals report <run_id>`

Note: eval runs need **both** groups (`dev` provides `homeassistant`, `evals` provides `litellm`). Artifacts go to the gitignored `eval_data/runs/`.

## Architecture And Data Flow

The harness (`harness.run_matrix`) runs, per `(candidate, model, case)`:

```
profile  = resolve_profile(config.prompt_profile)         # selected production prompt profile
snapshot = homes.get_home(case.home).snapshot()           # fresh frozen snapshot
messages = prompts.render_messages(candidate, case, snapshot)
schemas  = prompts.function_schemas(candidate)            # provider-native tool definitions
while turns < (case.max_turns or config.max_turns):
    step = await models.get_adapter(model).respond(model, messages, schemas)
    if not step.tool_calls:
        final_answer = step.text
        break
    for call in step.tool_calls:
        outcome = await tools.run_tool(call, case, snapshot, profile, invoker=invoker)
        messages.append(tools.tool_result_message(call.id, outcome.result))
checks = scoring.check_case(case, final_answer, recorded_actions, statuses, snapshot, steps)
score  = scoring.score_case(checks, turns, case.par_turns, config.efficiency_k, config.efficiency_floor)
```

The harness owns the snapshot lifecycle (build once per case evaluation, pass to render/run/score). Tool turns may contain one or more native tool calls. Correct outcomes score `1.0` at or under the case's `par_turns`, then decay by `efficiency_k` per extra tool-turn down to `efficiency_floor`; failed required outcome gates score `0.0`. The real executor activates/clears its own runtime contextvars internally — the harness does **not** call `activate_runtime`/`clear_runtime`.

### Module map

- `schema.py` — **stable shared contracts** (`PromptCandidate`, `EvalCase`, `Expected`, `ExpectedAction`, `CaseContext`, `ToolCall`, `AgentStep`, `StepTrace`, `ToolOutcome`, `CheckResult`, `ToolSpec`). Do not rename fields without updating all consumers.
- `config.py` — `EvalConfig` + `load_config()` (defaults: `models=["stub"]`, `candidates=["baseline"]`).
- `cases.py` — `CASES: list[EvalCase]`, the predefined suite (simple -> complex, all categories).
- `homes/` — frozen fixture modules (`snapshot() -> HomeSnapshot`, `recorder() -> dict`) + `get_home(name)` registry.
- `prompts.py` — `baseline_candidate()` (from production builders), `load_candidates(ids, prompt_profile_id)`, `render_messages(...)`, `function_schemas(...)`, `tool_specs(...)`. Reuses the selected production prompt profile / `ACTIONS_*_PROMPT` / tool-description builders; derives the request-location section from the frozen snapshot.
- `models.py` — `ModelAdapter` protocol, `StubAdapter` (offline, deterministic multi-turn validator), `LiteLLMAdapter` (any provider, lazy import), `get_adapter(id)`.
- `tools.py` — `run_tool(tool_call, case, snapshot, prompt_profile, invoker=...) -> ToolOutcome`. Real executor path + fixture-backed recorder emulators matching production response shapes + `RecordingInvoker`; `tool_result_message(...)` serializes bounded tool results for the next model turn.
- `scoring.py` — `check_case(...)`, `score_case(...)`, `mean_score(...)`. Outcome gates + turn-efficiency scoring.
- `harness.py` — `run_case(...) -> CaseTrace`, `run_matrix(config) -> RunResult`; `CaseTrace`, `CandidateModelScore`, `RunResult`.
- `reports.py` — `write_run(...)`, `render_leaderboard(...)`, `load_run_json(...)` (for `report`).
- `optimize_dspy.py` — DSPy COPRO prompt optimizer that exports optimized `PromptCandidate` artifacts and reuses the real harness metric path.
- `cli.py` / `__main__.py` — `eval`, `report`, and `optimize` subcommands.

### Key contracts

- Tool call shape: `ToolCall(id: str, tool_name: str, tool_args: dict)`, parsed from provider-native function calls. Tool names are the production constants from `custom_components.llm_sandbox.const`.
- `execute_home_code` `tool_args` = `{"code": str}`; the result dict carries `execution.status` (`ok|code_error|helper_error|setup_error`), `output`, and optional `printed`, `actions`, `note`, and `fix` fields on the relevant success/error payloads.
- Recorder `ToolOutcome.result` matches production: history `{"window": {...}, "entities": {id: {"unit"?: str, "rows": [[t, state]]}}}`, statistics `{"window": {...}, "period": str, "statistics": {id: {"rows": [[t, value]]}}}`, and logbook `{"window": {...}, "entries": [{"entity_id", "when", "name", "message", ...}]}`. Success omits `status`; `next_cursor` appears only when more rows remain. Errors are `{"status":"error","error":{"key": str, "message": str, "fix"?: list[str]}}`, including `entity_not_visible` with concrete visible candidates when available.
- `ToolOutcome.recorded_actions` for the execute path comes from `result["actions"]` (the facade's `ActionRecord` list), not the invoker's captured calls.

## How To Extend

- **Add a case:** append to `cases.CASES`. Reference a real fixture `home`; keep `expected` deterministic; set `par_turns` to the expected efficient tool-turn count. Recorder cases may use entity ids or supported selectors such as area/domain/device/floor/label in native tool args. Use `required_tool_names`, `required_tool_sequence`, `recorder_window`, `required_error_keys`, `required_result_paths`, `max_tool_turns`, `max_successful_actions`, and `evidence_*` fields for multi-tool, recovery, and no-retry cases. Do not require final-answer entity mentions for action cases; score actions from recorded side effects and intermediate tool evidence.
- **Add a fixture:** add `homes/<name>.py` with `snapshot()`/`recorder()` and `NAME`, then register in `homes/__init__.py`. Mirror `home_default.py`'s helpers (effective-area rule, sorted tuple `SnapshotIndexes`, nested `SafeContext`).
- **Add a candidate:** add a `PromptCandidate` and expose via `prompts.load_candidates`. `baseline` is auto-built; unknown ids currently raise.
- **Add a model:** no code needed — pass any litellm id to `--models`. To add a non-litellm backend, implement the `ModelAdapter` protocol and branch in `get_adapter`.

## The Stub Adapter

`StubAdapter` is a **pipeline validator**, not a scoring benchmark. It keyword-detects the tool from the **user request only** (not the whole prompt, which lists every tool name), emits one runnable native tool call, then returns a terminal answer echoing the latest tool result. Recorder cases without explicit ids use broad selectors to exercise resolver support. A low stub score in a category is expected and honest; use real models for prompt-quality signal.

## Safety Verification

When changing `tools.py` or `homes/`, confirm: no `HomeAssistant` instantiation, no `hass.services.async_call`, no recorder DB imports, no `subprocess`/network/OS APIs. The only live seam must be `RecordingInvoker`. Run `scripts/check-evals` and the integration `scripts/check` (must stay green and must show no `custom_components/` changes).

## Code Style

- Python >=3.14.2, ruff `py314`, `line-length=119`, mypy strict. Concrete annotations; `from datetime import UTC, datetime` (use `UTC`, not `timezone.utc`).
- Comments at branch boundaries and safety constraints. Type-annotate all helper params.
- Keep `__init__.py` a stable surface. KISS/YAGNI.
- Do not write tests that pass only because mocks return expected values; assert observable behavior. No regression tests unless requested.
