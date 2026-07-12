# LLM Sandbox Evals

## Project identity

`llm_sandbox_evals/` is a **development-only** package. It evaluates the
`llm_sandbox` LLM tools (`execute_home_code`, `get_history`,
`get_statistics`, `get_logbook`, and `get_automation`) with a real
`pydantic_ai.Agent` over fresh, frozen Home Assistant fixtures. It calls the
production, Home-Assistant-free tool cores and ranks prompt candidates through
native `pydantic_evals` reports. It is not part of the integration runtime and
never connects to a live Home Assistant instance.

## Authoring rules

Cases are realistic human Home Assistant requests, not tool-contract,
sandbox-enforcement, malformed-input, or unit/integration-test cases.

- Every case is oracle version 2 (`oracle_version: 2`). Preserve the existing
  nine categories and their 80-case distribution.
- The final model result is an `EvalAnswer`: display-only `answer` text plus a
  typed `claims` list. The answer text is never parsed or scored.
- Every read/answer case declares one or more typed `ExpectedConclusion`s.
  Action-only cases may have no conclusions. A case must otherwise declare an
  allowed action ledger or a blocked-action outcome.
- Claims must describe facts, not predicates or free-form paths. Use only the
  finite claim fields and assertion vocabulary documented in the README.
- Evidence must come from successful production-tool result envelopes. Tool
  arguments, model prose, `printed`, notes, error metadata, and unrelated
  records are not evidence. The scorer preserves call/turn/batch metadata for
  diagnostics and unions facts independently of the path used to obtain them.
- Allowed actions are checked from the successful ledger. Blocked actions are
  checked from the rejected ledger and require the expected policy effect,
  allowed error key, and no successful effect. Failed exploratory attempts are
  diagnostics unless a blocked-action oracle explicitly requires them.
- Direct logbook and flat declarative-history no-data conclusions must name the
  exact resolved entity scope. The production response supplies
  `scope.entity_ids`; do not infer scope from raw selectors or call arguments.
- Keep production query behavior and fixture behavior aligned. Do not add an
  eval-only tool emulator or pass a live Home Assistant object, registry,
  recorder, service callable, filesystem, network, or OS/process API to the
  model.

## Outcome and diagnostics contract

Each completed cell has exactly one state: `correct`, `incorrect`, or
`incomplete`. Correctness requires every expected conclusion to be both
semantically matched and grounded in normalized successful evidence; all
submitted claims must also be grounded. Action effects are checked separately.
The derived native score is binary: `1.0` for correct and `0.0` otherwise.

Provider, transport, timeout, model-protocol, and unexpected harness failures
are `incomplete` and are excluded from quality denominators. Tool calls,
failed calls, repairs, model turns, parallel batches, elapsed time, token
usage, cost, and cap exhaustion are diagnostics only. The hard tool-call cap
remains a runaway safety stop; exhausting it without a valid final
`EvalAnswer` is `incorrect`, not an efficiency penalty.

## Non-negotiables

- Build a fresh `HomeSnapshot` for every cell; never cache or mutate fixtures.
- Run production `run_execute` / `run_query` cores against fixture-backed data.
- Keep eval dependencies in the `evals` dependency group. Never add them to
  `[project].dependencies`, `manifest.json`, or `custom_components/**`.
- Keep `scripts/check` unchanged. The eval package has its own eval checks.
- Keep DSPy imports in `optimize_dspy.py` and the lazy CLI optimize path so
  offline eval and report commands do not require DSPy at import time.

## Commands

```text
scripts/setup-evals
scripts/check-evals
scripts/format-evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub --prompt-profile balanced
uv run --group dev --group evals python -m llm_sandbox_evals optimize --target-model <real-model>
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id>
```

`check-evals` runs Ruff, mypy, the eval tests, and an offline stub matrix.
`format-evals` runs Ruff format. Eval runs need both `dev` (Home Assistant)
and `evals` (Pydantic AI/Evals) groups. Artifacts are written to the ignored
`eval_data/runs/`; a non-empty `LOGFIRE_TOKEN` enables Logfire export while
console telemetry remains disabled.

Repository-wide documentation and YAML checks are `scripts/markdown-check` and
`scripts/yaml-check`; `scripts/check` runs both in addition to the integration
checks.

## Architecture and data flow

```text
case YAML + fixture
        |
        v
fresh scoped HomeSnapshot -> EvalRuntime -> production tool cores
        |                                      |
        |                                      v
        |                         successful tool envelopes + action records
        v                                      |
Pydantic AI Agent -> EvalAnswer(answer, claims)
        |                                      |
        +--------------------+-----------------+
                             v
          normalize evidence / score claims / score action ledgers
                             |
                             v
                 CaseTrace -> native report + HTML
                             |
                             v
                  correct / incorrect / incomplete
```

### Module map

- `schema.py` — versioned `EvalAnswer`/claim models, v2 case oracles,
  `CaseOutcome`, `ConclusionResult`, action ledgers, diagnostics, and
  self-contained `CaseTrace` records.
- `cases.py` and `data/cases.yaml` — native Dataset loading and the 80 authored
  cases; `data/cases_schema.json` is the focused v2 authoring schema.
- `homes/` — frozen Python fixture modules and the `get_home()` registry.
- `prompts.py` — production-profile candidates and prompt-size helpers.
- `agent_runner.py` — Pydantic AI agent, production tool schemas, and offline
  `FunctionModel` stub; the agent output type is `EvalAnswer`.
- `runtime.py` — fixture-backed recorder source and eval runtime construction.
- `tools.py` — evaluation scope, fixture scoping, recording action seam, and
  action normalization; it contains no tool emulators.
- `scoring/contracts.py` — immutable normalized fact records and call metadata.
- `scoring/evidence.py` — selects successful envelopes and unions normalized
  facts from execute, history, statistics, logbook, and automation results.
- `scoring/assertions.py` — finite equality, tolerance, set, and empty-scope
  comparisons.
- `scoring/actions.py` — canonical successful/rejected action-ledger checks.
- `scoring/{execute,history,statistics,logbook,automation}.py` — payload
  normalizers and source-specific grounding logic.
- `scoring/evaluate.py` — composes conclusion and action results into the
  binary/incomplete outcome and native score.
- `harness.py` — fresh snapshot lifecycle, structured agent run, chronological
  tool events, action ledgers, diagnostics, and failure classification.
- `experiment.py` — native Dataset matrix, outcome/coverage aggregation, and
  quality ranking by correct rate rather than operational diagnostics.
- `terminal.py` — stderr-only live progress and correct/incorrect/incomplete
  summaries.
- `reports.py` — v2 `report.json` persistence and strict artifact loading.
- `html_report.py` — self-contained report dashboard with claims, grounding,
  ledgers, evidence, diagnostics, and answer details.
- `optimize_dspy.py` — DSPy COPRO prompt export and binary harness metric.
- `cli.py` / `__main__.py` — `eval`, `report`, and `optimize` commands.

## Production result contracts used by scoring

- Successful `execute_home_code` results have `execution.status == "ok"`.
  Facts are extracted only from JSON-safe `output` records, never from
  `printed`, notes, metadata, arguments, or answer text.
- Successful recorder/automation envelopes do not have top-level
  `status: "error"`. History and statistics keyed records retain their IDs
  even when rows are empty.
- Flat declarative history returns `{window, scope: {entity_ids}, rows}`.
- Logbook returns `{window, scope: {entity_ids}, entries}`. The scope is the
  resolved, sorted visible scope and is present on empty and continuation
  pages. It grants no access and must not expose raw hidden IDs.

## Safety verification

When changing fixtures or runtime seams, confirm that no live Home Assistant
object, recorder database, service dispatcher, subprocess, network, or OS API
can reach the model. `RecordingInvoker` is the only action seam. Run
`scripts/check-evals` and `scripts/check`; the latter must remain free of
`custom_components/` changes for eval-only work.

## Style

Use Python >=3.14.2, Ruff `py314`, line length 119, strict mypy, concrete
annotations, and comments at branch boundaries or safety constraints. Keep
`__init__.py` a stable public surface. Tests should assert observable
behavior, use typed parameters, and avoid brittle presentation-copy checks.
