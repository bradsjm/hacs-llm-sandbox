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

- Every case is oracle version 3 (`oracle_version: 3`). Keep the existing nine
  categories without preserving the former 80-case quotas.
- The final model result is one code-selected concrete shape: `EntityAnswer`,
  `EntityCollectionAnswer`, `AggregateAnswer`, `EntityRelationAnswer`,
  `NoDataAnswer`, or `ActionAnswer`. Every shape has display-only `answer` text,
  and the model sees only its selected shape, never a union.
- Every read case declares exactly one typed internal expectation. Action-only
  cases have no expectation. Conditional actions may pair one entity or
  aggregate expectation with an allowed action ledger.
- Expectations describe finite facts, not predicates or free-form paths.
  Optional tolerance on entity and aggregate expectations means approximate.
- Entity expectations may use states, history, logbook, or automation evidence;
  statistics use aggregate expectations. No-data expectations are limited to
  history, statistics, and logbook because those sources expose resolved scope.
- Entity expectation fields are source-specific: states allow state/name/
  attribute, history allows state/attribute, logbook allows message, and
  automation allows enabled/name/value/run. Only attribute expectations set
  `input_value`, using it as the required nonempty attribute name.
- Evidence must come from successful production-tool result envelopes. Tool
  arguments, model prose, `printed`, notes, error metadata, and unrelated
  records are not evidence. The scorer preserves call/turn/batch metadata for
  diagnostics and unions facts independently of the path used to obtain them.
- Allowed actions are checked from the successful ledger. Blocked actions are
  checked from the rejected ledger and require the expected policy effect,
  allowed error key, and no successful effect. Failed exploratory attempts are
  diagnostics unless a blocked-action oracle explicitly requires them.
- Direct logbook and flat declarative-history no-data expectations must name the
  exact resolved entity scope. The production response supplies
  `scope.entity_ids`; do not infer scope from raw selectors or call arguments.
  Scope and emptiness must be established by the same successful event, never
  by unioning fragments from separate events. History duration
  aggregates derive intervals independently per declared entity, require
  source rows for every entity, and extend a final matching interval through a
  valid, non-conflicting production `window.start`/`window.end` pair; pages
  are unioned only when that complete normalized window matches.
- Conditional predicates are fully authored in the request, including true
  action and false no-action behavior. Grounded antecedent facts plus the
  action ledger validate the branch; no model-provided predicate field exists.
- Keep production query behavior and fixture behavior aligned. Do not add an
  eval-only tool emulator or pass a live Home Assistant object, registry,
  recorder, service callable, filesystem, network, or OS/process API to the
  model.

## Outcome and diagnostics contract

Each completed cell has exactly one state: `correct`, `incorrect`, or
`incomplete`. Correctness requires the selected shape's required fields to
match and ground in normalized successful evidence. Aggregates are recomputed,
collections require exact returned ID sets, no-data preserves same-envelope
scope, and action effects are checked separately from the ledger. There is no
global extra-content grounding gate.
The derived native score is binary: `1.0` for correct and `0.0` otherwise.

Provider, transport, timeout, model-protocol, and unexpected harness failures
are `incomplete` and are excluded from quality denominators. Tool calls,
failed calls, repairs, model turns, parallel batches, elapsed time, token
usage, cost, and cap exhaustion are diagnostics only. The hard tool-call cap
remains a runaway safety stop; exhausting it without a valid final
flat answer is `incorrect`, not an operational penalty.
Read-only traces have no synthetic action result, and zero-completion pair,
model, or category quality rates are `N/A` rather than zero.

## Non-negotiables

- Build a fresh `HomeSnapshot` for every cell; never cache or mutate fixtures.
- Run production `run_execute` / `run_query` cores against fixture-backed data.
- Keep eval dependencies in the `evals` dependency group. Never add them to
  `[project].dependencies`, `manifest.json`, or `custom_components/**`.
- Keep `scripts/check` unchanged. The eval package has its own eval checks.
- Keep DSPy imports in `optimize_dspy.py` and the lazy CLI optimize path so
  offline eval and report commands do not require DSPy at import time. Contain
  only DSPy's known field-prefix import warning at that lazy boundary.

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
Pydantic AI Agent -> one of six concrete FinalAnswer subclasses
        |                                      |
        +--------------------+-----------------+
                             v
       normalize evidence / score required shape fields / score action ledgers
                             |
                             v
                 CaseTrace -> native report + HTML
                             |
                             v
                  correct / incorrect / incomplete
```

### Module map

- `schema.py` — six concrete answer models, one-expectation oracle models,
  `CaseOutcome`, `ConclusionResult`, action ledgers, diagnostics, and
  self-contained `CaseTrace` records with required scoring version 4.
- `cases.py` and `data/cases.yaml` — native Dataset loading and the 24-case first
  tranche; `data/cases_schema.json` is the focused oracle-v3 authoring schema.
- `homes/` — frozen Python fixture modules and the `get_home()` registry.
- `prompts.py` — production-profile candidates and prompt-size helpers.
- `agent_runner.py` — Pydantic AI agent, production tool schemas, and offline
  `FunctionModel` stub; the output shape is selected code-side per case.
- `runtime.py` — fixture-backed recorder source and eval runtime construction.
- `tools.py` — evaluation scope, fixture scoping, recording action seam, and
  action normalization; it contains no tool emulators.
- `scoring/contracts.py` — immutable normalized fact records and call metadata.
- `scoring/evidence.py` — selects successful envelopes and unions normalized
  facts from execute, history, statistics, logbook, and automation results.
- `scoring/assertions.py` — per-shape equality, tolerance, set, relation, and
  no-data comparisons.
- `scoring/actions.py` — canonical successful/rejected action-ledger checks.
- `scoring/{execute,history,statistics,logbook,automation}.py` — payload
  normalizers and source-specific grounding logic.
- `scoring/evaluate.py` — composes shape-grounding and action results into the
  binary/incomplete outcome and native score.
- `harness.py` — fresh snapshot lifecycle, structured agent run, chronological
  tool events, action ledgers, diagnostics, and failure classification.
- `experiment.py` — native Dataset matrix, outcome/coverage aggregation, and
  quality ranking by correct rate rather than operational diagnostics.
- `terminal.py` — stderr-only live progress and correct/incorrect/incomplete
  summaries.
- `reports.py` — v4 `report.json` persistence and strict artifact loading.
- `html_report.py` — self-contained report dashboard with answer grounding,
  ledgers, all chronological evidence, diagnostics, and answer details.
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
