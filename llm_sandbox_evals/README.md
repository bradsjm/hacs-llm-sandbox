# LLM Sandbox Evals

Development-only evaluation harness for the `llm_sandbox` Home Assistant
integration. It runs a real `pydantic_ai.Agent` against fresh, frozen fixtures,
executes the production tool cores, and evaluates prompt candidates across a
candidate × model matrix with native `pydantic_evals` reports. It never touches
a live Home Assistant instance and is not imported by the integration runtime.

## Quick start

```bash
scripts/setup-evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
```

The deterministic `stub` FunctionModel is keyless and validates the pipeline:
structured agent output, production tool calls, v2 scoring, and report writing.
Every run writes `report.json` and `report.html` under
`eval_data/runs/<run_id>/`. Standard output contains machine-readable artifact
paths and correct-rate summaries; interactive progress is written to stderr.

## Commands

```text
python -m llm_sandbox_evals eval [--models id,...] [--candidates id,...]
  [--prompt-profile ID] [--cases id,...|category,...] [--concurrency N]
  [--max-tool-calls N] [--model-timeout SECONDS] [--reasoning LEVEL]
  [--temperature FLOAT] [--runs-dir PATH]

python -m llm_sandbox_evals optimize --target-model ID [--proposer-model ID]
  [--prompt-profile ID] [--breadth N] [--depth N] [--length-penalty COEFF]
  [--cases ...] [--cross-eval-models ...] [--target-reasoning LEVEL]
  [--proposer-reasoning LEVEL] [--reasoning LEVEL] [--runs-dir PATH]

python -m llm_sandbox_evals report <run_id> [--html] [--runs-dir PATH]
```

- `eval` runs every selected candidate/model/case cell. Defaults are the
  `stub` model, `baseline` candidate, `balanced` production profile, all cases,
  concurrency 5, a 10-call safety cap, and a 75-second model-generation
  timeout.
- `report` reloads a saved v2 report without model calls. `--html` regenerates
  only the self-contained HTML dashboard.
- `--cases` accepts case IDs or `state`, `registry`, `history`, `statistics`,
  `logbook`, `automation`, `action`, `safety`, and `system`.
- `--candidates` accepts `baseline`, `profile:<id>`, and
  `optimized:<path>` candidate artifacts.
- Real Pydantic AI models use provider-prefixed IDs such as
  `openai:gpt-4o-mini` or `openrouter:...`; keys come from the environment or
  a gitignored root `.env` file.

## V2 scoring model

The only model-produced eval result is:

```python
EvalAnswer(answer: str, claims: list[AnswerClaim])
```

`answer` is for display and diagnostics only. It is never parsed, searched, or
used as evidence. `claims` is a discriminated, extra-forbidden union of these
typed forms:

- `value`: one entity, device, area, automation, repair, notification, or
  service field and scalar value;
- `relation`: one finite relationship such as entity-to-device,
  entity-to-area, device-to-area, automation-to-target, or entity-to-service;
- `collection`: sorted unique IDs filtered by `all`, area, device, floor, label,
  domain, or state;
- `aggregate`: a fixed operation over declared state, history, statistics, or
  logbook subjects;
- `event`: one history transition, logbook message, or automation run;
- `no_data`: an empty history, statistics, logbook, or automation source with an
  exact resolved entity scope.

Authored v2 conclusions use exactly one of these assertions:

| Assertion        | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `equals`         | All identifying fields and the value/event match exactly.           |
| `approximate`    | Numeric value or aggregate is within the positive finite tolerance. |
| `exact_items`    | A collection has exactly the authored item set.                     |
| `contains_items` | A collection contains every authored item.                          |
| `empty`          | The source is empty and its resolved scope exactly matches.         |

Every expected conclusion must have a semantically matching answer claim and a
grounded normalized fact. All submitted claims must also be grounded, so an
extra unrelated claim makes the cell incorrect. Aggregates are recomputed from
all declared source records; the expected number appearing in prose or a
different record is not evidence.

Evidence is path-independent: successful production envelopes from any number
of calls, turns, or parallel batches are normalized and unioned. A failed call
followed by a successful call can therefore be correct, and a final failed
tool event does not erase earlier valid evidence. Facts retain tool, call,
turn, batch, and record metadata for traceability, but scoring never reads raw
arguments, `printed`, notes, error metadata, or model prose.

Direct recorder no-data results need identity as well as emptiness. The flat
declarative-history envelope is `{window, scope: {entity_ids}, rows}` and the
logbook envelope is `{window, scope: {entity_ids}, entries}`. `scope.entity_ids`
is the sorted, visible scope resolved by the production query core. It appears
on successful empty results, normal pages, and cursor continuations. It is not
copied from the request and does not grant access. A no-data claim must be
proven by one successful envelope whose scope exactly matches and whose
same-envelope source collection has no relevant rows; scope fragments from
separate calls are not combined. History duration claims require a valid,
non-conflicting `window.start`/`window.end` pair and include each entity's
terminal matching-state interval through that endpoint. Pagination pages are
unioned only when the complete normalized window matches.

Actions are scored from two separate ledgers:

- The successful ledger must match the authored allowed effects, including
  domain, service, canonical relevant service data, and the exact target union.
  Split disjoint calls are allowed; duplicates, overlaps, supersets, and
  unrelated successful effects are not.
- The rejected ledger is used for blocked-action cases. Required policy/error
  keys and rejected effects must be present, with no successful effects.

Read-only cases fail on unexpected successful effects. Failed exploratory or
recovery attempts remain diagnostics unless the case explicitly authors a
blocked effect. Their trace has no synthetic action-result row: an empty
`actions` tuple means no action contract, while the ledger still exposes any
unexpected effect.

### Cell states and aggregation

Each cell is exactly `correct`, `incorrect`, or `incomplete`:

- `correct` means all conclusions, claims, and action expectations pass; its
  native score is `1.0`.
- `incorrect` means the model completed but semantic grounding or action
  expectations failed; its native score is `0.0`.
- `incomplete` means provider, transport, timeout, model-protocol, or harness
  failure. It is excluded from correct-rate denominators and shown as coverage.

The hard call cap is a safety stop. Cap exhaustion without a valid final
`EvalAnswer` is `incorrect`, not an operational penalty. Calls, failed calls,
repairs, turns, parallelism, elapsed time, token usage, cost, and cap status
are diagnostics only; they cannot change a completed cell's score or rank.
Reports rank by correct rate, then the minimum completed-model correct rate,
then authored prompt size. Pair, model, and category rates with no completed
cells are `N/A` and are excluded from quality ordering.

## Adding cases

Edit `data/cases.yaml`; it is loaded through native `Dataset.from_file()`.
`data/cases_schema.json` is the focused v2 authoring sidecar. Cases retain the
current categories:
`state`, `registry`, `history`, `statistics`, `logbook`, `automation`,
`action`, `safety`, and `system`.

Minimal state example:

```yaml
- name: my_temperature_case
  inputs:
    id: my_temperature_case
    category: state
    home: home_default
    user_request: What is the living room temperature?
    actions_enabled: false
    oracle_version: 2
    expected:
      conclusions:
        - claim:
            kind: value
            subject_kind: entity
            subject_id: sensor.living_temp
            field: state
            attribute_name: null
            value: "25.2"
          assertion: equals
          tolerance: null
      actions: []
      blocked_outcome: null
```

Use exact IDs and values from a real frozen fixture. Action cases declare
`expected.actions`; blocked cases declare `expected.blocked_outcome`; read
cases declare `expected.conclusions`. Do not add legacy token lists, generic
predicates, free-form result paths, or call-count expectations.
Conditional predicates belong in the complete human request, including the
true action and false no-action instruction. Grounded antecedent conclusions
plus the action ledger validate the branch; the model does not submit a
separate predicate field.

## Fixtures and production alignment

Add a Python module under `homes/` exposing `snapshot() -> HomeSnapshot`,
`recorder() -> dict`, and `NAME`, then register it in `homes/__init__.py`.
Fixtures are fresh, frozen, deterministic Python data; there is no JSON
deserializer for `HomeSnapshot`. Recorder cases exercise the production
`GetHistoryTool`, `GetStatisticsTool`, and `GetLogbookTool` query cores with a
fixture-backed `RecorderSource`. The eval harness must not duplicate those
query implementations.

## Prompt optimization

`optimize` uses DSPy COPRO to propose `api_prompt` rewrites and evaluates each
proposal through the real v2 harness. The quality metric is the binary
`trace.outcome.score` (`1.0` only for correct cells, otherwise `0.0`). The
optional length penalty is used only inside COPRO to prefer a smaller authored
prompt when quality is tied; reported baseline and optimized means remain raw
correct rates. The exported `optimized_candidate.json` and
`optimized_prompt.md` never patch production prompts automatically.
DSPy remains a lazy optimize-only import; the CLI import boundary contains
only its known `InputField`/`OutputField` prefix deprecation warning.

## Artifacts and report versioning

Each run directory contains:

- `report.json` — native analyses plus self-contained v2 traces containing the
  authored oracle, `EvalAnswer`, normalized conclusion results, action ledgers,
  chronological tool events, diagnostics, outcome, and per-trace
  `scoring_version: 2`;
- `report.html` — an interactive dashboard for outcomes, expected conclusions,
  claim semantic/grounding results, successful/rejected effects, evidence,
  diagnostics, and the unrestricted final answer.

`reports.load_report()` requires version 2 on both the artifact envelope and
every trace, and rejects missing or older markers with
`legacy scoring artifact; rerun evaluation`.
Only scoring v2 artifacts are accepted; unsupported artifacts must be rerun.
Saved optimized candidate JSON remains loadable because it contains prompt
fields only, but it must be evaluated again under v2.

## Checks

```bash
scripts/check-evals
scripts/format-evals
scripts/markdown-check
scripts/yaml-check
scripts/check
```

`check-evals` runs Ruff, mypy, eval tests, and the offline stub matrix.
`scripts/check` is the repository-wide check and must remain unchanged for
eval work. `scripts/markdown-check` uses markdownlint and `scripts/yaml-check`
uses yamllint over repository YAML configuration.
