# LLM Sandbox action evals

`llm_sandbox_evals` is a development-only harness for one capability: whether
an Assist model successfully invokes the expected Home Assistant service. It
runs the production tools against a fresh frozen fixture snapshot and records
validated service effects through the non-live `RecordingInvoker` seam.

The baseline does not score reads, evidence, recorder output, answer structure,
blocked actions, conditions, collections, aggregates, relations, or no-data
behavior. The model returns plain text. That prose is retained for display and
never parsed or scored.

## Run

```text
scripts/setup-evals
scripts/check-evals
scripts/format-evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id> --html
```

Real Pydantic AI model IDs may replace `stub`; provider credentials come from
the environment. Artifacts are written below `eval_data/runs/<run_id>/`.

## Case contract

`data/cases.yaml` uses the Pydantic Evals `name` / `cases[*].inputs` wrapper.
Each `inputs` object has exactly:

```yaml
id: action_turn_on_bedroom_light
home: home_minimal
user_request: Turn on bedroom light
expected_actions:
  - domain: light
    service: turn_on
    target_entity_ids: [light.bedroom]
    # service_data: {brightness: 100}  # optional, exact when authored
```

`target_entity_ids` is required and nonempty. Runtime actions are always
enabled without a domain allowlist.

The initial dataset is exactly four direct requests on `home_minimal`:

| Request                    | Successful effect                  |
| -------------------------- | ---------------------------------- |
| Turn on bedroom light      | `light.turn_on` → `light.bedroom`  |
| Turn off bedroom light     | `light.turn_off` → `light.bedroom` |
| Turn on living room light  | `light.turn_on` → `light.living`   |
| Turn off living room light | `light.turn_off` → `light.living`  |

The offline stub supports exactly this matrix. It calls
`execute_home_code`, emits the matching service call, then returns `Done.` as
plain text.

## Scoring v5

Only the successful action ledger is scored. Expected and actual effects are
compared as exact multisets of:

- domain;
- service;
- target entity IDs;
- non-target `service_data` when authored.

Missing, wrong, extra, and duplicate successful effects fail. Rejected action
records remain diagnostic and cannot satisfy or invalidate an otherwise exact
successful ledger. Structured action comparisons identify service, target, and
service-data agreement, preserve unexpected effects, and expose stable reason
codes; `action_mismatch` is reserved for operational fallback traces.
Operational provider, timeout, and harness failures remain `incomplete` and are
classified in diagnostics rather than the scoring reason.

Reports use scoring version 5. Version 4 and older artifacts are rejected;
there is no compatibility decoder or rescoring shim.

## Architecture

```text
four direct YAML cases
        |
fresh scoped HomeSnapshot
        |
Pydantic AI Agent[EvalRuntime, str]
        |
production tools -> RecordingInvoker -> action ledger
        |
exact successful-effect comparison
        |
correct / incorrect / incomplete + diagnostics
```

Production read tools remain registered because they are part of the product
surface, but their outputs have no scoring role in this baseline. Independent
production-tool contract tests remain useful for protecting the fixture/runtime
seam.

The fixture registry intentionally exposes only `home_minimal`, whose complete
surface is the two lights used by these cases, and `home_full`, whose 288
entities support inventory-scale development checks.

## Staged expansion

Future capabilities should be introduced one observable action boundary at a
time, with new authored cases and direct behavioral tests. Likely stages are:

1. service data on one target;
2. multiple explicit targets;
3. selector-resolved targets;
4. conditional actions with a separately specified branch contract;
5. policy rejection behavior.

Read-answer or recorder scoring is not implicit in those stages. It requires a
separate design with an observable contract and must not be restored through
the v4 evidence or structured-answer abstractions.

## Safety

Every case gets a fresh snapshot. No live Home Assistant object, registry,
service dispatcher, recorder database, filesystem, network, or process API is
passed to Monty. `RecordingInvoker` copies validated proposed actions and never
dispatches them to Home Assistant.
