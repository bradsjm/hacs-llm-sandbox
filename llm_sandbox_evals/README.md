# LLM Sandbox action evals

`llm_sandbox_evals` is a development-only harness for one capability: whether
an Assist model successfully invokes the required Home Assistant service. It
runs the production tools against a fresh frozen fixture snapshot and records
validated service effects through the non-live `RecordingInvoker` seam.

The baseline does not score reads, evidence, recorder output, answer structure,
blocked actions, policy rejection, service data, clarification quality,
collections, aggregates, relations, or no-data behavior. Conditional state,
history, and logbook action selection, including true no-action outcomes, is
covered by the current corpus. The model returns plain text. That prose is
retained for display and never parsed or scored.

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
id: direct_turn_on_utility_room_ceiling
home: home_full
user_request: Turn on the Utility Room ceiling light.
required_actions:
  - domain: light
    service: turn_on
    target_entity_ids: [light.utility_room_ceiling]
```

A valid no-action case uses an empty list:

```yaml
id: no_action_light_already_on
home: home_full
user_request: Turn on the Living Room ceiling light if it is off.
required_actions: []
```

`target_entity_ids` is required and nonempty. Runtime actions are always
enabled without a domain allowlist. `required_actions: []` is valid and means
that the case is correct only when the model produces zero successful actions.
Any extra successful action fails, including a different fallback service or a
duplicate of an otherwise required action. The current corpus does not author
`service_data`; service-data matching remains deferred.

## The `home_full` corpus

The corpus contains 14 cases in this progression:

1. **Direct (3):** `direct_turn_on_utility_room_ceiling`,
   `direct_turn_off_utility_room_accent`, and
   `direct_toggle_utility_room_outlet` act on Utility Room lights and a switch.
2. **Discovery (2):** `discover_utility_room_lights` selects the two lights in
   the Utility Room, and `discover_basement_ceiling_lights` selects the nine
   ceiling lights in the Basement. Each is authored as one multi-target action.
3. **Brightness/color service selection (2):**
   `brightness_utility_room_ceiling` and `color_utility_room_accent` both
   select `light.turn_on`; neither authors service data.
4. **State/history/logbook conditions (4):**
   `no_action_light_already_on` and `no_action_history_no_recent_change` have
   empty required-action lists; `condition_turn_off_living_room_ceiling` uses
   Living Room current state; `condition_history_change_turn_off` is the
   positive two-hour history case for `light.living_room_ceiling`; and the
   no-recent-change case checks the Hallway outlet.
5. **Ambiguity (3):** `ambiguous_bare_light` and
   `ambiguous_ceiling_no_area` are valid no-action cases, while
   `ambiguous_logic_living_room_recent` uses recorder evidence to select
   `light.living_room_accent`, the most recently switched-on Living Room light,
   from the otherwise ambiguous Living Room lights.

For a multi-target case, one successful call containing all resolved target IDs
is required. Multiple successful single-target calls are a different action
multiset and score incorrectly.

The offline stub is deliberately narrower than the corpus. It exact-matches
the five direct/brightness/color requests, routes them through
`execute_home_code`, emits the corresponding service call, and then returns
`Done.` as plain text:

| Request                                                 | Stub action                                    |
| ------------------------------------------------------- | ---------------------------------------------- |
| `Turn on the Utility Room ceiling light.`               | `light.turn_on` → `light.utility_room_ceiling` |
| `Turn off the Utility Room accent light.`               | `light.turn_off` → `light.utility_room_accent` |
| `Toggle the Utility Room outlet.`                       | `switch.toggle` → `switch.utility_room_outlet` |
| `Set the Utility Room ceiling light to 50% brightness.` | `light.turn_on` → `light.utility_room_ceiling` |
| `Make the Utility Room accent light warm white.`        | `light.turn_on` → `light.utility_room_accent`  |

Unmatched requests produce no tool call. Consequently, the four empty-required
cases are valid no-action stub smoke cases, while the non-empty discovery,
condition, and ambiguity-with-logic cases are intentionally not routed by the
stub. This documents stub coverage only; it does not claim real-model results.

## Scoring v5

Only the successful action ledger is scored. Required and actual effects are
compared as exact multisets of:

- domain;
- service;
- target entity IDs;

Missing, wrong, extra, and duplicate successful effects fail. An empty
`required_actions` list therefore passes only with zero successful effects and
fails on any successful effect, including a different fallback service. Rejected
action records remain diagnostic and cannot satisfy or invalidate an otherwise
exact successful ledger. Service-data matching is deferred; the current
brightness/color cases assert service and target only. Structured action
comparisons preserve unexpected effects and expose stable reason codes;
`action_mismatch` is reserved for operational fallback traces.
Operational provider, timeout, and harness failures remain `incomplete` and are
classified in diagnostics rather than the scoring reason.

Reports use scoring version 5. Version 4 and older artifacts are rejected. The
required-action trace-field rename also makes prior v5 artifacts using the old
trace contract fail validation and be rejected as legacy. There is no
compatibility decoder or rescoring shim.

## Architecture

```text
14 home_full YAML cases
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

The fixture registry intentionally exposes `home_minimal` for small synthetic
tool-contract checks and `home_full`, whose 288 entities support the corpus and
inventory-scale development checks.

## Staged expansion

Future capabilities should be introduced one observable action boundary at a
time with explicit authored contracts. The following remain deferred:

1. service-data matching;
2. policy/rejection behavior and disabled-action behavior;
3. response and clarification-quality scoring.

Conditional state/history/logbook behavior, true no-action outcomes, and
multi-target selector resolution are already in scope in the `home_full`
corpus. Read-answer or recorder-answer scoring is not implicit in those stages;
it requires a separate design with an observable contract and must not be
restored through the v4 evidence or structured-answer abstractions.

## Safety

Every case gets a fresh snapshot. No live Home Assistant object, registry,
service dispatcher, recorder database, filesystem, network, or process API is
passed to Monty. `RecordingInvoker` copies validated proposed actions and never
dispatches them to Home Assistant.
