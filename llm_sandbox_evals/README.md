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
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini,stub
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id> --html
```

Real Pydantic AI model IDs may replace `stub`; bare IDs such as `gpt-4o-mini`
are resolved to `openai-chat:gpt-4o-mini`, while `stub` and explicit
provider-prefixed IDs containing `:` are preserved. Provider credentials come
from the environment. Artifacts are written below `eval_data/runs/<run_id>/`.
Interactive terminals receive one Rich summary on stderr and the artifact
location once. Redirected output, or `--machine`, emits deterministic KV on
stdout. Non-zero exits leave stdout empty.

`optimize --cross-eval-models` uses the same Pydantic AI model ID behavior for
its baseline-vs-optimized leaderboard. `optimize --target-model` and
`--proposer-model` remain DSPy/LiteLLM IDs, for example
`openrouter/openai/gpt-4o-mini`.

Each run creates `manifest.json` before model calls. Its status moves from
`running` to `complete`, `cancelled`, or `failed`. Cancellation and operational
failure also write `partial.json`: a typed journal of terminal cells, **not** a
native Pydantic Evals report. Partials cannot be rendered as HTML or resumed.

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
`service_data`.

## The `home_full` corpus

The corpus contains 14 cases in this progression:

1. **Direct (3):** `direct_turn_on_utility_room_ceiling`,
   `direct_turn_off_utility_room_accent`, and
   `direct_toggle_utility_room_outlet` act on Utility Room lights and a switch.
2. **Discovery (2):** `discover_utility_room_lights` selects the two lights in
   the Utility Room, and `discover_basement_ceiling_lights` selects the twelve
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
is the exact match. Scoring v7 has one narrow equivalence: when exact matching
leaves exactly one unmatched authored multi-target action, the remaining
successful concrete entity-ID calls may score as `equivalent_target_partition`
if they form a complete, disjoint, duplicate-free partition of the authored
target set across at least two calls with matching domain, service, and
comparable service data. This is not general action merging.

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

## Scoring v7

Only the successful action ledger is scored. Required and actual effects are
matched by exact call equality first across:

- domain;
- service;
- target entity IDs;
- canonical comparable service data, when present.

If exact matching leaves exactly one unmatched authored action, and that action
has multiple target entity IDs, the remaining successful concrete entity-ID
calls may pass as `equivalent_target_partition` only when all of these are true:

- at least two successful calls remain;
- every remaining call target collection is nonempty and duplicate-free;
- the remaining call target collections are pairwise disjoint;
- the exact union of remaining targets equals the authored target set;
- every remaining call has the authored domain and service;
- all actual canonical comparable `service_data` values are identical;
- authored `service_data`, if present, matches that actual comparable data.

Missing, extra, duplicate, wrong-service, and different-data successful effects
fail. An empty `required_actions` list therefore passes only with zero
successful effects and fails on any successful effect, including a different
fallback service. Raw and rejected action records remain diagnostic and cannot
satisfy or invalidate an otherwise matched successful ledger. The current corpus
does not author `service_data`; the comparable-data checks exist to prevent
partition equivalence across different actual payloads or against authored
payloads if later cases add them. Structured action comparisons preserve
unexpected effects and expose stable action reason codes.
`CaseOutcome.action_reason` is present only for scored correct/incorrect cells.
Operational provider, timeout, and harness failures remain `incomplete`, have
`action_reason: null`, and use `diagnostics.failure` as their effective cause.
Cap exhaustion is scored incorrect with its real action reason and the distinct
effective cause `cap_exhausted`.

Reports use scoring version 7. Version 6 and older artifacts are rejected as
legacy; there is no compatibility decoder or rescoring shim. `model_id` remains
the provider id, while every trace and descriptor persist the resolved run-wide
`reasoning_effort` and `temperature`. Presentation derives labels such as
`luna(high)` or `luna(default)` without changing provider routing.

User-facing counts are `total` cells, `finished` terminal cells, and `scored`
correct plus incorrect cells. `quality_rate = correct / scored`; `coverage_rate
= scored / total`. Incomplete operational cells are excluded from quality but
remain visible in coverage and cause groupings. Stub usage and cost are
unavailable rather than zero; real-model task metrics take precedence over the
self-contained trace usage fallback.

## Presentation

Every agent run consumes Pydantic AI's native `run_stream_events`; there is no
streaming option or non-streaming fallback. Live lanes retain their five-column
layout (request, variant, elapsed/timeout, and tools/cap) unless a real
`thinking` event is observed for an active lane. That event makes a structured,
payload-free Activity column sticky for the run. Providers that do not emit
`ThinkingPart` therefore keep the five-column layout: the harness never
synthesizes reasoning or a `Waiting` state.

Activity uses only structured phase labels. Actual `running` and `processing`
tool phases include the validated tool name; provider/model-supplied
`preparing` names are neither retained nor rendered. The transient
phase/activity channel is label/tool-name only and is not persisted in reports
or artifacts. Interactive Activity and machine output do not render reasoning
content, model responses, tool arguments, or tool results. Existing durable
reports retain their established `CaseTrace` answer and tool-diagnostics
contract. After the transient Live frame ends, one durable final compares
candidate × variant quality, coverage, operational issues, and usage.

`report.html`, CSV export, and `report <run_id> --html` are all rebuilt from an
immutable saved-report presentation model. The HTML hero shows quality,
coverage, incomplete cells, and resolved variant configuration; incomplete
inspectors show their operational cause rather than an action mismatch.

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
exact-first action comparison + narrow partition equivalence
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

1. authored service-data coverage in the corpus;
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
