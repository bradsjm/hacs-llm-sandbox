# LLM Sandbox evals

`llm_sandbox_evals` is a development-only harness for one capability: whether
an Assist model reaches the desired Home Assistant end state. It runs the
production tools against a fresh frozen fixture snapshot, records validated
service effects through the non-live `RecordingInvoker` seam, and derives a
post-run overlay from ordered calls to evaluate authored desired-state
predicates. When no predicate is authored or the end state is unevaluable,
exact action-ledger matching is used as fallback.

The baseline does not score reads, evidence, recorder output, answer structure,
blocked actions, policy rejection, generic service-data coverage, clarification
quality, collections, aggregates, relations, or no-data behavior. Action
fallback compares authored canonical service data for brightness and color, but
service data has no overlay effect. Conditional state,
history, and logbook action selection, including true no-action outcomes, is
covered by the current corpus with end-state predicates. The model returns
plain text. That prose is retained for display and never parsed or scored.

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
Completed report folders contain `report.json`, `errors.log`, and `report.html`
alongside the manifest. `errors.log` is UTF-8 NDJSON with one record per
incomplete execution error in report order, preserving repeated incidents and
full error/provider detail; it is zero bytes when no execution errors occurred.
Completed-report writing atomically replaces `errors.log` before atomically
replacing `report.json`, so a newly completed `report.json` has its companion
log without claiming a cross-file transaction. It is written with completed
reports only, not backfilled for older runs or promised for failed/cancelled
partial-only folders.

## Case contract

`data/cases.yaml` uses the Pydantic Evals `name` / `cases[*].inputs` wrapper.
Each `inputs` object has:

```yaml
id: direct_turn_on_utility_room_ceiling
home: home_full
user_request: Turn on the Utility Room ceiling light.
required_actions:
  - domain: light
    service: turn_on
    target_entity_ids: [light.utility_room_ceiling]
desired_states:
  - entity_id: light.utility_room_ceiling
    state: "on"
```

`desired_states` is optional. Omitting it or providing an empty list selects
action fallback. Predicates are restricted to `light`/`switch` entities with
`on`/`off` state. Duplicate predicate entity IDs are rejected.

A valid no-action state case uses an empty required-action list with a
desired state that is already satisfied:

```yaml
id: no_action_light_already_on
home: home_full
user_request: Turn on the Living Room ceiling light if it is off.
required_actions: []
desired_states:
  - entity_id: light.living_room_ceiling
    state: "on"
```

`target_entity_ids` is required and nonempty. Runtime actions are always
enabled without a domain allowlist. `required_actions: []` is valid and is
correct when the desired state is satisfied (even with zero actions) or when
no desired states are authored and the model produces zero successful actions.

## The `home_full` corpus

The corpus contains 14 cases in this progression:

1. **Direct (3):** `direct_turn_on_utility_room_ceiling`,
   `direct_turn_off_utility_room_accent`, and
   `direct_toggle_utility_room_outlet` act on Utility Room lights and a switch.
2. **Discovery (2):** `discover_utility_room_lights` selects the two lights in
   the Utility Room and uses action fallback (no `desired_states`) because its
   shared initial state cannot distinguish partial target selection;
   `discover_basement_ceiling_lights` remains state-primary with all 12 ceiling
   lights seeded off. Each is authored as one multi-target action.
3. **Brightness/color service selection (2):**
   `brightness_utility_room_ceiling` and `color_utility_room_accent` both
   select `light.turn_on` and author canonical service data: `brightness_pct: 50`
   and `color_temp_kelvin: 2700`. The color request text is `Set the Utility Room
accent light to 2700 K warm white.`
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
is the exact match. Scoring v8 has one narrow action equivalence: when exact
matching leaves exactly one unmatched authored multi-target action, the remaining
successful concrete entity-ID calls may score as `equivalent_target_partition`
if they form a complete, disjoint, duplicate-free partition of the authored
target set across at least two calls with matching domain, service, and
comparable service data. This is not general action merging.

Nine of the fourteen cases author `desired_states` and use end-state primary
scoring. The remaining five (`discover_utility_room_lights`, brightness, color,
bare ambiguity, and ceiling ambiguity) use action fallback because Utility
discovery needs complete target matching, brightness and color require
canonical service data, and the ambiguity cases require safe abstention.

The three recorder-evidence entities — Living Room ceiling, Living Room accent,
and Hallway outlet — have snapshot `last_changed` values matching their
terminal recorder history/logbook timestamps.

The offline stub is deliberately narrower than the corpus. It exact-matches
the five direct/brightness/color requests, routes them through
`execute_home_code`, emits the corresponding service call, and then returns
`Done.` as plain text:

| Request                                                   | Stub action                                                                              |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `Turn on the Utility Room ceiling light.`                 | `light.turn_on` → `light.utility_room_ceiling`                                           |
| `Turn off the Utility Room accent light.`                 | `light.turn_off` → `light.utility_room_accent`                                           |
| `Toggle the Utility Room outlet.`                         | `switch.toggle` → `switch.utility_room_outlet`                                           |
| `Set the Utility Room ceiling light to 50% brightness.`   | `light.turn_on` → `light.utility_room_ceiling`; `service_data: {brightness_pct: 50}`     |
| `Set the Utility Room accent light to 2700 K warm white.` | `light.turn_on` → `light.utility_room_accent`; `service_data: {color_temp_kelvin: 2700}` |

Unmatched requests produce no tool call. Consequently, the four empty-required
cases are valid no-action stub smoke cases, while the non-empty discovery,
condition, and ambiguity-with-logic cases are intentionally not routed by the
stub. This documents stub coverage only; it does not claim real-model results.

## Scoring v8

End-state predicates are scored primary. When `desired_states` are authored
and evaluable (every predicate entity exists in the scoped snapshot with a
binary `on`/`off` state and matching `light`/`switch` domain), the overlay
reducer applies ordered `RecordingInvoker` calls to a copied seed state map
and evaluates the final state:

- all predicates satisfied → correct `end_state_satisfied`
- any predicate unsatisfied → incorrect `end_state_unsatisfied`

A satisfied state passes even with zero actions (e.g. light already on), extra
actions, or action-ledger mismatches. An unsatisfied state fails even if the
action ledger matches. The overlay reducer supports only direct `light`/`switch`
`turn_on`, `turn_off`, and `toggle` transitions; unsupported services, indirect
selectors, attribute effects, and service data leave the overlay unchanged.

When no `desired_states` are authored or they are unevaluable, the exact
action multiset is scored as fallback. Required and actual effects are matched
by exact call equality first across:

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
fail the action fallback. Raw and rejected action records remain diagnostic and
cannot satisfy or invalidate an otherwise matched successful ledger. The action
ledger and comparison are always computed and retained as diagnostics regardless
of scoring mode.

`CaseOutcome` carries `scoring_mode` (`end_state`, `actions`, `cap_exhausted`,
or `None` for incomplete) and `score_reason`. Operational provider, timeout, and
harness failures remain `incomplete` with `scoring_mode=None` and use
`diagnostics.failure` as their effective cause. Provider HTTP 429 responses and
provider bodies containing `token_quota_exceeded` classify as `rate_limit`.
Cap exhaustion is scored incorrect with `scoring_mode="cap_exhausted"` and the
distinct effective cause `cap_exhausted`.

Reports use scoring version 8. Version 7 and older artifacts are rejected as
legacy; there is no compatibility decoder or rescoring shim. `rescore_trace()`
rebuilds the outcome from persisted `desired_states`, `overlay_state_seeds`,
`recorded_invocations`, and `action_ledger` without consulting fixture code.

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
candidate × variant quality, coverage, operational issues, and usage. The human
terminal `Operational issues` section is a full-width actionable table in both
live and persistent durable final output. Exact duplicate issues group for
display with affected cells, while `errors.log` remains one record per trace;
machine output remains payload-free.

`report.html`, CSV export, and `report <run_id> --html` are all rebuilt from an
immutable saved-report presentation model. The HTML hero shows quality,
coverage, incomplete cells, and resolved variant configuration; incomplete
inspectors show their operational cause rather than an action mismatch.

## Architecture

```text
14 home_full YAML cases (10 with desired_states, 4 action-only)
         |
fresh scoped HomeSnapshot
         |
Pydantic AI Agent[EvalRuntime, str]
         |
production tools -> RecordingInvoker -> ordered calls + action ledger
         |
overlay reducer (light/switch turn_on/off/toggle) + exact action matching
         |
end_state primary / actions fallback + diagnostics
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

Future capabilities should be introduced one observable contract at a time.
The overlay reducer currently supports only direct `light`/`switch`
`turn_on`/`turn_off`/`toggle` state transitions. Expanding it to attributes
(brightness, color) or other domains requires a new state-based case and
explicit reducer support. The following remain deferred:

1. attribute-level end-state predicates (brightness, color);
2. generic service-data coverage in the corpus;
3. policy/rejection behavior and disabled-action behavior;
4. response and clarification-quality scoring.

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
