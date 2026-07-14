# LLM Sandbox evals

## Scope

`llm_sandbox_evals/` is a development-only capability harness. It asks a
real Pydantic AI agent to use the production LLM tools against fresh frozen Home
Assistant fixtures. It scores desired end-state predicates primary, falling
back to exact action-ledger matching when no predicate is authored or the
end state is unevaluable.

## Current contract

- Cases contain `id`, `home`, `user_request`, `required_actions`, and optional
  `desired_states`.
- `desired_states` is a list of `{entity_id, state}` predicates restricted to
  `light`/`switch` entities with `on`/`off` state. Omitting it or providing an
  empty list selects action fallback. Duplicate predicate entity IDs are
  rejected at construction.
- `required_actions` may be empty and is always retained as action diagnostics
  and fallback. Required actions contain `domain`, `service`, and required
  nonempty `target_entity_ids`. Service data is authored for brightness
  (`brightness_pct: 50`) and color (`color_temp_kelvin: 2700`).
- Runtime actions are always enabled without a domain allowlist.
- Agent output is plain text (`Agent[EvalRuntime, str]`). Prose is display-only
  and is never parsed or scored.
- When `desired_states` are authored and evaluable (every predicate entity
  exists in the scoped snapshot with a binary `on`/`off` state and matching
  `light`/`switch` domain), end-state scoring is primary: all predicates
  satisfied → correct, any unsatisfied → incorrect. A satisfied state passes
  even with zero actions, extra actions, or action-ledger mismatches. An
  unsatisfied state fails even if the action ledger matches.
- When `desired_states` are absent or unevaluable, the exact action multiset
  determines the outcome. Exact call matching is tried first. If exactly one
  unmatched authored multi-target action remains, the successful concrete
  entity-ID calls may pass as `equivalent_target_partition` only when there are
  at least two calls; every target collection is nonempty and duplicate-free;
  the calls are pairwise disjoint; their exact union equals the authored target
  set; domain and service match; all actual canonical comparable
  `service_data` values are identical; and authored `service_data`, when
  present, matches. Duplicate, missing, extra, wrong-service, and
  different-data successes fail.
- The overlay reducer applies only direct `light`/`switch` `turn_on`,
  `turn_off`, and `toggle` transitions from ordered `RecordingInvoker` calls
  to a copied seed state map. Unsupported services, indirect selectors,
  attribute effects, and service data leave the overlay unchanged.
- Raw and rejected records are diagnostics only. Action results preserve
  normalized observed effects, one-to-one dimension comparisons, unexpected
  actions, and stable reason codes without weakening action matching.
- `CaseOutcome` carries `scoring_mode` (`end_state`, `actions`,
  `cap_exhausted`, or `None` for incomplete) and `score_reason`
  (`end_state_satisfied`, `end_state_unsatisfied`, an `ActionOutcomeReason`
  for fallback, `cap_exhausted`, or `None` for incomplete).
- Provider HTTP 429 responses and provider bodies containing
  `token_quota_exceeded` classify incomplete execution failures as `rate_limit`.
  Structured execution metadata is additive diagnostic context only.
- Traces and reports use scoring version 8. Version 7 and older artifacts are
  rejected as legacy; there is no compatibility decoder or rescoring path.

Do not add read scoring, evidence normalization, answer schemas, collections,
aggregates, relations, no-data, recorder-answer scoring, generic service-data
coverage, policy/rejection scoring, disabled-action behavior, or
clarification-quality scoring to this baseline. Conditional state/history/logbook
cases, true no-action cases, and multi-target selector resolution are now part
of the corpus with end-state predicates.

## Dataset and stub

The corpus is 14 `home_full` cases progressing from direct actions to discovery,
brightness/color service selection, state/history/logbook conditions including
true no-action, and ambiguity plus ambiguity-with-logic:

| Stage | Cases | Coverage |
| --- | --- | --- |
| Direct (3) | `direct_turn_on_utility_room_ceiling`, `direct_turn_off_utility_room_accent`, `direct_toggle_utility_room_outlet` | Utility Room single-target light and switch actions |
| Discovery (2) | `discover_utility_room_lights`, `discover_basement_ceiling_lights` | Utility multi-target selection uses action fallback because partial selection is not state-discriminative; Basement selects twelve ceiling lights with all targets seeded off |
| Brightness/color (2) | `brightness_utility_room_ceiling`, `color_utility_room_accent` | Utility Room `light.turn_on` with canonical `brightness_pct: 50` and `color_temp_kelvin: 2700` service data |
| Conditions (4) | `no_action_light_already_on`, `condition_turn_off_living_room_ceiling`, `condition_history_change_turn_off`, `no_action_history_no_recent_change` | Living Room current state and recent history, Hallway no-recent-change logic, and valid no-op outcomes |
| Ambiguity (3) | `ambiguous_bare_light`, `ambiguous_ceiling_no_area`, `ambiguous_logic_living_room_recent` | No-op ambiguity and Living Room recorder-based disambiguation |

Each multi-target discovery case has one required action whose target list
contains every resolved entity. Scoring still tries exact call matching first;
only one unmatched authored multi-target action can accept a complete,
disjoint, duplicate-free partition across two or more successful concrete
entity-ID calls with the same domain, service, and comparable service data.

`discover_utility_room_lights` now omits `desired_states` and uses action
fallback because the Utility Room accent must remain on for the direct turn-off
case; its shared initial state cannot distinguish a partial target selection.
`discover_basement_ceiling_lights` remains state-primary with all 12 ceiling
lights seeded off.

The offline stub is intentionally limited to the five exact, normalized
`home_full` phrases below. It calls `execute_home_code` and then emits plain
text only for these routes:

| Request | Stub action |
| --- | --- |
| `Turn on the Utility Room ceiling light.` | `light.turn_on` → `light.utility_room_ceiling` |
| `Turn off the Utility Room accent light.` | `light.turn_off` → `light.utility_room_accent` |
| `Toggle the Utility Room outlet.` | `switch.toggle` → `switch.utility_room_outlet` |
| `Set the Utility Room ceiling light to 50% brightness.` | `light.turn_on` → `light.utility_room_ceiling`; `service_data: {brightness_pct: 50}` |
| `Set the Utility Room accent light to 2700 K warm white.` | `light.turn_on` → `light.utility_room_accent`; `service_data: {color_temp_kelvin: 2700}` |

Unmatched requests produce no tool call. That makes the four empty-required
cases valid no-action stub smoke cases; non-empty discovery, condition, and
ambiguity-with-logic cases are intentionally outside the stub route surface.
This stub behavior is not a claim about real-model results.

The homes package retains `home_minimal` for small synthetic tool-contract
checks and uses `home_full` for the corpus and its complete 288-entity fixture.

## Architecture

- `schema.py` — case, action, trace, ledger, diagnostic, and outcome
  records.
- `data/cases.yaml` / `cases_schema.json` — 14 `home_full` cases and their
  focused authoring schema.
- `agent_runner.py` — plain-text agent, production tool registration, and the
  five-route direct/brightness/color offline stub.
- `runtime.py` — fresh fixture runtime with actions enabled.
- `tools.py` — visibility scoping, non-live `RecordingInvoker`, and compact
  action normalization.
- `scoring/actions.py` / `scoring/evaluate.py` — successful-ledger construction,
  exact call matching, the narrow multi-target partition equivalence, and
  end-state-primary scoring selection.
- `scoring/end_state.py` — pure overlay reducer: seeds predicate entities from
  the scoped snapshot, replays ordered `RecordingInvoker` calls as direct
  light/switch transitions, and returns a satisfied/unsatisfied/unevaluable
  assessment.
- `harness.py` — lifecycle, tool events, action extraction, overlay seed
  extraction, minimal successful tool diagnostics, and trace assembly.
- `experiment.py`, `reports.py`, `presentation.py`, `terminal.py`, `html_report.py` — overall model
  comparison, v8 persistence, immutable and runtime presentation projections, diagnostics, and end-state
  plus action-ledger display without category analysis.

## Eval UX and artifacts

- Keep `model_id` as the provider id. Persist the run-wide resolved reasoning
  and temperature fields and derive display-only variant labels.
- `scoring_mode` and `score_reason` identify which assessment determined the
  verdict. Incomplete operational failures carry `scoring_mode=None` and
  `diagnostics.failure`; presentation must use the shared effective cause.
- Count `total`, `finished`, and `scored`; quality is correct/scored and
  coverage is scored/total. Do not reintroduce user-facing `completed` counts.
- Create an atomic `manifest.json` before model calls. A cancelled or failed run
  writes the typed `partial.json` journal, which is explicitly not a report and
  has no HTML/resume path.
- Completed report folders contain `report.json`, `errors.log`, and
  `report.html` alongside the manifest. `errors.log` is UTF-8 NDJSON with one
  record per incomplete execution error in report order, including repeats and
  full error/provider detail; it is zero bytes when no execution errors occur
  and is produced only with completed reports. Completed-report writing
  atomically replaces `errors.log` before atomically replacing `report.json`, so
  a newly completed `report.json` has its companion log without a cross-file
  transaction guarantee.
- Interactive output is Rich on stderr; redirected or `--machine` output is
  deterministic stdout KV. Non-zero exits have empty stdout. Every agent run
  uses native Pydantic AI `run_stream_events`, with no streaming flag or
  non-streaming fallback. Lanes retain their five-column layout unless a real
  `thinking` event arrives for an active lane; then a sticky, structured
  Activity column appears for the run. Providers without `ThinkingPart` retain
  the five-column layout, with no synthesized reasoning or `Waiting` state.
  Activity shows only phase labels: actual `running` and `processing` tool
  phases include the validated tool name, while provider-supplied `preparing`
  names are never retained or rendered. The transient phase/activity channel is
  label/tool-name only and is not persisted in reports or artifacts. Interactive
  Activity and machine output do not render reasoning content, model responses,
  tool arguments, or tool results. Existing durable reports retain their
  established `CaseTrace` answer and tool-diagnostics contract. Human live and
  persistent durable final output render `Operational issues` as a full-width
  actionable table; exact duplicates group for display with affected cells,
  while `errors.log` remains per-trace and machine output remains payload-free.

Production read tools may remain registered because they are part of the product
surface. Their eval-specific scoring and stub routes must not return.

## Staged future work

Expand only through explicit plans and observable contracts. The current
corpus authors `desired_states` for 9 of 14 cases; the remaining 5
(`discover_utility_room_lights`, brightness, color, bare ambiguity, ceiling
ambiguity) use action fallback: Utility discovery requires complete target
matching, brightness and color require canonical service data, and the
ambiguity cases require safe abstention. The overlay reducer supports only direct `light`/`switch`
`turn_on`/`turn_off`/`toggle` state transitions; expanding it to attributes
or other domains requires a new state-based case and explicit reducer support.
Policy/rejection behavior, disabled-action behavior, and
response/clarification-quality scoring remain deferred. Read-answer scoring
requires a separate design rather than restoration of v4 models or evidence
modules.

## Safety and commands

- Build a fresh snapshot for every cell.
- Never expose live Home Assistant objects, registries, services, recorder
  databases, filesystem, network, or OS/process APIs to Monty.
- `RecordingInvoker` is the only action seam and never dispatches live calls.
- Keep eval dependencies in the `evals` group and do not modify
  `custom_components/` for eval-only changes.

```text
scripts/setup-evals
scripts/format-evals
scripts/check-evals
scripts/yaml-check
scripts/markdown-check
scripts/check
```

Use Python >=3.14.2, Ruff `py314`, strict mypy, typed test parameters, and
behavioral assertions on action effects rather than implementation choreography.
