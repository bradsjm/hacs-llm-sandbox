# LLM Sandbox action evals

## Scope

`llm_sandbox_evals/` is a development-only action-capability harness. It asks a
real Pydantic AI agent to use the production LLM tools against fresh frozen Home
Assistant fixtures. It evaluates only successful service invocation effects.

## Current contract

- Cases contain only `id`, `home`, `user_request`, and `required_actions`.
- `required_actions` may be empty. An empty list is correct only when the
  successful action ledger is also empty.
- Required actions contain `domain`, `service`, and required nonempty
  `target_entity_ids`. The current corpus does not author `service_data`.
- Runtime actions are always enabled without a domain allowlist.
- Agent output is plain text (`Agent[EvalRuntime, str]`). Prose is display-only
  and is never parsed or scored.
- Correctness checks exact call matching first. If exact matching leaves exactly
  one unmatched authored multi-target action, the remaining successful concrete
  entity-ID calls may pass as `equivalent_target_partition` only when there are
  at least two calls; every target collection is nonempty and duplicate-free;
  the calls are pairwise disjoint; their exact union equals the authored target
  set; domain and service match; all actual canonical comparable
  `service_data` values are identical; and authored `service_data`, when
  present, matches. Duplicate, missing, extra, wrong-service, and
  different-data successes fail. Empty `required_actions` still rejects every
  successful action.
- Raw and rejected records are diagnostics only. Action results preserve
  normalized observed effects, one-to-one dimension comparisons, unexpected
  actions, and stable reason codes without weakening action matching.
- Traces and reports use scoring version 7. Version 6 and older artifacts are
  rejected as legacy; there is no compatibility decoder or rescoring path.

Do not add read scoring, evidence normalization, answer schemas, collections,
aggregates, relations, no-data, recorder-answer scoring, authored service-data
coverage, policy/rejection scoring, disabled-action behavior, or
clarification-quality scoring to this baseline. Conditional state/history/logbook
cases, true no-action cases, and multi-target selector resolution are now part
of the action corpus.

## Dataset and stub

The corpus is 14 `home_full` cases progressing from direct actions to discovery,
brightness/color service selection, state/history/logbook conditions including
true no-action, and ambiguity plus ambiguity-with-logic:

| Stage | Cases | Coverage |
| --- | --- | --- |
| Direct (3) | `direct_turn_on_utility_room_ceiling`, `direct_turn_off_utility_room_accent`, `direct_toggle_utility_room_outlet` | Utility Room single-target light and switch actions |
| Discovery (2) | `discover_utility_room_lights`, `discover_basement_ceiling_lights` | Multi-target selection of two Utility Room lights, then twelve Basement ceiling lights |
| Brightness/color (2) | `brightness_utility_room_ceiling`, `color_utility_room_accent` | Utility Room `light.turn_on` service selection; no service data is authored |
| Conditions (4) | `no_action_light_already_on`, `condition_turn_off_living_room_ceiling`, `condition_history_change_turn_off`, `no_action_history_no_recent_change` | Living Room current state and recent history, Hallway no-recent-change logic, and valid no-op outcomes |
| Ambiguity (3) | `ambiguous_bare_light`, `ambiguous_ceiling_no_area`, `ambiguous_logic_living_room_recent` | No-op ambiguity and Living Room recorder-based disambiguation |

Each multi-target discovery case has one required action whose target list
contains every resolved entity. Scoring still tries exact call matching first;
only one unmatched authored multi-target action can accept a complete,
disjoint, duplicate-free partition across two or more successful concrete
entity-ID calls with the same domain, service, and comparable service data.

The offline stub is intentionally limited to the five exact, normalized
`home_full` phrases below. It calls `execute_home_code` and then emits plain
text only for these routes:

| Request | Stub action |
| --- | --- |
| `Turn on the Utility Room ceiling light.` | `light.turn_on` → `light.utility_room_ceiling` |
| `Turn off the Utility Room accent light.` | `light.turn_off` → `light.utility_room_accent` |
| `Toggle the Utility Room outlet.` | `switch.toggle` → `switch.utility_room_outlet` |
| `Set the Utility Room ceiling light to 50% brightness.` | `light.turn_on` → `light.utility_room_ceiling` |
| `Make the Utility Room accent light warm white.` | `light.turn_on` → `light.utility_room_accent` |

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
  exact call matching, and the narrow multi-target partition equivalence.
- `harness.py` — lifecycle, tool events, action extraction, minimal successful
  tool diagnostics, and trace assembly.
- `experiment.py`, `reports.py`, `presentation.py`, `terminal.py`, `html_report.py` — overall model
  comparison, v7 persistence, immutable and runtime presentation projections, diagnostics, and action-ledger display without
  category analysis.

## Eval UX and artifacts

- Keep `model_id` as the provider id. Persist the run-wide resolved reasoning
  and temperature fields and derive display-only variant labels.
- `action_reason` is scored-action-only. Incomplete operational failures carry
  `diagnostics.failure`; presentation must use the shared effective cause.
- Count `total`, `finished`, and `scored`; quality is correct/scored and
  coverage is scored/total. Do not reintroduce user-facing `completed` counts.
- Create an atomic `manifest.json` before model calls. A cancelled or failed run
  writes the typed `partial.json` journal, which is explicitly not a report and
  has no HTML/resume path.
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
  established `CaseTrace` answer and tool-diagnostics contract.

Production read tools may remain registered because they are part of the product
surface. Their eval-specific scoring and stub routes must not return.

## Staged future work

Expand only through explicit plans and observable action contracts. The current
corpus still does not author service data; v7 only compares comparable
`service_data` to prevent partition equivalence across different action payloads
or against authored payloads if they are introduced later. Policy/rejection
behavior, disabled-action behavior, and response/clarification-quality scoring
remain deferred. Conditional state/history/logbook behavior, true no-action
outcomes, and multi-target selector resolution are already represented in the
current corpus. Read-answer scoring requires a separate design rather than
restoration of v4 models or evidence modules.

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
