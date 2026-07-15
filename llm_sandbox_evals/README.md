# LLM Sandbox evals

`llm_sandbox_evals` is a development-only harness for Assist behavior against
fresh frozen Home Assistant fixture snapshots. Its 17-case corpus uses three
explicit oracle types: `effect` for sparse final-state effects with exact action
fallback, `tool_calls` for normalized production-tool events, and `answer` for
deterministic typed predicates over plain-text read answers.

The narrow effect overlay handles binary light/switch state plus light
brightness and color temperature. The harness does not score blocked actions,
policy rejection, generic service-data coverage, or clarification quality.
Every task has a stable canonical request; future paraphrases remain distinct
utterance-level cells and also contribute to task-level robustness.

## Run

```text
scripts/setup-evals
scripts/check-evals
scripts/format-evals
uv run --group dev --group evals python -m llm_sandbox_evals eval --models stub
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini,stub
# With --judge-model, judging is limited to the five opted-in complex-code cases.
uv run --group dev --group evals python -m llm_sandbox_evals eval --models gpt-4o-mini --judge-model gpt-5.4
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id> --html
uv run --group dev --group evals python -m llm_sandbox_evals report <run_id> --markdown
```

Real Pydantic AI model IDs may replace `stub`; bare candidate or judge IDs such
as `gpt-4o-mini` are resolved to `openai-chat:gpt-4o-mini`, while explicit
provider-prefixed IDs containing `:` are preserved. The singular
`--judge-model MODEL` is optional; its normalized `stub` value is rejected.
Provider credentials come from the environment. Artifacts are written below
`eval_data/runs/<run_id>/`.
Interactive terminals receive one Rich summary on stderr and the artifact
location once. Redirected output, or `--machine`, emits deterministic KV on
stdout. Non-zero exits leave stdout empty.

`optimize --cross-eval-models` uses the same Pydantic AI model ID behavior for
its baseline-vs-optimized leaderboard. `optimize --target-model` and
`--proposer-model` remain DSPy/LiteLLM IDs, for example
`openrouter/openai/gpt-4o-mini`.

### Optional code-quality judge

The code judge has two independent gates: the case author must set
`judge_code: true`, and the run must supply `--judge-model MODEL`. The default
is false. Five current complex-code cases are opted in:
`discover_utility_room_lights` (area discovery and coordinated multi-target
action), `discover_basement_ceiling_lights` (large inventory area/name
filtering and twelve targets), `condition_history_change_turn_off` (history
processing and conditional action), `no_action_history_no_recent_change`
(history processing and conditional no-op), and
`ambiguous_logic_living_room_recent` (comparing recent histories across
candidates). All other current cases omit `judge_code` and remain false. When
`--judge-model MODEL` is supplied, it invokes the judge only for cells from
those five selected cases; without the model, no cells are judged. A separately
authored case can opt in without changing its oracle; the judge is advisory and
oracle-agnostic.

It assesses the complete `execute_home_code` trajectory as ephemeral
request-scoped glue code, prioritizing effective task contribution, minimal
model/tool round trips, scoped reads, in-sandbox computation, and compact useful
output. Ruff, formatting, comments, abstractions, typing, tests, and
maintainability are not part of this advisory rubric. The bounded context
contains the request, trusted deterministic outcome, every ordered code call
with execution status and bounded output/action/resolution/note evidence, and
compact summaries of relevant interleaved non-code tools. It excludes the
answer, expected evidence, and live objects. If complete code source or action
evidence cannot fit, no provider call is made and the result is unavailable
rather than partially judged. Zero-submission cells still receive one judge
call. The existing model timeout applies; there are no judge retries or
fallbacks. Provider, validation, and timeout failures are native
`EvaluatorFailure` records and do not change the deterministic outcome.

Native results remain `code_quality_score` and `code_quality_pass`, from the
stable evaluator `code_quality_judge` using rubric `llm_sandbox_code_quality`
version `2`. The resolved model and rubric identity are persisted in the run
descriptor/metadata. `CaseTrace` remains unchanged and scoring stays v9: judge
results do not affect correctness, quality, coverage, Wilson intervals,
ranking, rescoring, `partial.json`, `errors.log`, machine phase lines, or the
deterministic CSV score. A cancellation during judging emits no false
`cell_finished` or partial record; an ordinary judge failure releases and
completes the lane while persisting the native evaluator failure.

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
`report <run_id> --markdown` reloads `report.json` without model calls and
atomically writes deterministic `report.md` beside those artifacts.

## Case contract

`data/cases.yaml` uses the Pydantic Evals `name` / `cases[*].inputs` wrapper.
Each `inputs` object has:

```yaml
id: direct_turn_on_utility_room_ceiling
home: home_full
category: direct
oracle: effect
requests:
  - id: canonical
    text: Turn on the Utility Room ceiling light.
required_actions:
  - domain: light
    service: turn_on
    target_entity_ids: [light.utility_room_ceiling]
desired_entities:
  - entity_id: light.utility_room_ceiling
    state: "on"
```

`desired_entities` is optional. Omitting it or providing an empty list selects
action fallback. Each predicate authors a state, one or more named attributes,
or both. Duplicate predicate entity IDs are rejected.

A valid no-action state case uses an empty required-action list with a
desired state that is already satisfied:

```yaml
id: no_action_light_already_on
home: home_full
user_request: Turn on the Living Room ceiling light if it is off.
required_actions: []
desired_entities:
  - entity_id: light.living_room_ceiling
    state: "on"
```

`target_entity_ids` is required and nonempty. Runtime actions are always
enabled without a domain allowlist. `required_actions: []` is valid and is
correct when the desired state is satisfied (even with zero actions) or when
no desired entities are authored and the model produces zero successful actions.

## The `home_full` corpus

The corpus contains 17 cases across seven categories:

1. **Direct (3):** `direct_turn_on_utility_room_ceiling`,
   `direct_turn_off_utility_room_accent`, and
   `direct_toggle_utility_room_outlet` act on Utility Room lights and a switch.
2. **Discovery (2):** `discover_utility_room_lights` selects the two lights in
   the Utility Room and uses action fallback (no `desired_entities`) because its
   shared initial state cannot distinguish partial target selection;
   `discover_basement_ceiling_lights` remains state-primary with all 12 ceiling
   lights seeded off. Each is authored as one multi-target action.
3. **Brightness/color service selection (2):**
   `brightness_utility_room_ceiling` and `color_utility_room_accent` both
   select `light.turn_on`, author canonical service data (`brightness_pct: 50`
   and `color_temp_kelvin: 2700`), and require the resulting `brightness: 128`
   and `color_temp_kelvin: 2700` final attributes. The color request text is
   `Set the Utility Room accent light to 2700 K warm white.`
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
6. **Tool contract (1):** `tool_call_get_history_utility_room` requires the
   normalized `get_history` call and arguments through the `tool_calls` oracle.
7. **Read answer (2):** `answer_count_lights_on_utility_room` and
   `answer_state_utility_room_accent` use typed count and state predicates
   through the `answer` oracle.

For a multi-target case, one successful call containing all resolved target IDs
is the exact match. Scoring v9 has one narrow action equivalence: when exact
matching leaves exactly one unmatched authored multi-target action, the remaining
successful concrete entity-ID calls may score as `equivalent_target_partition`
if they form a complete, disjoint, duplicate-free partition of the authored
target set across at least two calls with matching domain, service, and
comparable service data. This is not general action merging.

Eleven effect cases author `desired_entities` and use end-state primary
scoring. The remaining three (`discover_utility_room_lights`, bare ambiguity,
and ceiling ambiguity) use action fallback because Utility discovery needs
complete target matching and the ambiguity cases require safe abstention.

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

## Scoring v9

End-state predicates are scored primary. When `desired_entities` are authored
and evaluable, the overlay reducer applies ordered `RecordingInvoker` calls to
a copied sparse seed map and evaluates only authored final fields. State fields
require a binary `on`/`off` light or switch seed. Attribute fields support only
light `brightness` and `color_temp_kelvin`:

- all predicates satisfied → correct `end_state_satisfied`
- any predicate unsatisfied → incorrect `end_state_unsatisfied`

A satisfied predicate passes even with zero actions (e.g. light already on),
extra actions, or action-ledger mismatches. An unsatisfied predicate fails even
if the action ledger matches. The overlay reducer supports direct
`light`/`switch` `turn_on`, `turn_off`, and `toggle` transitions plus light
`brightness_pct`, `brightness`, and `color_temp_kelvin` turn-on effects.
Unsupported services, indirect selectors, and other attributes leave the
overlay unchanged or make an authored predicate unevaluable.

When no `desired_entities` are authored or they are unevaluable, the exact
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

`CaseOutcome` carries `scoring_mode` (`end_state`, `actions`, `tool_calls`,
`answer`, `cap_exhausted`, or `None` for incomplete) and `score_reason`. The
declared oracle selects the primary scorer: effect cases use end state/action
fallback, tool-contract cases compare normalized successful tool events, and
read-answer cases parse only their declared deterministic predicate type.
Operational provider, timeout, and
harness failures remain `incomplete` with `scoring_mode=None` and use
`diagnostics.failure` as their effective cause. Provider HTTP 429 responses and
provider bodies containing `token_quota_exceeded` classify as `rate_limit`.
Cap exhaustion is scored incorrect with `scoring_mode="cap_exhausted"` and the
distinct effective cause `cap_exhausted`.

Reports use scoring version 9. Version 8 and older artifacts are rejected as
legacy; there is no compatibility decoder or rescoring shim. `rescore_trace()`
rebuilds the outcome from persisted oracle evidence without consulting fixture
code.

User-facing counts are `total` cells, `finished` terminal cells, and `scored`
correct plus incorrect cells. `quality_rate = correct / scored`; `coverage_rate
= scored / total`. Incomplete operational cells are excluded from quality but
remain visible in coverage and cause groupings. Stub usage and cost are
unavailable rather than zero; real-model task metrics take precedence over the
self-contained trace usage fallback.
Canonical quality is the primary candidate/model leaderboard and uses Wilson
95% intervals over scored canonical cells. Paraphrase quality remains separate
at the utterance level; task robustness reports whether every request variant
for a task passed.

## Presentation

Every agent run consumes Pydantic AI's native `run_stream_events`; there is no
streaming option or non-streaming fallback. Rich Activity is always visible
from lane creation as `queued`, is phase-colored, and remains payload-free.
Judged lanes show transient `judging` and remain active until judge termination.
Narrow layouts drop only `Variant`; Activity remains visible. There is no
synthesized `Waiting` state.

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

`report.html`, deterministic `report.md`, CSV export, and the `report` renderer
flags are all rebuilt from an
immutable saved-report presentation model. The HTML hero shows quality,
Wilson 95% confidence interval, coverage, incomplete cells, category slices,
and resolved variant configuration; incomplete inspectors show their
operational cause rather than an action mismatch. Completed HTML and Markdown
reports and `report.json` retain full judge reasons when judging was requested.
After a completed interactive judged run, the durable Rich final conditionally
appends a separate `Code judge · advisory` panel sourced from the completed
native report. It
shows judge model/rubric identity, overall requested/available counts, pass
rate, mean score, evaluator-failure and unavailable counts, per-candidate/model-
variant aggregates, and a bounded five-item needs-attention preview with
overflow. The preview uses fixed classifications and only the safe evaluator
error type; it never renders judge reasons, provider messages, stacktraces, or
request/code/tool payloads. The panel does not affect deterministic quality,
ranking, coverage, or verdict; machine KV remains judge-free.

## Architecture

```text
17 home_full YAML cases (effect, tool_calls, and answer oracles)
         |
fresh scoped HomeSnapshot
         |
Pydantic AI Agent[EvalRuntime, str]
         |
production tools -> RecordingInvoker -> ordered calls + action ledger
         |
overlay reducer (binary state + light brightness/color temperature) + exact action matching
         |
oracle-selected scoring + diagnostics
         |
correct / incorrect / incomplete + diagnostics
```

Production read tools remain registered because they are part of the product
surface. Focused cases can score normalized tool events or deterministic read
answers without weakening effect scoring.

The fixture registry intentionally exposes `home_minimal` for small synthetic
tool-contract checks and `home_full`, whose 288 entities support the corpus and
inventory-scale development checks.

## Staged expansion

Future capabilities should be introduced one observable contract at a time.
The overlay reducer currently supports direct `light`/`switch`
`turn_on`/`turn_off`/`toggle` state transitions and narrow light brightness and
color-temperature turn-on effects. Other attributes or domains require a new
case and explicit reducer support. The following remain deferred:

1. generic service-data and attribute coverage beyond brightness and color temperature;
2. policy/rejection behavior and disabled-action behavior;
3. response and clarification-quality scoring.

Conditional state/history/logbook behavior, true no-action outcomes, and
multi-target selector resolution are already in scope in the `home_full`
corpus. Read-answer scoring remains limited to authored deterministic predicate
types; the optional code judge does not score those answers.

## Safety

Every case gets a fresh snapshot. No live Home Assistant object, registry,
service dispatcher, recorder database, filesystem, network, or process API is
passed to Monty. `RecordingInvoker` copies validated proposed actions and never
dispatches them to Home Assistant.
