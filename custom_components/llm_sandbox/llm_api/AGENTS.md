# Tool Purpose and Alignment

`execute_home_code` should help an LLM complete the user's Home Assistant task, not force the LLM to write perfect Python.

Treat the submitted code as short-lived task glue: interpret reasonable intent, accept common LLM coding patterns, and prefer "do what the LLM likely meant" over strict rejection when it is safe to do so.

Design for success in one call, and recovery in no more than one follow-up call.

On success, return the useful result directly. On failure, return actionable feedback that tells the next LLM call exactly what went wrong, what names or APIs are available, and what concrete change is likely to work.

Do not require the LLM to learn integration-specific tricks when normal Home Assistant knowledge can be adapted safely inside the tool.

## How the directive is applied

The choices below are the operational meaning of the directive. Each names the strategy, where it lives, and the rationale. Follow them when changing any tool in this package.

### Forgive in the pipeline, do not reject or instruct
The LLM's submitted code runs through a single shadow-safe rewrite engine (`normalization/`) before Monty sees it, followed by the separate result-binding stage (in `executor.py`). The engine runs one parse and one unparse and applies an ordered registry of fail-open `RewriteRule`s (datetime → builtin → state sugar → await), each proving resolution against one scope/shadow model before firing. When you find a widespread pattern the LLM writes that Monty rejects, append a `RewriteRule` to the registry with its safety precondition explicit in `apply` — do **not** add a prompt instruction ("don't use X") and do not let it fail. Only rewrite shapes whose evaluation semantics are provably preserved; leave ambiguous shapes untouched so the natural error surfaces and the refinement layer can guide recovery.

The builtin rules (`normalization/rules/builtin_rules.py`) today do two things:
- `TypeNameRule` resolves `type(<bare facade global>).__name__` to the friendly class name as a string constant, but only when `type` is not rebound and the receiver still resolves to the sandbox global. Rationale: Monty's native `type(x).__name__` returns the unhelpful literal `"dataclass"` for registered dataclasses; the static resolution gives the LLM a recognizable name (e.g. `"SafeFloorRegistry"` → scrubbed later to `floor_registry`).
- `NextIterRule` wraps the first argument of every `next(...)` in `iter(...)` (label `wrapped_next_iter`), but only when neither `next` nor `iter` is rebound. Rationale: Monty represents generator expressions and comprehension results as lists, not iterators, so `next(xs)` / `next(genexpr)` fail at runtime with `'list' object is not an iterator`; `iter()` makes any iterable a valid iterator and is idempotent on real iterators, so multi-`next` patterns are preserved. Explicit `next(iter(x))` written by the LLM is left untouched.

### Declare the builtin surface, do not rewrite it
Monty's **runtime** builtin surface is wider than its **type-checker** surface: `hasattr`, `getattr`, `next`, `iter`, `map`, and `filter` all run natively, but the type-checker rejects them (`error[unresolved-reference]: Name 'hasattr' used when not defined`) only because the auto-generated stubs do not declare them. `MONTY_BUILTIN_STUBS` (`contracts.py`) declares these builtins in the generated stubs so common LLM discovery patterns type-check **and** run.

Rationale (verified by probing the installed `pydantic_monty`):
- Runtime `getattr`/`hasattr` **cannot walk dunders** (`getattr(x, '__class__', d)` → `d`; `hasattr(x, '__dict__')` → `False`), so declaring them opens **no new escape surface**.
- Because they run natively, the old static `_resolve_hasattr`/`_resolve_getattr` and the `_MapFilterRewriter` (which turned `map`/`filter` into list comprehensions) were **removed** — declaring the stubs fixes the type-check with strictly less code.
- Keep this list to builtins Monty already runs; do not declare builtins that would require a runtime that does not exist.

### Fail open on type-check strictness the runtime tolerates
The Monty program is built once with `type_check=True` (`_build_monty` in `executor.py`). If the type-check fails with a **reference** error (`unresolved-reference` / `used-when-not-defined` — a genuinely undefined name that should refine into a `NameError`) or with a non-diagnostic construction failure (e.g. `SyntaxError`), the error surfaces normally. For any other type-check diagnostic (`error[…]` that is not a reference error — e.g. `invalid-assignment` from heterogeneous dict seeding, which the runtime accepts), the program is rebuilt with `type_check=False` and the `type_check_relaxed` normalization is recorded. Rationale: the type-checker is stricter than the runtime on some legitimate LLM shapes; relaxing those (and only those) lets the code run and surface a real error if one exists, rather than rejecting runnable code at the gate.

### Never leak integration internals
`refine_code_error` (`executor_support/refinement.py`, exported through `executor_support.REFINERS`) is the single funnel that turns Monty/type-check/runtime errors into LLM-facing guidance. It runs an **ordered `REFINERS` registry** of rule functions; the first rule that applies wins, and a rule returns `None` to defer. Add a new recovery hint by appending a rule — do not grow a prompt. It must:
- Reclassify into familiar Python error types (`NameError`, `ImportError`, `AttributeError`, `TypeError`).
- **Scrub internal class names** (`SafeFloorRegistry` → `floor_registry`, `SafeState` → `state`) via `_friendly_class_name`. A model-visible message must never name a `Safe*` dataclass or quote a Python internal like `unhashable type: 'dict'`.
- Emit **concrete next-step guidance** for the known traps. Today the registry covers: `dir`/`vars` (→ list the facade's public surface), `setattr`/`delattr`/`__import__` (→ use pre-bound globals), unresolved imports (→ only `json`/`math`/`re`), `%` and `str.format()` (→ f-string), dict-method misuse on list results like `async_all().items()` (→ iterate the list directly), method calls on `NoneType` (→ guard with `if x is not None:`), and missing attributes (→ surface the object's public surface).

### Enrich HA-native records with derived join keys
Snapshot records serialize via the `_JsonSafeRecord` mixin (`snapshot/models.py`), which derives `__llm_sandbox_json__` from `dataclass.fields()` and returns raw values the executor's `json_safe` recurses through — so adding a field is the only change needed to expose it.

`SafeState` carries registry-derived join keys (`area_id`, `device_id`, `platform`, `unique_id`, default `None`) filled by `enrich_states` (`snapshot/builder.py`) using the index rule `entity.area_id or device.area_id`. Rationale: an LLM filtering by area/device otherwise has to perform a manual state→entity→device join; the derived fields preserve the HA-native state shape while removing the join, which was a top eval failure cause (`'state' object has no attribute 'area_id'`). The eval fixtures reuse `enrich_states` so they apply the same effective-area rule.

### Self-describe empty results
An unguessable integration-specific entity id often surfaces as an empty result (e.g. `states.get("light.kitchen_main")` → `None`). When the final output is empty/`None` and the static scan found literal missing state ids, the executor adds a top-level `note` sourced from `advise(snapshot, FailureContext(intent=READ_STATE, ...))` in the structured `guidance/` engine. Imperative `Use X` wording and `memory.record` happen only for `exact`/`high` confidence; `ambiguous`/`listing`/`none` confidence produces non-imperative listing or absence guidance and **does not** write resolution memory. A later run that repeats a remembered missing literal may rewrite it before Monty sees the code, but only when the replacement still exists in the fresh snapshot, and success payloads report that transparency as `resolutions: [{requested, applied}]`. `referenced_missing` is an internal scan helper only, never a payload field.

`_referenced_missing` is computed by a **static AST scan** of the submitted code for literal `states.get("…")` / `states["…"]` / `hass.states.…` reads (`executor.py`). Rationale: Monty **copies input objects** and **does not propagate the runtime contextvar to synchronous methods** (verified: sync `get` sees the default contextvar; async methods see the set one), so a sync `states.get` cannot record to Python-side state at runtime. The static scan catches the dominant case — an LLM-typed literal id — without that constraint, and never fires on a successful (non-empty) result. In the errored read path, a `NoneType` `AttributeError` caused by `states.get("<bad id>")` attaches `execution.guidance` with confidence-ranked candidates via the guidance engine.

### Service calls: block without crashing, resolve before rejecting
Service-call outcomes (`SafeServiceRegistry.async_call` in `facades/services.py`) first apply response-mode accommodation, then split into three classes via `_policy_block` and `_visible_target`:
- **Response-mode accommodation** runs before the policy gate: any service whose snapshot `supports_response` is `ONLY` or `OPTIONAL` is run with `blocking=True, return_response=True`, so the LLM never has to remember the flag. Rationale: an `ONLY` service cannot execute in Home Assistant without `return_response=True`, so blocking on its absence rejected correct intent; lifting `OPTIONAL` calls into response mode prevents the downstream `NoneType` crash when the result is consumed. `NONE` services are untouched — passing `return_response=True` to a `NONE` service is contradictory intent (not incomplete), so it is still blocked below.
- **Policy blocks** (`actions_disabled`, `action_domain_not_allowed`, `service_not_found`, NONE response-mode mismatch, unresolved `service_target_not_visible`) **do not raise**. They record an errored action (carrying `error.key`/`error.message` and optional structured `error.guidance`) and `return None`, so execution stays `status="ok"` with a recorded errored action. Rationale: a policy gate is not a code bug; crashing the whole run to `helper_error` failed the `execution_status` and `no_action_when_disabled` eval gates even though the code's intent was correct.
- **Live failures** (exceptions from the live invoker, and the per-call `service_call_timeout` when no budget remains) **still raise** `HelperExecutionError` after recording the errored action and classifying via `_service_call_error`. Live failures are real and must surface.
- **Not-visible targets** are not rejected outright. `_visible_target` auto-resolves an entity id via `resolve_target_entity` (`resolution.py`): an exact visible id wins; otherwise a **unique token-containment match** (one token set a subset of the other over `object_id`/`name` tokens) resolves; an ambiguous match returns candidates. Unresolved selectors and service-target blocks route through `advise(intent=RESOLVE_SELECTOR/CALL_SERVICE, ...)` and record structured `error.guidance` with confidence-ranked candidates before returning `None`. Rationale: the LLM cannot guess integration-specific ids; resolving the unique intent and offering candidates targets success in one call instead of a hard rejection.

Successful action records are compact: `{service:"domain.service", target, status:"ok", resolved_from?}`. The domain is folded into `service`; the full request echo, `blocking`, `return_response`, `service_data`, and null response/error fields were removed so actions carry only decision-relevant fields.

If a success payload contains any `actions[].status == "ok"`, the executor adds a top-level `notes` entry explaining that service calls were accepted but later `hass.states` reads still reflect the frozen snapshot for this tool call. Rationale: LLMs otherwise reread stale state and repeat successful toggles or try alternate services.

If a success payload contains any `actions[].status == "error"`, the executor also adds a top-level `notes` entry summarizing blocked/failed actions while keeping `execution.status == "ok"`; the action error and its `error.guidance` remain the detailed recovery surface.

`resolution.py` is now the exact/unique auto-resolve primitive, not the recovery-ranker. `resolve_target_entity` remains the trusted service-target resolver and is reused internally by the `guidance/` engine; `candidates_for_domain`, `rank_candidates_for_service`, and `available_hint` were removed. Structured recovery ranking, bounding, confidence policy, and wording live in `guidance/advise()`. Optional conversation memory is explicit and advisory: it can bias ordering or break ambiguity ties only for entity ids still present in the fresh snapshot candidate set, and memory writes are gated by guidance confidence. It never touches live Home Assistant.

### Accommodate every documented HA idiom
Registry facades (`facades/registries.py`) accept the full HA-native surface under both short (`er`) and long (`entity_registry`) names as one object. `async_get` dispatches by argument shape; traversal methods accept both the two-arg HA form and the clean one-arg form. Do not add validation that rejects a legitimate HA call shape — adapt it internally instead.

### Recorder tools: scope and size naturally
Because the sandbox forbids `timedelta`/arithmetic, "last N hours" is expressed as a tool **input**, never computed in code. Every recorder tool accepts:
- A relative `hours=<n>` window, plus ISO `start`/`end` (`_clamp_window` in `tools/recorder.py`).
- HA-native scoping selectors `area_id`/`device_id`/`floor_id`/`label_id`/`domain` resolved against the snapshot indexes (`resolve_entity_ids`). Explicit IDs that are not visible name themselves in the error; selectors expand to the visible set. A selector that is present but matches nothing (e.g. a typo'd `area_id`) raises `selector_no_match` with candidate ids rather than widening — the pure-domain expansion only fires when no IDs and no selectors are given.

### Errors carry actionable guidance
Recoverable tool errors (`tool_error_envelope` in `errors.py`) and errored service actions carry a stable `key`, a single specific `message` (never the key itself), and an optional `guidance` object. The guidance payload is `{confidence, candidates, reason, next_step, cross_kind}` where candidates are bounded, ranked objects with `id`, `name`, `match`, and `detail`. Stable keys remain translated in `translations/en.json` for the human contract; `message`/`guidance` are the LLM-facing remediation that targets success on the next call. The structured `guidance` object replaces the old flat `fix` list entirely; there is no compatibility shim.

### Where new forgiveness lives
New accommodation belongs as code in this package, keeping `prompts.py` to a statement of available surface — not a list of integration-specific rules the LLM must obey. The sanctioned seams are:
- **Forgiveness pipeline** (`normalization/`): append a `RewriteRule` to the ordered registry, with its safety precondition explicit in `apply`; the engine's single scope/shadow model and framework-level fail-open gate cover every rule.
- **Builtin surface** (`contracts.MONTY_BUILTIN_STUBS`): declare a Monty-native builtin the type-checker lacks, after verifying it runs and cannot escape.
- **Refinement funnel** (`executor_support.REFINERS`): append a rule that turns one error shape into actionable guidance.
- **Guidance engine** (`guidance/advise()`): rank recovery candidates, apply confidence policy, and produce structured `guidance` instead of ad-hoc suggestions.
- **Facade behavior** (`facades/`, `resolution.py`): adapt an HA call shape, or exact/unique-resolve before routing unresolved recovery through guidance.
- **Record enrichment** (`snapshot/models.py` + `snapshot/builder.enrich_states`): add a derived join field that removes a manual join.
