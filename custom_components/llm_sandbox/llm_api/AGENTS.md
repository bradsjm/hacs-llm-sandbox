# Tool Purpose and Alignment

`execute_home_code` should help an LLM complete the user's Home Assistant task, not force the LLM to write perfect Python.

Treat the submitted code as short-lived task glue: interpret reasonable intent, accept common LLM coding patterns, and prefer "do what the LLM likely meant" over strict rejection when it is safe to do so.

Design for success in one call, and recovery in no more than one follow-up call.

On success, return the useful result directly. On failure, return actionable feedback that tells the next LLM call exactly what went wrong, what names or APIs are available, and what concrete change is likely to work.

Do not require the LLM to learn integration-specific tricks when normal Home Assistant knowledge can be adapted safely inside the tool.

## How the directive is applied

The choices below are the operational meaning of the directive. Each names the strategy, where it lives, and the rationale. Follow them when changing any tool in this package.

### Forgive in the pipeline, do not reject or instruct
The LLM's submitted code runs through a sequence of independent, **fail-open** AST passes before Monty sees it (datetime → builtin → await → result binding in `executor.py`). Each pass normalizes one category of common LLM variation into a form Monty can run. When you find a widespread pattern the LLM writes that Monty rejects, add a normalization pass — do **not** add a prompt instruction ("don't use X") and do not let it fail. Only rewrite shapes whose evaluation semantics are provably preserved; leave ambiguous shapes untouched so the natural error surfaces and the refinement layer can guide recovery.

The builtin pass (`builtin_normalization.py`) today does two things:
- Resolves `type(<bare facade global>).__name__` to the friendly class name as a string constant. Rationale: Monty's native `type(x).__name__` returns the unhelpful literal `"dataclass"` for registered dataclasses; the static resolution gives the LLM a recognizable name (e.g. `"SafeFloorRegistry"` → scrubbed later to `floor_registry`).
- Wraps the first argument of every `next(...)` in `iter(...)` (`_NextIterWrapper`, label `wrapped_next_iter`). Rationale: Monty represents generator expressions and comprehension results as lists, not iterators, so `next(xs)` / `next(genexpr)` fail at runtime with `'list' object is not an iterator`; `iter()` makes any iterable a valid iterator and is idempotent on real iterators, so multi-`next` patterns are preserved. Explicit `next(iter(x))` written by the LLM is left untouched.

### Declare the builtin surface, do not rewrite it
Monty's **runtime** builtin surface is wider than its **type-checker** surface: `hasattr`, `getattr`, `next`, `iter`, `map`, and `filter` all run natively, but the type-checker rejects them (`error[unresolved-reference]: Name 'hasattr' used when not defined`) only because the auto-generated stubs do not declare them. `MONTY_BUILTIN_STUBS` (`contracts.py`) declares these builtins in the generated stubs so common LLM discovery patterns type-check **and** run.

Rationale (verified by probing the installed `pydantic_monty`):
- Runtime `getattr`/`hasattr` **cannot walk dunders** (`getattr(x, '__class__', d)` → `d`; `hasattr(x, '__dict__')` → `False`), so declaring them opens **no new escape surface**.
- Because they run natively, the old static `_resolve_hasattr`/`_resolve_getattr` and the `_MapFilterRewriter` (which turned `map`/`filter` into list comprehensions) were **removed** — declaring the stubs fixes the type-check with strictly less code.
- Keep this list to builtins Monty already runs; do not declare builtins that would require a runtime that does not exist.

### Fail open on type-check strictness the runtime tolerates
The Monty program is built once with `type_check=True` (`_build_monty` in `executor.py`). If the type-check fails with a **reference** error (`unresolved-reference` / `used-when-not-defined` — a genuinely undefined name that should refine into a `NameError`) or with a non-diagnostic construction failure (e.g. `SyntaxError`), the error surfaces normally. For any other type-check diagnostic (`error[…]` that is not a reference error — e.g. `invalid-assignment` from heterogeneous dict seeding, which the runtime accepts), the program is rebuilt with `type_check=False` and the `type_check_relaxed` normalization is recorded. Rationale: the type-checker is stricter than the runtime on some legitimate LLM shapes; relaxing those (and only those) lets the code run and surface a real error if one exists, rather than rejecting runnable code at the gate.

### Never leak integration internals
`refine_code_error` (`executor_support.py`) is the single funnel that turns Monty/type-check/runtime errors into LLM-facing guidance. It runs an **ordered `REFINERS` registry** of rule functions; the first rule that applies wins, and a rule returns `None` to defer. Add a new recovery hint by appending a rule — do not grow a prompt. It must:
- Reclassify into familiar Python error types (`NameError`, `ImportError`, `AttributeError`, `TypeError`).
- **Scrub internal class names** (`SafeFloorRegistry` → `floor_registry`, `SafeState` → `state`) via `_friendly_class_name`. A model-visible message must never name a `Safe*` dataclass or quote a Python internal like `unhashable type: 'dict'`.
- Emit **concrete next-step guidance** for the known traps. Today the registry covers: `dir`/`vars` (→ list the facade's public surface), `setattr`/`delattr`/`__import__` (→ use pre-bound globals), unresolved imports (→ only `json`/`math`/`re`), `%` and `str.format()` (→ f-string), dict-method misuse on list results like `async_all().items()` (→ iterate the list directly), method calls on `NoneType` (→ guard with `if x is not None:`), and missing attributes (→ surface the object's public surface).

### Enrich HA-native records with derived join keys
Snapshot records serialize via the `_JsonSafeRecord` mixin (`snapshot/models.py`), which derives `__llm_sandbox_json__` from `dataclass.fields()` and returns raw values the executor's `json_safe` recurses through — so adding a field is the only change needed to expose it.

`SafeState` carries registry-derived join keys (`area_id`, `device_id`, `platform`, `unique_id`, default `None`) filled by `enrich_states` (`snapshot/builder.py`) using the index rule `entity.area_id or device.area_id`. Rationale: an LLM filtering by area/device otherwise has to perform a manual state→entity→device join; the derived fields preserve the HA-native state shape while removing the join, which was a top eval failure cause (`'state' object has no attribute 'area_id'`). The eval fixtures reuse `enrich_states` so they apply the same effective-area rule.

### Self-describe empty results
An unguessable integration-specific entity id often surfaces as an empty result (e.g. `states.get("light.kitchen_main")` → `None`). When the final output is empty/`None`, the executor attaches an `available_hint` naming the visible entities in the first referenced domain, and always reports `referenced_missing` (the literal entity ids the code read that are absent).

`referenced_missing` is computed by a **static AST scan** of the submitted code for literal `states.get("…")` / `states["…"]` / `hass.states.…` reads (`_referenced_missing` in `executor.py`). Rationale: Monty **copies input objects** and **does not propagate the runtime contextvar to synchronous methods** (verified: sync `get` sees the default contextvar; async methods see the set one), so a sync `states.get` cannot record to Python-side state at runtime. The static scan catches the dominant case — an LLM-typed literal id — without that constraint, and never fires on a successful (non-empty) result.

### Service calls: block without crashing, resolve before rejecting
Service-call outcomes (`SafeServiceRegistry.async_call` in `facade_views.py`) split into three classes via `_policy_block` and `_visible_target`:
- **Policy blocks** (`actions_disabled`, `action_domain_not_allowed`, `service_not_found`, response-mode mismatch, unresolved `service_target_not_visible`) **do not raise**. They record an errored action (carrying `key`/`placeholders`/`message` and optional `hints`) and `return None`, so execution stays `status="ok"` with a recorded errored action. Rationale: a policy gate is not a code bug; crashing the whole run to `helper_error` failed the `execution_status` and `no_action_when_disabled` eval gates even though the code's intent was correct.
- **Live failures** (exceptions from the live invoker, and the per-call `service_call_timeout` when no budget remains) **still raise** `HelperExecutionError` after recording the errored action and classifying via `_service_call_error`. Live failures are real and must surface.
- **Not-visible targets** are not rejected outright. `_visible_target` auto-resolves an entity id via `resolve_target_entity` (`resolution.py`): an exact visible id wins; otherwise a **unique same-domain fuzzy match** (token overlap on `object_id`/`name`) resolves; an ambiguous match returns candidates. Only when nothing matches does it record a blocked action with a candidate hint and return `None`. Rationale: the LLM cannot guess integration-specific ids; resolving the unique intent and offering candidates targets success in one call instead of a hard rejection.

`resolution.py` is the pure, snapshot-aware Suggester that backs this: `resolve_target_entity`, `candidates_for_domain`, and `available_hint` (also reused for empty-result hints). It never touches live Home Assistant.

### Accommodate every documented HA idiom
Registry facades (`facade_views.py`) accept the full HA-native surface under both short (`er`) and long (`entity_registry`) names as one object. `async_get` dispatches by argument shape; traversal methods accept both the two-arg HA form and the clean one-arg form. Do not add validation that rejects a legitimate HA call shape — adapt it internally instead.

### Recorder tools: scope and size naturally
Because the sandbox forbids `timedelta`/arithmetic, "last N hours" is expressed as a tool **input**, never computed in code. Every recorder tool accepts:
- A relative `hours=<n>` window, plus ISO `start`/`end` (`_clamp_window` in `tools/recorder.py`).
- HA-native scoping selectors `area_id`/`device_id`/`floor_id`/`label_id`/`domain` resolved against the snapshot indexes (`_resolve_entity_ids`). Explicit IDs that are not visible name themselves in the error; selectors expand to the visible set.

### Errors carry actionable hints
Recoverable tool errors (`tool_error_envelope` in `errors.py`) include `message` + `hints` (concrete next steps), not just a stable key. Stable keys remain translated in `translations/en.json` for the human contract; `hints`/`message` are the LLM-facing remediation that targets success on the next call.

### Where new forgiveness lives
New accommodation belongs as code in this package, keeping `prompts.py` to a statement of available surface — not a list of integration-specific rules the LLM must obey. The sanctioned seams are:
- **Forgiveness pipeline** (`executor.py`, `builtin_normalization.py`): a new fail-open AST pass for a widespread rejected shape.
- **Builtin surface** (`contracts.MONTY_BUILTIN_STUBS`): declare a Monty-native builtin the type-checker lacks, after verifying it runs and cannot escape.
- **Refinement funnel** (`executor_support.REFINERS`): append a rule that turns one error shape into actionable guidance.
- **Facade behavior** (`facade_views.py`, `resolution.py`): adapt an HA call shape, or resolve/offer candidates instead of rejecting.
- **Record enrichment** (`snapshot/models.py` + `snapshot/builder.enrich_states`): add a derived join field that removes a manual join.
