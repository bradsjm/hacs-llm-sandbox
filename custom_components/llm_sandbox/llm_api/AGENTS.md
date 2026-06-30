# Tool Purpose and Alignment

`execute_home_code` should help an LLM complete the user's Home Assistant task, not force the LLM to write perfect Python.

Treat the submitted code as short-lived task glue: interpret reasonable intent, accept common LLM coding patterns, and prefer "do what the user likely meant" over strict rejection when it is safe to do so.

Design for success in one call, and recovery in no more than one follow-up call.

On success, return the useful result directly. On failure, return actionable feedback that tells the next LLM call exactly what went wrong, what names or APIs are available, and what concrete change is likely to work.

Do not require the LLM to learn integration-specific tricks when normal Home Assistant knowledge can be adapted safely inside the tool.

## How the directive is applied

The choices below are the operational meaning of the directive. Follow them when changing any tool in this package.

### Forgive in the pipeline, do not reject or instruct
The LLM's submitted code runs through a sequence of independent, **fail-open** AST passes (datetime → builtin → await → result binding in `executor.py`). Each pass normalizes one category of common LLM variation into a form Monty can run. When you find a widespread pattern the LLM writes that Monty rejects (e.g. `map`/`filter` → comprehensions in `builtin_normalization._MapFilterRewriter`), add a rewrite pass — do **not** add a prompt instruction ("don't use X") and do not let it fail. Only rewrite shapes whose evaluation semantics are provably preserved; leave ambiguous shapes untouched so the natural error surfaces and the refinement layer can guide recovery.

### Never leak integration internals
`refine_code_error` (`executor_support.py`) is the single funnel that turns Monty/type-check errors into LLM-facing guidance. It must:
- Reclassify into familiar Python error types (`NameError`, `ImportError`, `AttributeError`, `TypeError`).
- **Scrub internal class names** (`SafeFloorRegistry` → `floor_registry`, `SafeState` → `state`) via `_friendly_class_name`. A model-visible message must never name a `Safe*` dataclass or quote a Python internal like `unhashable type: 'dict'`.
- Emit **concrete next-step guidance** for the known traps: imports beyond `json`/`math`/`re` (→ built-ins), `%`/`str.format()` formatting (→ f-string), and missing attributes (surface the object's public surface).

### Accommodate every documented HA idiom
Registry facades (`facade_views.py`) accept the full HA-native surface under both short (`er`) and long (`entity_registry`) names as one object. `async_get` dispatches by argument shape; traversal methods accept both the two-arg HA form and the clean one-arg form. Do not add validation that rejects a legitimate HA call shape — adapt it internally instead.

### Recorder tools: scope and size naturally
Because the sandbox forbids `timedelta`/arithmetic, "last N hours" is expressed as a tool **input**, never computed in code. Every recorder tool accepts:
- A relative `hours=<n>` window, plus ISO `start`/`end` (`_clamp_window` in `tools/recorder.py`).
- HA-native scoping selectors `area_id`/`device_id`/`floor_id`/`label_id`/`domain` resolved against the snapshot indexes (`_resolve_entity_ids`). Explicit IDs that are not visible name themselves in the error; selectors expand to the visible set.

### Errors carry actionable hints
Recoverable tool errors (`tool_error_envelope` in `errors.py`) include `message` + `hints` (concrete next steps), not just a stable key. Stable keys remain translated in `translations/en.json` for the human contract; `hints`/`message` are the LLM-facing remediation that targets success on the next call.

### Where new forgiveness lives
New accommodation belongs as code in this package (a normalization pass, an argument-shape dispatch, or a refinement case), keeping `prompts.py` to a statement of available surface — not a list of integration-specific rules the LLM must obey.
