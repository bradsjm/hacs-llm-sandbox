# LLM Sandbox

LLM Sandbox is a Home Assistant custom integration that exposes Assist LLM API tools for running bounded Python/Monty code and bounded recorder history, long-term statistics, and logbook queries against fresh visibility-filtered Home Assistant snapshots.

The sandbox never receives the live `hass` object, live registries, the event bus, auth, config, filesystem, network, or OS/process APIs. It receives safe facade objects built from snapshot records.

Service calls through `hass.services.async_call(...)` are read-only by default. When actions are enabled, each call is validated against the snapshot (domain allowlist, service catalog, target visibility, response mode) and then dispatched live through a private invoker that keeps the live callable and the real Home Assistant context out of Monty. Per-call outcomes are returned in `actions`.

## Capabilities

- Config flow for one `llm_sandbox` entry scoped to the default `conversation` assistant.
- LLM API tools: `execute_home_code`, `get_history`, `get_statistics`, and `get_logbook`.
- `get_history` reads bounded recorder state history for visible entities.
- `get_statistics` reads bounded long-term recorder statistics for visible entity-backed statistic IDs.
- `get_logbook` reads bounded logbook events for visible entities.
- Fresh per-call snapshot of states, entity/device/area/floor registries, and the service catalog.
- HA-style read globals in Monty: `hass`, `states`, `er`, `dr`, `ar`, `fr`, `lr`, `cr`, `entity_registry`, `device_registry`, `area_registry`, `floor_registry`, `label_registry`, `category_registry`, `repairs`, `config_entries`, `date`, `datetime`, `now`, and `llm_context`. The `date` and `datetime` globals are frozen snapshot-backed facades.
- Label registry snapshot exposed via `lr` / `label_registry` (read-only, list and by-name lookup).
- Category registry snapshot exposed via `cr` / `category_registry` (read-only, scope-keyed list and id lookup).
- Repairs issues (issue registry) exposed via `repairs` (read-only list, filtered by domain/severity/active/dismissed).
- Config entries exposed via `config_entries` with credentials (data/options) stripped; lookup by entry_id and filter by domain.
- `llm_context` includes the initiating `device_id` plus derived `area_id`, `area_name`, `floor_id`, and `floor_name` when Home Assistant provides a satellite device.
- Live service calls through `await hass.services.async_call(...)` when actions are enabled; per-call outcomes are returned in `actions`.
- Action safety controls: `actions_enabled` gates all service calls, `action_domains` restricts controllable domains, targets must be visible to the sandbox, and real Home Assistant context is used for attribution.
- Options for execution timeout and helper-call budget.

## Recorder tools

`get_history` returns recorded state history for one or more visible entities. Pass `entity_ids` and optional ISO-8601 `start`/`end` timestamps; omitted timestamps default to the last hour. History windows are capped at 24 hours.

`get_statistics` returns long-term recorder statistics for one or more visible entity-backed statistic IDs. Pass `statistic_ids`, optional ISO-8601 `start`/`end` timestamps, and optional `period` (`5minute`, `hour`, or `day`; default `hour`). Statistics windows default to 24 hours and are capped at 30 days.

`get_logbook` returns logbook events for one or more visible entities. Pass `entity_ids` and optional ISO-8601 `start`/`end` timestamps; omitted timestamps default to the last 24 hours. Logbook windows are capped at 24 hours.

All recorder-tool windows are UTC. Results use ISO-8601 timestamps and row caps, with `truncated` set when older rows were omitted.

Recorder tools are always registered and are gated at call time. They return `recorder_unavailable` when recorder is not running, `entity_not_visible` when a requested entity is outside the sandbox's fresh per-call snapshot, `time_window_too_large` when the requested window exceeds the cap, `logbook_unavailable` when logbook is not running, and `query_failed` when a recorder query fails unexpectedly.

Out of scope: beyond secret-stripping config-entry credentials, no attribute-value redaction is applied — the model sees every value in the visible snapshot. No non-Home-Assistant helper globals are exposed.

## Tool behavior

`execute_home_code` accepts one argument:

```json
{"code": "result = hass.states.get('light.kitchen').state"}
```

It returns:

```json
{
  "execution": {"status": "ok"},
  "output": "on",
  "printed": [],
  "actions": []
}
```

`execution.status` can be `ok`, `code_error`, `helper_error`, or `setup_error`.
Execution timeouts are returned as `code_error` with `kind` set to `TimeoutError`.
Service call errors are captured and returned to the LLM so it can recover. If a service name is wrong, the response includes the valid services for that domain plus brief parameter schemas.
Every `execution` object also reports `helper_calls`/`helper_call_limit` and the forgiveness-layer `normalizations` applied to the code; `code_error` and `helper_error` payloads additionally list `available_globals` and `suggested_methods`.

Example action:

```py
await hass.services.async_call(
    "light",
    "turn_on",
    {"brightness_pct": 80},
    target={"entity_id": "light.bedroom"},
)
result = "called"
```

When actions are enabled and the domain and target are allowed, Home Assistant runs the service call. The outcome is returned in `actions`.

## Forgiveness layer

`execute_home_code` runs the submitted code through a fail-open AST normalization pipeline before Monty type-checks it. Each pass is independent, returns the original code unchanged on any failure, and records the labels it applied in `execution.normalizations` so the model can see what was rewritten:

- **datetime imports** — `from datetime import datetime`/`date` and `import datetime` (`as alias`) resolve to the frozen `date`/`datetime` facade globals. A locally shadowed alias (e.g. a `dt` parameter) is skipped so the real import surfaces a natural error.
- **builtin reflection** — `getattr`/`hasattr` with literal names and `type(x).__name__` over known facade globals are statically resolved to direct attribute access, without enabling dunder walking or dynamic resolution.
- **await** — missing `await` on known async facade methods is added; `await` over a provably-synchronous facade chain is stripped; and state-machine sugar (`states['light.x']`, `'light.x' in states`, `len(states)`) is rewritten to public method calls. Async/sync classification is derived from the facade dataclasses, not a hand-maintained list.
- **result binding** — a trailing bare expression is promoted to `result = ...`, and an explicit `result` assignment is returned as the tool output.

This keeps common, harmless variations (a missing `await`, `from datetime import datetime`, a forgotten `result =`) from forcing retry loops.

## Development

Install the locked development environment:

```bash
scripts/setup
```

Run checks:

```bash
scripts/check
```

Focused commands:

```bash
scripts/lint-check
scripts/type-check
scripts/yaml-check
scripts/test
scripts/format
```

Run a development Home Assistant container:

```bash
scripts/run-docker
```

After changing files under `custom_components/`, restart the container:

```bash
docker restart home-assistant
```
