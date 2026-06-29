# LLM Sandbox

LLM Sandbox is a Home Assistant custom integration that exposes one Assist LLM API tool, `execute_home_code`, for running bounded Python/Monty code against a fresh read-only snapshot of Home Assistant state and registries.

The sandbox never receives the live `hass` object, live registries, the event bus, auth, config, filesystem, network, or OS/process APIs. It receives safe facade objects built from snapshot records.

## Current scope

- Config flow for one `llm_sandbox` entry scoped to the default `conversation` assistant.
- One LLM API tool: `execute_home_code`.
- Fresh per-call snapshot of states, entity/device/area/floor registries, and the service catalog.
- HA-style read globals in Monty: `hass`, `states`, `er`, `dr`, `ar`, `fr`, `entity_registry`, `device_registry`, `area_registry`, `floor_registry`, `now`, and `llm_context`.
- Propose-only service calls through `await hass.services.async_call(...)`; calls are recorded in `proposed_actions` and are not executed.
- Options for execution timeout and helper-call budget.

Not included in this MVP: live service execution, exposure filtering, redaction, or non-Home-Assistant helper globals.

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
  "proposed_actions": []
}
```

`execution.status` can be `ok`, `code_error`, `helper_error`, or `setup_error`.
Execution timeouts are returned as `code_error` with `kind` set to `TimeoutError`.

Example propose-only action:

```py
await hass.services.async_call(
    "light",
    "turn_on",
    {"brightness_pct": 80},
    target={"entity_id": "light.bedroom"},
)
result = "proposed"
```

The real Home Assistant service is not called. The requested call is returned in `proposed_actions`.

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
