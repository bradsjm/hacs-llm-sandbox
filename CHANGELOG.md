# Changelog

## [Unreleased]

### Improve eval harness lifecycle and presentation

- Bumped `llm_sandbox_evals` artifacts to scoring v6 with persisted model
  variants, strict legacy rejection, separate action reasons and operational
  causes, and corrected native usage metrics.
- Added atomic run manifests and typed cancellation/failure journals that are
  explicitly not reports, plus pre-flight matrix validation.
- Added Auto TTY / `--machine` output behavior, a durable terminal final with
  no simulated activity phase, and report-model-driven HTML/CSV presentation.

### Restore safe configuration minima

- Restored the minimum execution timeout to 3 seconds and the minimum camera/image target width to 384 pixels.

### Improve read-only facade and eval contracts

- Added immutable mapping-style reads to snapshot records and `llm_context` in Monty.
- Kept print-only executions as `output: null` with captured `printed` lines, and made eval structured checks use only execute result output.

### Expose persistent notifications

- Added a `persistent_notifications` Monty global for reading active Home Assistant persistent notifications.
- Read persistent notifications from the notification store instead of the visibility-filtered state machine.

### Perform service calls instead of proposing them

- Changed `hass.services.async_call` from propose-only collection to live Home Assistant service invocation when actions are enabled.
- Captured service call errors for the LLM so it can recover from failed calls.
- Counted only validated service calls dispatched to Home Assistant toward the per-request limit, including dispatched failures and timeouts.
- Added valid-service listings with brief parameter schemas when a service name is wrong.
- Replaced `proposed_actions` tool output with per-call `actions` outcomes.
- Added translation keys for hidden targets, timed-out service calls, and failed service calls.

## 0.1.0

- Initial `llm_sandbox` custom integration scaffold.
- Added config flow and options flow for execution timeout and service-call limit.
- Added entry-scoped Home Assistant LLM API registration.
- Added `execute_home_code` Monty tool with HA-style snapshot facades.
- Added read-only state, registry, area, floor, device, and service-catalog facades.
- Added propose-only `hass.services.async_call` collection through `proposed_actions`.
- Added tests and validation scripts for linting, typing, YAML, and pytest.
