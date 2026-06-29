# Changelog

## 0.1.0

- Initial `llm_sandbox` custom integration scaffold.
- Added config flow and options flow for execution timeout and helper-call budget.
- Added entry-scoped Home Assistant LLM API registration.
- Added `execute_home_code` Monty tool with HA-style snapshot facades.
- Added read-only state, registry, area, floor, device, and service-catalog facades.
- Added propose-only `hass.services.async_call` collection through `proposed_actions`.
- Added tests and validation scripts for linting, typing, YAML, and pytest.
