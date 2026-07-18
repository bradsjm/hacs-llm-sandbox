# Changelog

## [Unreleased]

### Added

- Added the optional `get_energy` tool and `await hass.energy(...)` facade for dashboard-configured Energy queries
- Shared one bounded, visibility-safe Energy query core across direct, Monty, and eval execution
- Extended prompts, documentation, translations, fixtures, and focused production/eval coverage for Energy routing and output contracts

## 0.1.0

- Initial `llm_sandbox` custom integration scaffold.
- Added config flow and options flow for execution timeout and service-call limit.
- Added entry-scoped Home Assistant LLM API registration.
- Added `execute_home_code` Monty tool with HA-style snapshot facades.
- Added read-only state, registry, area, floor, device, and service-catalog facades.
- Added propose-only `hass.services.async_call` collection through `proposed_actions`.
- Added tests and validation scripts for linting, typing, YAML, and pytest.
