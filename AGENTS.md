# Repository Instructions

## Project Identity

This repository contains the `llm_sandbox` Home Assistant custom integration. It provides a Monty-backed LLM API tool for read-only Home Assistant inspection and propose-only service-call collection.

## Non-Negotiables

- Never pass live Home Assistant objects, live registries, service handles, event bus, config, auth, filesystem, network, or OS/process APIs into Monty.
- Build a fresh snapshot for every `execute_home_code` call.
- Keep Monty-visible objects safe, JSON-compatible, and derived from snapshot records.
- Preserve HA-native read API shapes where practical (`hass.states`, `er.async_get(hass)`, registry instance methods).
- Keep service calls propose-only unless a future task explicitly designs and tests a live-action boundary.
- Do not add semantic index, query, enrichment, timeseries, or panel code to this MVP repo.
- Store per-entry runtime state on typed `entry.runtime_data`.
- Register LLM APIs and unload callbacks through Home Assistant lifecycle helpers.

## Commands

- Setup: `scripts/setup`
- Full check: `scripts/check`
- Lint: `scripts/lint-check`
- Type check: `scripts/type-check`
- YAML: `scripts/yaml-check`
- Tests: `scripts/test`
- Format: `scripts/format`
- Dev container: `scripts/run-docker`

## Testing Expectations

- Cover user-visible behavior: config flow, setup/unload, snapshot conversion, facade reads, Monty execution, await normalization, result binding, and propose-only action collection.
- Tests should not assert incidental prose or implementation choreography when stable behavior can be asserted instead.
- If service-call behavior changes, prove whether a real Home Assistant service call does or does not fire.

## Code Style

- Target Python `>=3.14.2`.
- Prefer direct, simple implementations over compatibility shims.
- Keep public docs, `manifest.json`, `hacs.json`, translations, scripts, and tests aligned with source behavior.
- Run `scripts/check` before considering work complete.
