# Repository Instructions

## Project Identity

This repository contains the `llm_sandbox` Home Assistant custom integration. It provides a Monty-backed LLM API tool for read-only Home Assistant inspection and propose-only service-call collection.

## Non-Negotiables

- Never pass live Home Assistant objects, live registries, service handles, event bus, config, auth, filesystem, network, or OS/process APIs into Monty.
- Build a fresh snapshot for every `execute_home_code` call.
- Keep Monty-visible objects safe, JSON-compatible, and derived from snapshot records.
- Preserve HA-native read API shapes where practical (`hass.states`, `er.async_get(hass)`, registry instance methods).
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

- Tests must only be used for meaningful behavioral verification, asserting the expected **critical path** behavior of the component, including edge cases and error conditions.
- Tests must not be used for non-functional behavior (e.g., UI copy, translated errors, diagnostics prose, or provider-facing text).
- Avoid overly brittle tests that depend on implementation details or specific error messages.
- Reserve mocking for external dependencies (databases, networks, third-party APIs) that are slow, non-deterministic, or outside your control.
- Never create tests that pass simply because the mock is configured to return expected values.

## Testing Guidance

- Prefer assertions on user-visible behavior, persisted data, emitted events, stable error keys, and runtime side effects over assertions on constructor kwargs, mock call choreography, private helpers, import paths, or other implementation details that can change during harmless refactors.
- When changing config flows, assert flow result types, translated error keys, placeholders, created subentry data, and reconfigure behavior. Do not over-specify serialized selector structure, field ordering, or exact UI text unless that exact presentation is itself the contract being protected.
- When changing runtime behavior, cover metrics, repairs, diagnostics, system health counts, live overlay state, and cleanup/unload when relevant.
- For diagnostics, tracing, and observability tests, assert stable metadata, support-useful exposed index facts, targeted redaction, classification, and presence of key attributes rather than exact human-readable formatting such as span titles, warning prose, or incidental ordering.
- Avoid exact English string assertions for UI copy, translated errors, diagnostics prose, and provider-facing text when a stable reason key, placeholder, classification, or behavioral outcome can be asserted instead.
- When removing behavior, delete obsolete tests and docs that only validate the removed path. Do not preserve compatibility shims without a concrete need.
- When writing or modifying tests, ensure all test function parameters have type annotations.
- Prefer concrete types (for example, HomeAssistant, MockConfigEntry, etc.) over Any.
- Prefer @pytest.mark.usefixtures over arguments, if the argument is not going to be used.
- Avoid using conditions/branching in tests. Instead, either split tests or adjust the test parametrization to cover all cases without branching.
- If multiple tests share most of their code, use pytest.mark.parametrize to merge them into a single parameterized test instead of duplicating the body. Use pytest.param with an id parameter to name the test cases clearly.

## Documentation And Metadata

- Keep `README.md`, `manifest.json`, `hacs.json`, translations, `services.yaml`, `icons.json`, docs, and tests aligned when behavior changes.
- Documentation must reflect current source, not extraction plans or previous integration behavior.
- Treat manifests, translations, services, diagnostics, and tests as first-class validation surfaces.
- Provide inline comments at all branch boundaries and state mutation points.
- Add comments for non-obvious Home Assistant lifecycle or safety constraints.

## Code Style

- Target Python `>=3.14.2`, Ruff `py314`, and mypy as the canonical type checker. Do not add compatibility workarounds for older Python.
- Keep changes small and source-grounded.
- Format files after making changes using `scripts/format`.
- Prefer direct implementation over new helpers unless reuse is clear.
- Do not add backward-compatibility shims unless there is an explicit user requirement.
- Prefer explicit registries and per-type modules over growing `if`/`elif` blocks.
- Keep package `__init__.py` files as stable public surfaces; move internal logic into neighboring modules rather than growing the barrel.
- Python 3.14 explicitly allows except TypeA, TypeB: without parentheses. Never flag this as an issue.
- Python 3.14 evaluates annotations lazily (PEP 649). Forward references in annotations do not need to be quoted — annotations can reference names defined later in the module without quoting them or using from __future__ import annotations. Do not flag unquoted forward references in annotations as issues.
- When reviewing entity actions, do not suggest extra defensive checks for input fields that are already validated by Home Assistant's service/action schemas and entity selection filters. Suggest additional guards only when data bypasses those validators or is transformed into a less-safe form.
When validation guarantees a dict key exists, prefer direct key access (data["key"]) instead of .get("key") so contract violations are surfaced instead of silently masked.

## Quality Scale Alignment

- Changes should align with current Home Assistant quality-scale and developer-tooling standards unless the user explicitly asks for a repo-specific exception.
- Highest priority: preserve a Home Assistant-native config entry UX. New setup paths should go through UI flows, validate connectivity during the flow when a real backend connection exists, prevent duplicate entries with stable unique IDs, and add reauth or reconfigure flows when credentials or required setup data can change.
- High priority: use Home Assistant failure semantics during setup. In `async_setup_entry`, raise `ConfigEntryNotReady` for temporary backend outages, `ConfigEntryAuthFailed` for expired or invalid auth, and `ConfigEntryError` for permanent unsupported states instead of swallowing errors or inventing custom retry loops.
- High priority: keep domain services and responses aligned with HA expectations. Register services in `async_setup()`, not per-entry setup, validate targeted entries inside the handler, and raise `ServiceValidationError` for bad input or unloaded entries and `HomeAssistantError` for runtime failures.
- High priority: keep config entry lifecycle complete and reversible. Support unload and removal cleanly, attach listeners and background cleanup through `entry.async_on_unload()`, and keep per-entry runtime state on typed `entry.runtime_data`.
- Medium priority: match HA availability and recovery behavior. If a backend becomes unavailable, surface unavailable state where applicable, recover automatically when it returns, and avoid log spam by logging the outage once and the recovery once.
- Medium priority: test Home Assistant behavior, not just helper functions. Prefer coverage for config flows and subentry flows, setup and unload, index build and rebuild lifecycle, introspection services, diagnostics usefulness, system health, and service registration or error surfacing.
