# LLM Sandbox action evals

## Scope

`llm_sandbox_evals/` is a development-only action-capability harness. It asks a
real Pydantic AI agent to use the production LLM tools against fresh frozen Home
Assistant fixtures. It evaluates only successful service invocation effects.

## Current contract

- Cases contain only `id`, `home`, `user_request`, and nonempty
  `expected_actions`.
- Expected actions contain `domain`, `service`, required nonempty
  `target_entity_ids`, and optional `service_data`.
- Runtime actions are always enabled without a domain allowlist.
- Agent output is plain text (`Agent[EvalRuntime, str]`). Prose is display-only
  and is never parsed or scored.
- Correctness is exact multiset equality between expected actions and successful
  ledger effects. Missing, wrong, extra, and duplicate successes fail.
- Rejected records are diagnostics only. Action results preserve normalized
  observed effects, one-to-one dimension comparisons, unexpected actions, and
  stable reason codes without weakening exact matching.
- Traces and reports are scoring version 5. Reject v4 and older artifacts with
  no compatibility path.

Do not add read scoring, evidence normalization, answer schemas, collections,
aggregates, relations, no-data, recorder scoring, blocked actions, or
conditional action behavior to this baseline.

## Dataset and stub

The dataset is exactly four `home_minimal` cases: bedroom and living-room light
turn-on/turn-off. The stub supports exactly those four requests, calls
`execute_home_code`, and then emits plain text.

The homes package contains only `home_minimal` for the current two-light action
surface and `home_full` for the complete 288-entity inventory fixture.

## Architecture

- `schema.py` — minimal case, action, trace, ledger, diagnostic, and outcome
  records.
- `data/cases.yaml` / `cases_schema.json` — four direct action cases and their
  focused authoring schema.
- `agent_runner.py` — plain-text agent, production tool registration, and the
  four-route offline stub.
- `runtime.py` — fresh fixture runtime with actions enabled.
- `tools.py` — visibility scoping, non-live `RecordingInvoker`, and compact
  action normalization.
- `scoring/actions.py` / `scoring/evaluate.py` — successful-ledger construction
  and exact effect matching.
- `harness.py` — lifecycle, tool events, action extraction, minimal successful
  tool diagnostics, and trace assembly.
- `experiment.py`, `reports.py`, `terminal.py`, `html_report.py` — overall model
  comparison, v5 persistence, diagnostics, and action-ledger display without
  category analysis.

Production read tools may remain registered because they are part of the product
surface. Their eval-specific scoring and stub routes must not return.

## Staged future work

Expand only through explicit plans and observable action contracts: service
data, explicit multi-target actions, selector resolution, conditional actions,
then policy rejection. Read-answer scoring requires a separate design rather
than restoration of v4 models or evidence modules.

## Safety and commands

- Build a fresh snapshot for every cell.
- Never expose live Home Assistant objects, registries, services, recorder
  databases, filesystem, network, or OS/process APIs to Monty.
- `RecordingInvoker` is the only action seam and never dispatches live calls.
- Keep eval dependencies in the `evals` group and do not modify
  `custom_components/` for eval-only changes.

```text
scripts/setup-evals
scripts/format-evals
scripts/check-evals
scripts/yaml-check
scripts/markdown-check
scripts/check
```

Use Python >=3.14.2, Ruff `py314`, strict mypy, typed test parameters, and
behavioral assertions on action effects rather than implementation choreography.
