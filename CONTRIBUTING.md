# Contributing

## Setup

```bash
scripts/setup
```

## Validate changes

```bash
scripts/check
```

The full check runs Ruff, mypy, yamllint, and pytest.

## Local Home Assistant testing

```bash
scripts/run-docker
```

After changes under `custom_components/llm_sandbox`, restart Home Assistant:

```bash
docker restart home-assistant
```

## Pull request expectations

- Keep the Monty sandbox boundary intact: safe snapshot-derived facades only.
- Add or update tests for changed behavior.
- Keep README, translations, manifest metadata, scripts, and tests aligned with source changes.
- Run `scripts/check` before submitting.
