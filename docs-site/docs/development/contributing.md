---
title: Contributing
description: Contribution expectations for code, tests, docs, and safety boundaries.
---

# Contributing

Start with the repository's [`CONTRIBUTING.md`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/CONTRIBUTING.md).

## Expectations

- Keep the Monty boundary intact.
- Do not pass live Home Assistant objects, registries, service handles, event bus, auth, config, filesystem, network, or OS/process APIs into Monty.
- Build a fresh snapshot for each `execute_home_code` call.
- Keep README, docs, translations, manifest, services, diagnostics, and tests aligned when behavior changes.
- Add or update tests for meaningful behavioral changes.

## Documentation changes

Docs source lives under `docs-site/docs/`. Build before opening a docs PR:

```bash
cd docs-site
pnpm install
pnpm build
```
