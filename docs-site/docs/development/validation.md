---
title: Validation
description: Commands for validating integration code and documentation.
---

# Validation

## Integration checks

From the repository root:

```bash
scripts/check
```

Focused commands include:

```bash
scripts/lint-check
scripts/type-check
scripts/yaml-check
scripts/test
```

## Documentation checks

From `docs-site/`:

```bash
pnpm install
pnpm build
pnpm serve
```

The build verifies Docusaurus config, sidebar IDs, internal links, and static output generation for the configured GitHub Pages base URL.

## CI

The integration validation workflow lives in [`.github/workflows/validate.yml`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/.github/workflows/validate.yml). The docs Pages workflow lives in [`.github/workflows/docs.yml`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/.github/workflows/docs.yml).
