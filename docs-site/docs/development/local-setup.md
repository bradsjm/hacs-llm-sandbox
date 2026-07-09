---
title: Local Setup
description: Set up the repository for contribution and local Home Assistant testing.
---

# Local Setup

The repository's contributor workflow is documented in [`CONTRIBUTING.md`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/CONTRIBUTING.md).

## Python environment

From the repository root:

```bash
scripts/setup
```

## Full repository check

```bash
scripts/check
```

## Local Home Assistant container

```bash
scripts/run-docker
```

Restart the container after changing integration code so Home Assistant reloads the custom component.

## Docs site setup

From the docs site folder:

```bash
pnpm install
pnpm build
pnpm serve
```
