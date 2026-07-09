---
title: Upgrades
description: How to approach updates while the integration is early preview software.
---

# Upgrades

Assist Agent Sandbox is currently an early `0.1.0` preview. Treat upgrades like any other Home Assistant custom integration update.

## Before upgrading

- Read [`CHANGELOG.md`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/CHANGELOG.md).
- Review changes to action behavior before using actions.
- Keep a note of your current visibility, action, execution-limit, and prompt settings.

## After upgrading

- Restart Home Assistant if HACS or Home Assistant requires it.
- Confirm the integration loads.
- Confirm the tool set is still enabled in your conversation agent.
- Run a read-only quickstart prompt before trying service actions.
