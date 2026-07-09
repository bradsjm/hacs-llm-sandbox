---
title: Snapshot Pipeline
description: How live Home Assistant data becomes frozen JSON-compatible records.
---

# Snapshot Pipeline

The snapshot pipeline converts live Home Assistant state into frozen, JSON-compatible records before Monty or a tool-specific helper sees it.

## Build variants

[`snapshot/builder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/builder.py) supports focused variants for full execution, recorder tools, and vision. Each variant includes only the records needed by that tool path.

## Visibility filtering

Visibility filters apply before records are exposed. Filtering can restrict to Assist-exposed entities, remove hidden entities, remove config-category entities, and narrow diagnostic entities.

## Safe models

[`snapshot/models.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/models.py) defines the frozen safe record dataclasses. Config-entry records intentionally omit data, options, runtime state, and subentries because those can contain credentials or internal objects.

## Indexes

Snapshot indexes support selector expansion, target resolution, and common joins without exposing live registries.
