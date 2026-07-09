---
title: Visibility
description: How visibility settings shape the snapshot that the model can inspect.
---

# Visibility

Visibility settings decide what records are included in the fresh snapshot for each tool call. They reduce model exposure and token usage, but they are not the primary isolation boundary.

| Option | Default | Effect |
| --- | --- | --- |
| Restrict to Assist-exposed entities | On | Includes only entities you have exposed to Assist. |
| Exclude hidden entities | On | Drops hidden entity-registry entries. |
| Exclude configuration entities | On | Drops `config`-category entities. |
| Include all diagnostic entities | Off | When off, keeps only useful diagnostic device classes. |

Snapshot filtering is implemented in [`snapshot/builder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/builder.py). Snapshot records are frozen dataclasses from [`snapshot/models.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/models.py).

## Important boundary

Visibility is filtering, not a hard security boundary. The hard runtime boundary is that Monty receives only frozen JSON-compatible facade data and cannot access live Home Assistant objects, filesystem, network, event bus, auth, or process APIs.

## Practical recommendation

Keep Assist exposure limited to entities you are comfortable sending to your model provider. Entity state attributes are visible when the entity is visible.
