---
title: Snapshots And Visibility
description: Fresh snapshots are the data source for every tool call.
---

# Snapshots And Visibility

Every tool call builds a fresh snapshot from live Home Assistant state and registries, then filters that snapshot according to the integration settings. Monty-visible objects are built from snapshot records, not from live Home Assistant objects.

## Snapshot contents

Depending on the tool flavor, a snapshot can include:

- Current states.
- Entity, device, area, floor, label, and category records.
- Repairs and persistent notifications.
- Secret-stripped config-entry metadata.
- Service catalog and response-support metadata.
- Indexes used for selector and target resolution.

Snapshot models are in [`snapshot/models.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/models.py). Snapshot construction and filtering are in [`snapshot/builder.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/snapshot/builder.py).

## Frozen during a run

The snapshot is a point-in-time view. If a service action changes a device during a code run, later reads in that same run still reflect the original snapshot.
