---
title: State And Registry Questions
description: Ask Assist questions over current state and Home Assistant registries.
---

# State And Registry Questions

`execute_home_code` gives the model read facades for current states and registry-like data. This supports questions that require joining entities, devices, areas, floors, labels, repairs, persistent notifications, and secret-stripped config entries.

## Example prompts

```text
List every battery-powered device under 20 percent, grouped by area.
```

```text
Which entities are assigned to devices but have no area?
```

```text
What repairs are currently open, and which integration do they belong to?
```

## Available patterns

The facade intentionally mirrors familiar Home Assistant read shapes:

- `hass.states.get("light.kitchen")`
- `hass.states.async_all()`
- `hass.states.async_entity_ids("sensor")`
- `er.async_get(hass)` for entity registry access
- Registry instance methods for entity, device, area, floor, label, issue, notification, and config-entry records

The state facade is implemented in [`facades/state.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/state.py). Registry facades are implemented in [`facades/registries.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/registries.py).

## Derived join fields

Visible state records include useful registry-derived fields such as `area_id`, `floor_id`, `device_id`, `platform`, and `unique_id`. These preserve the familiar state shape while making common questions easier for a model.
