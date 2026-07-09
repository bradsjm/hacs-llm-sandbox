---
title: Quickstart
description: First prompts to confirm the integration is working.
---

# Quickstart

After installing and enabling the tool set in your conversation agent, try questions that require cross-entity reasoning.

## Read current state

```text
Which lights are currently on, grouped by area?
```

Expected behavior: the model should call `execute_home_code`, inspect `hass.states`, and summarize visible light entities.

## Ask about history

```text
Did any exterior doors open after midnight? Summarize the timeline.
```

Expected behavior: the model may call `get_history`, `get_logbook`, or `execute_home_code` with `await hass.history(...)`, depending on the agent.

## Ask for a statistic

```text
What was the average bedroom humidity over the last 24 hours?
```

Expected behavior: the model should use recorder-backed statistics if matching statistic IDs are visible and available.

## Ask for a visual check

```text
Look at the front porch camera. Is there a package visible?
```

Expected behavior: the model should call `get_camera_image`. The model must be multimodal to interpret the image.

## If nothing happens

If the model answers without using tools, return to [Enable in Assist](../installation/enable-in-assist.md) and confirm the tool set is enabled in the conversation agent.
