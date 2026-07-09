---
title: Troubleshooting
description: Common setup, tool availability, recorder, camera, and action issues.
---

# Troubleshooting

## The model never calls the tools

Confirm the `Assist Agent Sandbox` tool set is enabled in your conversation agent settings. Installing the integration is not enough.

## Recorder tools are missing

`get_history` and `get_statistics` are exposed only when recorder support is available. `get_logbook` also requires logbook runtime data. Confirm recorder and logbook are enabled in Home Assistant.

## A visible entity is missing

Check the integration visibility settings and Home Assistant Assist exposure. Hidden entities, config-category entities, and many diagnostic entities are filtered by default.

## Camera image capture fails

Confirm the entity is visible, belongs to `camera` or `image`, and can provide an image in Home Assistant. Confirm your model is multimodal.

## Service calls are blocked

Check whether actions are enabled, whether the domain is allowed, and whether the target entity is visible. Policy blocks are reported as action errors rather than Python crashes.

## Code errors repeat

Use a stronger model or a more explicit prompt. The executor includes normalization and guidance, but the model still has to choose the right follow-up.
