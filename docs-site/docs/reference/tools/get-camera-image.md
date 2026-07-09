---
title: get_camera_image
description: Reference for camera and image entity capture.
---

# `get_camera_image`

Captures a visible `camera.*` or `image.*` entity and returns an inline image result for a multimodal model.

## Inputs

| Field | Meaning |
| --- | --- |
| `entity_id` | Camera or image entity to capture. |
| `target_width` | Optional output width from 512 to 1920 pixels. Defaults to 1280. |
| `question` | Optional caption or task for the multimodal model. |

## Bounds

Images are downscaled and normalized to JPEG. The returned attachment must fit within the configured image attachment budget.

## Source

[`tools/vision.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/vision.py).
