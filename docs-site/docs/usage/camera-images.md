---
title: Camera Images
description: Use visible camera and image entities with a multimodal model.
---

# Camera Images

`get_camera_image` captures one visible `camera.*` or `image.*` entity and returns it as an inline image result for a multimodal model.

## Example prompts

```text
Look at the driveway camera. Is a car parked there?
```

```text
Check the front porch camera for a package.
```

## Requirements

- The entity must be visible under the integration's visibility settings.
- The entity must be in the `camera` or `image` domain.
- The conversation model must be able to interpret image attachments.

## Bounds

The tool accepts `target_width` from 512 to 1920 pixels and defaults to 1280 pixels. Images are normalized to JPEG and checked against the attachment-size budget before they are returned.

The implementation lives in [`tools/vision.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/vision.py). Image bytes do not enter Monty.
