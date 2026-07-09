---
title: Vision Tool
description: How camera and image entity capture works.
---

# Vision Tool

The vision tool is implemented in [`tools/vision.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/vision.py).

## Flow

1. Validate input and loaded integration state.
2. Build a fresh vision snapshot.
3. Confirm the requested entity is visible and belongs to the `camera` or `image` domain.
4. Capture the image through Home Assistant camera/image support.
5. Normalize and downscale to JPEG off the event loop.
6. Return an inline multimodal tool result.

## Boundary

Image bytes are not passed into Monty. The tool returns an image payload directly to the LLM tool caller.
