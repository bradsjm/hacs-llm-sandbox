---
title: Model Quality And Cost
description: Why model choice affects reliability, latency, and token usage.
---

# Model Quality And Cost

Assist Agent Sandbox shifts work from fixed intents to an LLM that chooses tools and writes short code. The model matters.

## Quality

A strong model is more likely to:

- Choose the correct tool.
- Write valid Python for the facade surface.
- Recover from structured guidance.
- Avoid over-broad action plans.
- Interpret image results correctly when using cameras.

## Cost

Tool definitions, prompts, snapshot data, recorder rows, and image attachments can increase per-conversation cost. Visibility settings and prompt profiles are the main knobs for reducing unnecessary context.

## Recommendation

Start with read-only usage and a capable model. Watch tool calls and outputs before enabling actions.
