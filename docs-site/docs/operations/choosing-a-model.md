---
title: Choosing A Model
description: Capabilities to look for in a conversation model.
---

# Choosing A Model

Assist Agent Sandbox works best with a model that is good at tool use, small Python snippets, and following structured recovery guidance.

## Required capabilities

- Home Assistant conversation-agent tool calling.
- Reliable JSON/tool argument generation.
- Basic Python reasoning.
- Ability to interpret structured outputs.

## Recommended capabilities

- Strong long-context behavior for large snapshots.
- Good recovery from tool errors and guidance.
- Multimodal image understanding if you plan to use `get_camera_image`.
- Careful planning before service actions.

## Evaluation approach

Start with read-only prompts. Watch whether the model chooses appropriate tools, writes concise code, and summarizes results accurately. Enable actions only after that behavior is predictable.
