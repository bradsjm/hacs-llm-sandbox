---
title: Privacy And Security
description: What data can reach the model and which boundaries protect Home Assistant.
---

# Privacy And Security

Assist Agent Sandbox reduces model exposure through visibility filtering and protects Home Assistant through the Monty sandbox and private runtime boundaries.

## What can reach the model

Visible snapshot records can include entity states, attributes, registry metadata, repairs, persistent notifications, service catalog metadata, recorder rows, logbook entries, and camera images requested by a tool call. For `hass.logbook(...)`, activity entries enter Monty only as copied JSON-safe dictionaries through a private runtime seam.

Beyond omitting config-entry data/options/runtime/subentries, the integration does not promise broad value redaction. If a visible entity carries sensitive data in its state or attributes, the model may see it.

## What does not enter Monty

Monty does not receive live Home Assistant objects, live registries, event bus, auth, config files, filesystem, network, or OS/process APIs.

## Visibility is not the hard boundary

Visibility filtering is important for privacy and cost, but the runtime security boundary is the isolated sandbox plus frozen facades. Treat Assist exposure settings as a privacy control, not as the only safety mechanism.

## Recommendations

- Expose only entities you are comfortable sending to your model provider.
- Keep actions disabled until read-only behavior is reliable.
- If actions are enabled, use a narrow domain allowlist.
- Avoid exposing entities whose attributes contain secrets, codes, tokens, or highly sensitive personal data.
