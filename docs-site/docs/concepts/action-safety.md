---
title: Action Safety
description: Why service calls are opt-in and how they are constrained.
---

# Action Safety

Actions are powerful because they can operate real devices. Assist Agent Sandbox keeps them off by default and checks every call when they are enabled.

## Safety model

1. The model can only call services through the sandbox facade.
2. The facade validates the call against the fresh snapshot and action settings.
3. Live dispatch happens only through a private runtime invoker.
4. The live Home Assistant object is never handed to Monty.

## Visible targets

Service targets must resolve through visible snapshot data. If a target is not visible or is ambiguous, the tool records structured guidance rather than widening scope.

## Recommendation

Use actions only after the read-only experience is working well. Keep allowed domains as narrow as practical.
