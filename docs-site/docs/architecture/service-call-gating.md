---
title: Service Call Gating
description: Policy checks applied before any live service action is dispatched.
---

# Service Call Gating

Service calls are exposed through the facade in [`facades/services.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/facades/services.py). Live dispatch is private and created by [`tools/code.py`](https://github.com/bradsjm/hacs-llm-sandbox/blob/main/custom_components/llm_sandbox/llm_api/tools/code.py).

## Gates

The facade checks:

- Action master switch.
- Domain allowlist.
- Service catalog membership.
- Response-mode compatibility.
- Target visibility and resolution.
- Service-call limit and execution deadline.

## Policy blocks

Policy blocks record errored action records and return `None` to the sandbox code. They do not turn the whole run into a code failure.

After validation passes, a call consumes the service-call limit immediately before the private live invocation. Reads and pre-dispatch blocks do not consume the limit; dispatched failures and timeouts do.

## Live failures

Exceptions from the live service call path are treated as tool errors after an action record is captured.
