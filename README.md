# Assist Agent Sandbox

[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-%3E%3D2026.6.4-41BDF5.svg)](https://www.home-assistant.io/)
[![HACS](https://img.shields.io/badge/HACS-%3E%3D2.0.0-41BDF5.svg)](https://hacs.xyz/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.14.2-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/github/license/bradsjm/hacs-llm-sandbox.svg)](LICENSE)

**Assist Agent Sandbox** supercharges Home Assistant's **Assist** with a set of tools that let the AI read your home, reason over it, and answer the kinds of questions a single built-in intent never could.

Instead of only triggering hard-coded scenes or one service at a time, the assistant can look across your whole home — current states, entity history, long-term statistics, the activity logbook, and even a live camera frame — write a small bit of logic, and give you a real answer or take a precise action.

It does all of this inside a [**minimal, secure Python interpreter written in Rust for use by AI**](https://github.com/pydantic/monty) that only ever sees a frozen, filtered copy of your home. It is **read-only by default**, and every safety boundary is opt-in.

> This is an early **0.1.0** release. Treat it as a preview.

---

## What it gives Assist

Once enabled, the assistant can call these tools during a conversation. Recorder-backed tools are offered only when Home Assistant's recorder integration is available; `get_logbook` also requires logbook runtime data.

| Tool | What it's for |
| --- | --- |
| **`execute_home_code`** | The assistant writes and runs a short Python snippet to read and reason over your home — current states, the entity/device/area/floor/label registries, repairs, persistent notifications, secret-stripped config entries, and read-only SQL via `await hass.query(...)` over visible snapshot states plus bounded recorder history/statistics. |
| **`get_history`** | Recorded **state history** — raw changes up to 24 hours, legacy summaries such as transitions and time-in-state, or declarative analytics (`aggregate`, `group_by`, `bucket`, `where`, `order_by`, `limit`) up to 30 days. |
| **`get_statistics`** | Pre-aggregated **long-term statistics** (`mean`, `min`, `max`, `state`, `sum`) over a period. Up to 30 days. |
| **`get_logbook`** | The **activity timeline** — what happened and why (e.g. "did the front door open after midnight?"). Up to 24 hours. |
| **`get_camera_image`** | Captures a **live frame** from a camera or image entity so a multimodal model can look at it ("what's on the front porch right now?"). |

## Why you'd install it

These tools turn Assist from a voice remote into something that actually *understands* your home. Things that become possible:

- **Cross-device reasoning** — "Which lights are on in the living room, and what's drawing the most power right now?"
- **Trend questions** — "What's the average bedroom humidity over the last day, and is it trending up?"
- **Timeline questions** — "Did anyone open the garage after 10pm last night? Show me what happened."
- **Discovery** — "List every Zigbee device assigned to the wrong room."
- **Visual checks** — "Is there a package at the front door?" (with a multimodal model + camera)
- **Diagnostics** — "What repairs are currently open, and are any critical?"

The assistant figures out the answer itself — there's nothing for you to script per question.

## Requirements

- **Home Assistant 2026.6.4 or newer**, running on **Python 3.14.2+**.
- **A conversation agent that supports tools**, set as your Assist agent (e.g. OpenAI, Google Gemini, Anthropic, or a capable local model). This integration provides the tools; your agent provides the brain.
- **A strong, tool-calling model.** Because the assistant writes Python and decides which tool to use, model quality is the single biggest factor in how well this works. Cloud models cost money per call.
- **Recorder enabled** for `get_history` / `get_statistics`, and **Logbook enabled** for `get_logbook`. Both are on by default in Home Assistant.
- For **`get_camera_image`**, a camera or image entity plus a **multimodal model** that can actually interpret the image.

## Installation

1. In HACS, add this repository as a **custom repository** (type: *Integration*).
2. Find **Assist Agent Sandbox** and install it.
3. Restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration** and search for **Assist Agent Sandbox**.
5. Confirm the name and assistant scope (it attaches to your default `conversation` assistant).

## Enable for your conversation (Critical Step!)

Make sure your **conversation agent exposes these tools**: open your agent's settings (e.g. the OpenAI Conversation entry), and enable the **Assist Agent Sandbox** tool set so the model is allowed to call it.

## Configuration

Open the integration's **Configure** dialog. Options are grouped into four sections.

**Visibility restrictions** — what the sandbox can see.

| Option | Default | Meaning |
| --- | --- | --- |
| Restrict to Assist-exposed entities | On | Only entities you've exposed to Assist are visible. |
| Exclude hidden entities | On | Entities marked hidden in the registry are dropped. |
| Exclude configuration entities | On | `config`-category entities are dropped. |
| Include all diagnostic entities | Off | When off, only diagnostic entities with useful device classes are included. When on, every diagnostic entity is included. |

**Action restrictions** — whether the assistant can *do* things, not just read.

| Option | Default | Meaning |
| --- | --- | --- |
| Enable actions on visible entities | **Off** | Master switch. When off, every service call is rejected — the assistant is read-only. |
| Allowed service domains | Empty | When actions are on, restrict calls to specific domains (e.g. `light`, `switch`). Leave empty to allow all. |

**Execution limits** — runaway protection.

| Option | Default | Range |
| --- | --- | --- |
| Maximum execution time | 12 seconds | 3–30 s |
| Maximum service calls per request | 32 | 1–100 |

**Prompt** — the base instructions sent to the model. Ships with **Standard**, **Terse**, and **Minimal** profiles.

## How it works

The safety model rests on two ideas: a **frozen snapshot** and an **isolated sandbox**.

1. **Every request builds a fresh snapshot.** When the assistant calls a tool, the integration takes a point-in-time copy of your home's current states, all registries (entities, devices, areas, floors, labels), the service catalog, repairs, persistent notifications, and config entries. That snapshot is then narrowed by your visibility settings so only what you've allowed is included.

2. **Code runs inside Monty, an isolated Python sandbox.** The assistant's snippet never touches the *live* Home Assistant object. It only receives safe, frozen copies built from the snapshot. Read-only SQL is executed against a per-run in-memory SQLite database populated from that snapshot and bounded recorder rows. The sandbox has **no access** to your filesystem, the network, the event bus, authentication, your configuration files, or any OS/process APIs. Imports are limited to `json`, `math`, and `re`.

3. **Reads are always allowed; control is gated.** Reading state and history works out of the box. Calling services (turning things on/off) is **off by default**. When you enable it, every single call is re-checked against that request's snapshot: the domain must be allowed, the target must be visible, and it must fit within the call budget. Calls run through a private path that never hands the live Home Assistant object to the sandbox.

4. **A forgiveness layer fixes common mistakes.** The assistant's code passes through a normalization step that silently repairs harmless variations — a missing `await`, an imported `datetime`, a forgotten `result =` — so the assistant succeeds on the first try instead of burning retries (and tokens) on trivial errors. When the tool has already offered or applied entity-id guidance in the same conversation, it can also prefer that still-visible entity in later resolution and transparently report remembered literal rewrites in `resolutions`.

### Read-only SQL queries

`execute_home_code` can call `await hass.query(sql, hours=N)` to run read-only SQLite over a fresh per-run in-memory database, not Home Assistant's live recorder database. It exposes visible `states` plus bounded recorder `history` and `statistics`; `states.attributes` is JSON text queryable with SQLite JSON functions such as `json_extract()`. History and statistics rows load only when referenced, and their scope can be narrowed with `entity_ids` or HA-native selectors (`area_id`, `floor_id`, `device_id`, `label_id`, `domain`). For discoverability, the in-memory database also exposes recorder-compatible views: `states_meta`, `statistics_meta`, `statistics_short_term`, `state_history`, and `long_term_statistics`. It does not expose registry tables; use the registry facades or the denormalized state/history columns (`area_id`, `floor_id`, `device_id`, `domain`) for location/entity filtering.

## Things to know before you install

- **You need a capable model.** This integration lives or dies by model quality. A weak model will write broken code, pick the wrong tool, or misread results. Budget for a strong cloud model, or run a strong local model.
- **It costs tokens.** Each turn that uses these tools sends tool definitions, snapshots, and (for history/cameras) potentially large payloads to your model provider. Expect higher per-conversation cost than plain Assist.
- **Attribute values are exposed.** Beyond stripping credentials out of config-entry data, **no value redaction is performed.** If any visible entity carries sensitive data in its attributes (codes, tokens, personal info), the model will see every value. Use the visibility restrictions — and your Assist exposure settings — to keep sensitive entities out of scope.
- **Visibility is filtering, not a hard security boundary.** The visibility settings reduce *what's exposed* to the model; the actual isolation boundary is the Monty sandbox (no filesystem, network, live objects, or OS access). Don't rely on visibility toggles alone for sensitive environments.
- **Actions are powerful — keep the allowlist tight.** If you turn actions on, the assistant can operate real devices. Restrict it to the domains you're comfortable with, and remember a capable model can chain many calls within one turn (bounded only by the call budget).
- **The snapshot is frozen mid-request.** A service call made inside a code run won't be reflected in that same run's reads — the assistant is told this and will call the tool again to observe new state.
- **One entry per assistant.** The integration is scoped to your default `conversation` assistant; only one sandbox entry is supported.

## Development

This is a community integration. To contribute, set up a dev environment, or report issues, see [CONTRIBUTING.md](CONTRIBUTING.md) and the [issue tracker](https://github.com/bradsjm/hacs-llm-sandbox/issues).

## Support

- [Documentation & source](https://github.com/bradsjm/hacs-llm-sandbox)
- [Report an issue](https://github.com/bradsjm/hacs-llm-sandbox/issues)
- [Changelog](CHANGELOG.md)
- Licensed under the [LICENSE](LICENSE).
