---
title: get_automation
description: Read authorized Home Assistant automations and optional trigger entries.
---

# `get_automation`

`get_automation` is a read-only direct tool. It returns a stable collection of automation summaries sorted by `entity_id`.

## Inputs

- `query`: whitespace-tokenized search over titles, IDs, aliases, assigned metadata, and referenced objects.
- `entity_ids`: one or more `automation.*` IDs; they intersect the query.
- `include`: `content`, `runs`, or both.
- `hours`, `start`, and `end`: an optional run window. These are invalid without `include: ["runs"]`; the default is 24 hours and the maximum is 24 hours.
- `limit`: whole automation records per page, from 1 to 20, default 10.
- `cursor`: the opaque `next_cursor` from the prior response. A continuation accepts no other caller arguments.

## Authorization and projections

Summaries and runs follow the requesting Home Assistant user's entity `read` permission. The tool does not apply Assist exposure, Sandbox visibility, or custom redaction rules. Complete `content` follows Home Assistant Core's administrator-only automation configuration rule; a non-administrator content request fails rather than returning partial configuration.

`runs` are automation-triggered Logbook entries, not traces. They do not establish that conditions passed or actions succeeded. Recorder and Logbook runtime support are required only when `runs` is requested.

## Pagination

Each automation and its requested projections are one indivisible record. Pages are measured using compact UTF-8 JSON including the envelope and cursor, with a 16 KiB normal budget. A single oversized first record is returned intact so pagination can continue. Continue until `next_cursor` is absent.
