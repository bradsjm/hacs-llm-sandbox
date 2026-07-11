# Direct Tool Design

Direct tools other than `execute_home_code` are one family of bounded,
declarative Home Assistant queries. Keep their inputs, output envelopes,
authorization, visibility, pagination, and errors consistent even when their
underlying data sources differ. Do not force every source into SQL semantics;
relational operators such as aggregation and grouping belong only where the
data naturally supports them.

## Access Follows Home Assistant

- Return every detail the requesting Home Assistant user is authorized to
  access. Do not add assistant-specific redaction, Assist-exposure filtering,
  sandbox visibility filtering, or speculative sensitivity classifications.
- Use Home Assistant's authorization rules as the sole content-access policy.
  Administrator-only Home Assistant data remains administrator-only; finer
  permissions should be honored when Home Assistant provides them.
- Resolve and enforce the requesting user from the tool's `llm_context` on the
  host side for every call and every cursor continuation. If a tool requires
  user authorization but no attributable user exists, return an authorization
  error rather than guessing a restricted view or granting integration-level
  access.
- Keep live `hass`, auth objects, registries, recorder objects, automation
  entities, service handles, and other Home Assistant internals host-side.
  Tool responses contain only copied, JSON-compatible data.
- Content bounds are not redaction. Authorized data may be split across
  explicit cursor pages, but it must not be silently omitted or rewritten.

## Declarative Query Inputs

- Prefer optional named filters and HA-native selectors over RPC-style
  `operation=list|get|history` switches. One query shape should naturally
  narrow from discovery to exact retrieval.
- Reuse established parameter names and meanings wherever applicable:
  `entity_ids`, `area_id`, `device_id`, `floor_id`, `label_id`, `domain`,
  `hours`, `start`, `end`, `limit`, and `cursor`.
- Use Home Assistant terminology in public schemas. For example, expose
  `labels`, not an invented synonym such as `tags`.
- Use explicit projection controls such as `include`, `attributes`, or a
  domain-specific `section` when detailed records can be large. Broad queries
  should default to compact summaries; callers request expensive content or
  history deliberately.
- Accept caller limits below the component ceiling, but never let a caller
  raise fixed safety or response-size ceilings.
- Normalize null and empty optional values consistently through the shared
  support helpers. Malformed non-empty values must still fail validation.
- Validate the schema before checking runtime availability, matching the
  recorder tool contract and eval path.
- A selector that is present but matches nothing must return the stable
  no-match error with actionable candidates; it must never widen into an
  unscoped query.

## Response Consistency

- Return a stable collection envelope even when a filter selects one object.
  Use a domain-named collection, plus `returned`, `limit`, and `next_cursor`
  where pagination applies.
- Keep records structured and JSON-compatible. Preserve Home Assistant IDs and
  distinguish identifiers with different meanings rather than collapsing
  them, such as entity ID, configuration ID, and execution/run ID.
- Use deterministic ordering so repeated calls and cursor continuations are
  predictable.
- Make partial retrieval explicit through the requested projection or a
  non-null `next_cursor`. Never silently truncate a record set.
- Do not echo request fields or emit null metadata unless they help the model
  interpret or continue the result.
- Keep success payloads compact and decision-relevant. Use the shared stable
  error envelope and guidance conventions for recoverable failures.

## Byte-Bounded Pagination

- Raw direct-tool cursor pages normally fit the complete compact UTF-8 JSON
  response within `MAX_RECORDER_PAGE_BYTES` (16 KiB). The budget includes the
  collection envelope, window metadata, counts, and continuation cursor, not
  only the records.
- Serialized bytes, not row counts or Python object size, determine the normal
  page boundary. Item-count limits remain caller controls or emergency
  ceilings.
- Preserve whole semantic records. Do not split structured records merely to
  meet the normal byte target.
- If the first record alone exceeds 16 KiB, return that complete record by
  itself so the cursor can make progress. Do not skip it in favor of smaller
  later records, redact it, or loop on an empty page.
- Withheld records must remain eligible for the next page. Cursor state must
  resume the exact query stream without duplicates or silent gaps.
- Cursors are opaque, versioned, tool/query-kind-specific, and validated
  against the current request scope. A cursor must not widen access, selectors,
  time windows, projections, or fixed limits.
- Re-evaluate Home Assistant authorization and rebuild any required fresh view
  on every continuation. A cursor conveys position, never authority.
- When paging mutable structured content, bind the cursor to a stable revision
  or content hash. If the source changes between pages, return a stable changed
  or invalid-cursor error rather than combining versions.
- Prefer reusing or extracting the established compact-JSON byte-fitting and
  cursor machinery. Do not introduce a second page-size policy for a new tool.

## Data-Source Boundaries

- A direct tool may use the source best suited to its domain while preserving
  the common contract. Recorder history, Logbook events, live state, registry
  metadata, and host-side component APIs do not need identical internals.
- Build fresh data for each call at the point required by correctness and
  authorization. Never rely on cursor payloads or prior responses as current
  Home Assistant state.
- Distinguish related data with different guarantees. For example, entity
  state history is not an execution ledger, Logbook is not a detailed trace,
  and a reference discovered in configuration is not necessarily a trigger.
- Expose those limitations in concise tool descriptions and response fields;
  do not infer stronger semantics than the source provides.

## Adding A Direct Tool

Before registering a new direct tool:

1. Identify the Home Assistant authorization rule and enforce it host-side.
2. Reuse existing selector, time-window, normalization, error, cursor, and
   compact byte-fitting helpers where their semantics match.
3. Define the compact default projection and explicit detailed projections.
4. Define deterministic ordering, semantic page records, emergency item
   ceilings, and the oversized-first-record behavior.
5. Keep live Home Assistant objects outside returned records and Monty inputs.
6. Add behavior-focused tests for authorization, selector no-match behavior,
   UTF-8 byte fitting, continuation without gaps or duplicates, an oversized
   first record, source changes between pages where relevant, and unavailable
   optional dependencies.
7. Keep tool descriptions, prompt capability catalogs, documentation, limits,
   translations, eval adapters, and tests aligned with the implemented
   contract.
