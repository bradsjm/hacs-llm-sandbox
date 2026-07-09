---
title: Performance And Token Cost
description: Factors that influence latency, model context size, and cost.
---

# Performance And Token Cost

Tool use adds work to each conversation turn. The integration builds snapshots, may query recorder data, may run Monty code, and may return sizable structured outputs.

## Main cost drivers

- Number of visible entities and attributes.
- Prompt profile verbosity.
- Recorder query windows and output limits.
- SQL queries that load history or statistics.
- Camera images.
- Follow-up calls caused by weak model behavior.

## Main latency drivers

- Snapshot size.
- Recorder/logbook query time.
- Monty execution time.
- Image capture and JPEG normalization.
- Model latency from larger tool inputs and outputs.

## Practical tuning

- Keep Assist exposure focused.
- Use the default visibility filters first.
- Ask scoped questions: include area, domain, entity, or time window.
- Prefer smaller history windows unless a broader trend is needed.
- Use `terse` or `minimal` prompt profiles only after validating model quality.
