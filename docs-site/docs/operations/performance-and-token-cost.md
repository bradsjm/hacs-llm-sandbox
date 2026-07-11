---
title: Performance And Token Cost
description: Factors that influence latency, model context size, and cost.
---

# Performance And Token Cost

Tool use adds work to each conversation turn. The integration builds snapshots, may query recorder data, may run Monty code, and may return sizable structured outputs.

## Main cost drivers

- Number of visible entities and attributes.
- Prompt profile verbosity.
- Recorder query windows and byte-bounded cursor pages.
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
- Use the matching standalone recorder tool for a direct answer, and run independent direct reads in parallel. When recorder data must be combined with current snapshot data, conditional reasoning, or an action, one `execute_home_code` call avoids a sequential model/tool round, though its recorder work still consumes that execution deadline.
- Choose `guided` for weaker or literal models, keep `balanced` for readable general use, and use `frontier` with GPT-5.6-class models that benefit from a compact contract. Validate answer quality and token cost on representative requests before changing profiles.
