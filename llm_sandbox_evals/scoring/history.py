"""Normalize keyed and flat history payloads."""

from collections.abc import Mapping
from datetime import datetime

from llm_sandbox_evals.scoring._facts import fact
from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def normalize_history_output(output: Mapping[str, object], provenance: Provenance) -> tuple[EvidenceFact, ...]:
    """Normalize keyed and flat history records while preserving entity scope."""
    facts: list[EvidenceFact] = []
    window = output.get("window")
    window_start = window.get("start") if isinstance(window, Mapping) else None
    window_end = window.get("end") if isinstance(window, Mapping) else None
    start_time = _parse_time(window_start) if isinstance(window_start, str) else None
    end_time = _parse_time(window_end) if isinstance(window_end, str) else None
    try:
        valid_window = start_time is not None and end_time is not None and start_time <= end_time
    except TypeError:
        valid_window = False
    # Preserve the complete window identity: duration scoring must not infer bounds.
    fact(
        "history_window",
        {
            "start": start_time.isoformat() if valid_window and start_time is not None else window_start,
            "end": end_time.isoformat() if valid_window and end_time is not None else window_end,
            "valid": valid_window,
        },
        provenance,
        facts,
        "window",
    )
    entities = output.get("entities")
    if isinstance(entities, Mapping):
        for entity_id, payload in entities.items():
            if not isinstance(entity_id, str) or not isinstance(payload, Mapping):
                continue
            fact("scope", {"source": "history", "entity_id": entity_id}, provenance, facts, entity_id)
            for row in payload.get("rows", []):
                if isinstance(row, list) and len(row) >= 2 and isinstance(row[0], str):
                    fact(
                        "history_row",
                        {"entity_id": entity_id, "when": row[0], "state": row[1]},
                        provenance,
                        facts,
                        entity_id,
                    )
                    if len(row) >= 3 and isinstance(row[2], Mapping):
                        for name, item in row[2].items():
                            fact(
                                "history_attribute",
                                {"entity_id": entity_id, "when": row[0], "attribute_name": name, "value": item},
                                provenance,
                                facts,
                                entity_id,
                            )
    rows = output.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                fact(
                    "history_row",
                    {
                        key: value
                        for key, value in row.items()
                        if key in {"entity_id", "when", "state", "value", "unit"}
                    },
                    provenance,
                    facts,
                    str(row.get("entity_id", "rows")),
                )
                attributes = row.get("attributes")
                if isinstance(attributes, Mapping) and isinstance(row.get("entity_id"), str):
                    for name, item in attributes.items():
                        fact(
                            "history_attribute",
                            {
                                "entity_id": row["entity_id"],
                                "when": row.get("when"),
                                "attribute_name": name,
                                "value": item,
                            },
                            provenance,
                            facts,
                            str(row["entity_id"]),
                        )
    scope = output.get("scope")
    if isinstance(scope, Mapping) and isinstance(scope.get("entity_ids"), list):
        for entity_id in scope["entity_ids"]:
            if isinstance(entity_id, str):
                fact("scope", {"source": "history", "entity_id": entity_id}, provenance, facts, entity_id)
    return tuple(facts)


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
