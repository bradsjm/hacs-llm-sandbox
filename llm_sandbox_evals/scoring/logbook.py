"""Normalize logbook events and resolved empty-result scopes."""

from collections.abc import Mapping

from llm_sandbox_evals.scoring._facts import fact
from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def normalize_logbook_output(output: Mapping[str, object], provenance: Provenance) -> tuple[EvidenceFact, ...]:
    """Normalize logbook entries and the resolved scope of empty results."""
    facts: list[EvidenceFact] = []
    entries = output.get("entries")
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, Mapping) and isinstance(entry.get("entity_id"), str):
            fact(
                "logbook_event",
                {key: entry.get(key) for key in ("entity_id", "when", "message")},
                provenance,
                facts,
                str(entry["entity_id"]),
            )
    scope = output.get("scope")
    if isinstance(scope, Mapping) and isinstance(scope.get("entity_ids"), list):
        for entity_id in scope["entity_ids"]:
            if isinstance(entity_id, str):
                fact("scope", {"source": "logbook", "entity_id": entity_id}, provenance, facts, entity_id)
    return tuple(facts)
