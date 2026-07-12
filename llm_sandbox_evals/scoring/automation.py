"""Normalize automation records, content targets, and runs."""

from collections.abc import Mapping

from llm_sandbox_evals.scoring._facts import fact
from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def normalize_automation_output(output: Mapping[str, object], provenance: Provenance) -> tuple[EvidenceFact, ...]:
    """Normalize automation fields, targets, and run records."""
    facts: list[EvidenceFact] = []
    automations = output.get("automations")
    for record in automations if isinstance(automations, list) else []:
        if not isinstance(record, Mapping) or not isinstance(record.get("entity_id"), str):
            continue
        entity_id = str(record["entity_id"])
        for field, value in record.items():
            if field in {
                "state",
                "enabled",
                "is_on",
                "available",
                "name",
                "title",
                "message",
                "status",
            } and isinstance(value, (str, int, float, bool)):
                fact(
                    "value",
                    {
                        "subject_kind": "automation",
                        "subject_id": entity_id,
                        "field": "enabled" if field == "is_on" else "name" if field == "title" else field,
                        "value": value,
                    },
                    provenance,
                    facts,
                    entity_id,
                )
        content = record.get("content")
        trigger = content.get("trigger") if isinstance(content, Mapping) else None
        if isinstance(trigger, Mapping):
            for value in trigger.values():
                if isinstance(value, (str, int, float, bool)):
                    fact(
                        "value",
                        {"subject_kind": "automation", "subject_id": entity_id, "field": "value", "value": value},
                        provenance,
                        facts,
                        entity_id,
                    )
        action = content.get("action") if isinstance(content, Mapping) else None
        target = action.get("target") if isinstance(action, Mapping) else None
        targets = target.get("entity_id") if isinstance(target, Mapping) else None
        targets = [targets] if isinstance(targets, str) else targets
        for target_id in targets if isinstance(targets, list) else []:
            if isinstance(target_id, str):
                fact(
                    "relation",
                    {
                        "subject_kind": "automation",
                        "subject_id": entity_id,
                        "relation": "automation_target",
                        "object_kind": "entity",
                        "object_id": target_id,
                    },
                    provenance,
                    facts,
                    entity_id,
                )
        runs = record.get("runs")
        for run in runs if isinstance(runs, list) else []:
            if isinstance(run, Mapping) and isinstance(run.get("when"), str):
                fact(
                    "automation_run",
                    {"entity_id": entity_id, "when": run["when"], "value": str(run.get("message", "triggered"))},
                    provenance,
                    facts,
                    entity_id,
                )
    return tuple(facts)
