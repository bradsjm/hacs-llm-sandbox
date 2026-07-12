"""Normalize JSON-safe execute payload records into grounded facts."""

from collections.abc import Mapping

from llm_sandbox_evals.scoring._facts import fact, typed_record
from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def normalize_execute_output(output: object, provenance: Provenance) -> tuple[EvidenceFact, ...]:
    """Normalize JSON-safe facade records returned by execute_home_code."""
    facts: list[EvidenceFact] = []
    _normalize_execute(output, provenance, facts)
    return tuple(facts)


def _normalize_execute(value: object, provenance: Provenance, facts: list[EvidenceFact]) -> None:  # noqa: C901
    if isinstance(value, list):
        for item in value:
            _normalize_execute(item, provenance, facts)
        return
    if not isinstance(value, Mapping):
        return
    entity_id = value.get("entity_id")
    if isinstance(entity_id, str):
        for field in ("domain", "name", "area_id", "device_id", "floor_id", "state"):
            if field in value and isinstance(value[field], (str, int, float, bool)):
                fact(
                    "value",
                    {"subject_kind": "entity", "subject_id": entity_id, "field": field, "value": value[field]},
                    provenance,
                    facts,
                    entity_id,
                )
        if "state" in value:
            fact(
                "value",
                {"subject_kind": "entity", "subject_id": entity_id, "field": "state", "value": value["state"]},
                provenance,
                facts,
                entity_id,
            )
        labels = value.get("labels")
        if isinstance(labels, (list, tuple)):
            for label in labels:
                if isinstance(label, str):
                    fact(
                        "association",
                        {"entity_id": entity_id, "association": "label", "value": label},
                        provenance,
                        facts,
                        entity_id,
                    )
        attributes = value.get("attributes")
        if isinstance(attributes, Mapping):
            for name, item in attributes.items():
                fact(
                    "value",
                    {
                        "subject_kind": "entity",
                        "subject_id": entity_id,
                        "field": "attribute",
                        "attribute_name": name,
                        "value": item,
                    },
                    provenance,
                    facts,
                    entity_id,
                )
        for field in (
            "last_changed",
            "last_changed_timestamp",
            "last_updated",
            "last_updated_timestamp",
            "duration_seconds",
            "time_in_state",
        ):
            if field in value:
                fact(
                    "value",
                    {"subject_kind": "entity", "subject_id": entity_id, "field": field, "value": value[field]},
                    provenance,
                    facts,
                    entity_id,
                )
        for key, relation, object_kind in (
            ("device_id", "entity_device", "device"),
            ("area_id", "entity_area", "area"),
        ):
            if isinstance(value.get(key), str):
                fact(
                    "relation",
                    {
                        "subject_kind": "entity",
                        "subject_id": entity_id,
                        "relation": relation,
                        "object_kind": object_kind,
                        "object_id": value[key],
                    },
                    provenance,
                    facts,
                    entity_id,
                )
        services = value.get("services")
        if isinstance(services, (list, tuple)):
            service_values = services
        elif isinstance(services, Mapping):
            service_values = [
                f"{domain}.{service}"
                for domain, names in services.items()
                if isinstance(names, (list, tuple))
                for service in names
                if isinstance(service, str)
            ]
        else:
            service_values = ()
        for service in service_values:
            if isinstance(service, str):
                fact(
                    "relation",
                    {
                        "subject_kind": "entity",
                        "subject_id": entity_id,
                        "relation": "entity_service",
                        "object_kind": "service",
                        "object_id": service,
                    },
                    provenance,
                    facts,
                    entity_id,
                )
    typed_record(value, provenance, facts)
