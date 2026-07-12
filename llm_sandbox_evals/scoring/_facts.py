"""Shared fact construction for source-specific payload normalizers."""

from collections.abc import Mapping

from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def fact(
    kind: str, values: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact], record_id: str
) -> None:
    scalar_values = tuple(
        (key, value) for key, value in values.items() if isinstance(value, (str, int, float, bool)) or value is None
    )
    facts.append(
        EvidenceFact(
            kind,
            scalar_values,
            Provenance(
                provenance.tool_name, provenance.call_index, provenance.turn_index, provenance.batch_index, record_id
            ),
        )
    )


def typed_record(value: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
    """Normalize explicit facade registry and diagnostic records only."""
    # Entity/state records carry registry join keys but are never those joined records.
    if isinstance(value.get("entity_id"), str):
        return
    record_id: str | None = None
    kind: str | None = None
    if isinstance(value.get("id"), str) and any(
        key in value for key in ("manufacturer", "model", "config_entries", "identifiers", "connections")
    ):
        record_id, kind = str(value["id"]), "device"
    for key, candidate_kind in (
        ("area_id", "area"),
        ("floor_id", "floor"),
        ("issue_id", "repair"),
        ("notification_id", "notification"),
    ):
        candidate = value.get(key)
        if kind is None and isinstance(candidate, str):
            record_id, kind = candidate, candidate_kind
            break
    if record_id is None or kind is None:
        return
    field_names = {
        "device": ("name", "name_by_user", "manufacturer", "model", "model_id", "sw_version", "hw_version", "area_id"),
        "area": ("name", "floor_id"),
        "floor": ("name", "level"),
        "repair": ("active", "dismissed_version", "severity", "message", "domain", "issue_id"),
        "notification": ("title", "message", "created_at", "notification_id"),
    }[kind]
    for field_name in field_names:
        if field_name in value:
            output_field = (
                "status"
                if kind == "repair" and field_name == "active"
                else "name"
                if kind == "notification" and field_name == "title"
                else field_name
            )
            fact(
                "value",
                {"subject_kind": kind, "subject_id": record_id, "field": output_field, "value": value[field_name]},
                provenance,
                facts,
                record_id,
            )
    if kind == "device" and isinstance(value.get("area_id"), str):
        fact(
            "relation",
            {
                "subject_kind": "device",
                "subject_id": record_id,
                "relation": "device_area",
                "object_kind": "area",
                "object_id": value["area_id"],
            },
            provenance,
            facts,
            record_id,
        )
    if kind == "area" and isinstance(value.get("floor_id"), str):
        fact(
            "relation",
            {
                "subject_kind": "area",
                "subject_id": record_id,
                "relation": "area_floor",
                "object_kind": "floor",
                "object_id": value["floor_id"],
            },
            provenance,
            facts,
            record_id,
        )
