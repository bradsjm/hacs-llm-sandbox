"""Successful-event selection and provenance-preserving evidence union."""

from collections.abc import Mapping, Sequence

from llm_sandbox_evals.schema import ToolEvent
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence, Provenance

_TOOLS = frozenset({"execute_home_code", "get_history", "get_statistics", "get_logbook", "get_automation"})


def successful_events(events: Sequence[ToolEvent]) -> tuple[ToolEvent, ...]:
    """Return usable production events; failed calls remain available as diagnostics."""
    result: list[ToolEvent] = []
    for event in events:
        if event.tool_name not in _TOOLS or not isinstance(event.output, Mapping):
            continue
        execution = event.output.get("execution")
        if isinstance(execution, Mapping) and execution.get("status") != "ok":
            continue
        if event.output.get("status") == "error":
            continue
        result.append(event)
    return tuple(result)


def normalize_events(events: Sequence[ToolEvent]) -> NormalizedEvidence:
    """Union facts from every successful call, independent of call order or path."""
    facts: list[EvidenceFact] = []
    for event in successful_events(events):
        provenance = Provenance(
            event.tool_name, event.call_index, event.turn_index, event.batch_index, event.tool_name
        )
        if event.tool_name == "execute_home_code":
            _normalize_execute(event.output.get("output"), provenance, facts)
        elif event.tool_name == "get_history":
            _normalize_history(event.output, provenance, facts)
        elif event.tool_name == "get_statistics":
            _normalize_statistics(event.output, provenance, facts)
        elif event.tool_name == "get_logbook":
            _normalize_logbook(event.output, provenance, facts)
        else:
            _normalize_automation(event.output, provenance, facts)
    return NormalizedEvidence(tuple(facts))


def _fact(
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
                _fact(
                    "value",
                    {"subject_kind": "entity", "subject_id": entity_id, "field": field, "value": value[field]},
                    provenance,
                    facts,
                    entity_id,
                )
        labels = value.get("labels")
        if isinstance(labels, (list, tuple)):
            for label in labels:
                if isinstance(label, str):
                    _fact(
                        "association",
                        {"entity_id": entity_id, "association": "label", "value": label},
                        provenance,
                        facts,
                        entity_id,
                    )
        if "state" in value:
            _fact(
                "value",
                {"subject_kind": "entity", "subject_id": entity_id, "field": "state", "value": value["state"]},
                provenance,
                facts,
                entity_id,
            )
        attributes = value.get("attributes")
        if isinstance(attributes, Mapping):
            for name, item in attributes.items():
                _fact(
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
        for key in (
            "last_changed",
            "last_changed_timestamp",
            "last_updated",
            "last_updated_timestamp",
            "duration_seconds",
            "time_in_state",
        ):
            if key in value:
                _fact(
                    "value",
                    {"subject_kind": "entity", "subject_id": entity_id, "field": key, "value": value[key]},
                    provenance,
                    facts,
                    entity_id,
                )
    for key, subject_kind, relation, object_kind in (
        ("device_id", "entity", "entity_device", "device"),
        ("area_id", "entity", "entity_area", "area"),
    ):
        if isinstance(entity_id, str) and isinstance(value.get(key), str):
            _fact(
                "relation",
                {
                    "subject_kind": subject_kind,
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
    if isinstance(entity_id, str) and isinstance(services, (list, tuple, Mapping)):
        service_values = (
            services
            if isinstance(services, (list, tuple))
            else [
                f"{domain}.{service}"
                for domain, service_names in services.items()
                if isinstance(service_names, (list, tuple))
                for service in service_names
                if isinstance(service, str)
            ]
        )
        for service in service_values:
            if isinstance(service, str):
                _fact(
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
    _normalize_typed_record(value, provenance, facts)
    if isinstance(value.get("id"), str):
        record_id = str(value["id"])
        for field in (
            "name",
            "manufacturer",
            "model",
            "enabled",
            "available",
            "service",
            "message",
            "status",
            "value",
            "unit",
        ):
            if field in value:
                _fact(
                    "value",
                    {
                        "subject_kind": "service" if field == "service" else "entity",
                        "subject_id": record_id,
                        "field": field,
                        "value": value[field],
                    },
                    provenance,
                    facts,
                    record_id,
                )


def _normalize_typed_record(value: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
    """Normalize explicit facade registry/diagnostic records only."""
    record_id: str | None = None
    kind: str | None = None
    if isinstance(value.get("id"), str) and any(key in value for key in ("manufacturer", "model", "config_entries")):
        record_id, kind = str(value["id"]), "device"
    for key, candidate_kind in (
        ("device_id", "device"),
        ("area_id", "area"),
        ("floor_id", "floor"),
        ("issue_id", "repair"),
        ("notification_id", "notification"),
    ):
        candidate = value.get(key)
        if kind is None and isinstance(candidate, str):
            record_id, kind = candidate, candidate_kind
            break
    if record_id is None and isinstance(value.get("id"), str):
        record_id = str(value["id"])
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
            _fact(
                "value",
                {"subject_kind": kind, "subject_id": record_id, "field": output_field, "value": value[field_name]},
                provenance,
                facts,
                record_id,
            )
    if kind == "device" and isinstance(value.get("area_id"), str):
        _fact(
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
        _fact(
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


def _normalize_history(output: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
    entities = output.get("entities")
    if isinstance(entities, Mapping):
        for entity_id, payload in entities.items():
            if not isinstance(entity_id, str) or not isinstance(payload, Mapping):
                continue
            _fact("scope", {"source": "history", "entity_id": entity_id}, provenance, facts, entity_id)
            for row in payload.get("rows", []):
                if isinstance(row, list) and len(row) >= 2 and isinstance(row[0], str):
                    _fact(
                        "history_row",
                        {"entity_id": entity_id, "when": row[0], "state": row[1]},
                        provenance,
                        facts,
                        entity_id,
                    )
                    if len(row) >= 3 and isinstance(row[2], Mapping):
                        for name, item in row[2].items():
                            _fact(
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
                _fact(
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
                        _fact(
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
                _fact("scope", {"source": "history", "entity_id": entity_id}, provenance, facts, entity_id)


def _normalize_statistics(output: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
    statistics = output.get("statistics")
    if not isinstance(statistics, Mapping):
        return
    for statistic_id, payload in statistics.items():
        if not isinstance(statistic_id, str) or not isinstance(payload, Mapping):
            continue
        _fact("scope", {"source": "statistics", "entity_id": statistic_id}, provenance, facts, statistic_id)
        for row in payload.get("rows", []):
            if isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) and isinstance(row[1], Mapping):
                for field, value in row[1].items():
                    _fact(
                        "statistic_value",
                        {"statistic_id": statistic_id, "when": row[0], "field": field, "value": value},
                        provenance,
                        facts,
                        statistic_id,
                    )


def _normalize_logbook(output: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
    entries = output.get("entries")
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, Mapping) and isinstance(entry.get("entity_id"), str):
            _fact(
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
                _fact("scope", {"source": "logbook", "entity_id": entity_id}, provenance, facts, entity_id)


def _normalize_automation(output: Mapping[str, object], provenance: Provenance, facts: list[EvidenceFact]) -> None:
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
                _fact(
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
                    _fact(
                        "value",
                        {"subject_kind": "automation", "subject_id": entity_id, "field": "value", "value": value},
                        provenance,
                        facts,
                        entity_id,
                    )
        action = content.get("action") if isinstance(content, Mapping) else None
        target = action.get("target") if isinstance(action, Mapping) else None
        targets = target.get("entity_id") if isinstance(target, Mapping) else None
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(targets, list):
            for target_id in targets:
                if isinstance(target_id, str):
                    _fact(
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
        if isinstance(runs, list):
            for run in runs:
                if isinstance(run, Mapping) and isinstance(run.get("when"), str):
                    _fact(
                        "automation_run",
                        {"entity_id": entity_id, "when": run["when"], "value": str(run.get("message", "triggered"))},
                        provenance,
                        facts,
                        entity_id,
                    )
