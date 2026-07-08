"""Pure candidate enumeration from frozen home snapshots."""

from collections.abc import Iterable, Mapping
from typing import cast

from ...snapshot.models import HomeSnapshot, ServiceSchemaBrief
from ..data.home_db import _SCHEMA_TABLE_NAMES, columns_for_table
from ..target_matching import raw_service_field_names

type CandidateDict = dict[str, object]


def entity_candidates(snapshot: HomeSnapshot, domain: str = "") -> tuple[CandidateDict, ...]:
    """Yield visible entity candidates with registry-derived context."""
    candidates: list[CandidateDict] = []
    for entity_id, state in sorted(snapshot.states.items()):
        # Domain filters scope recovery to the surface that failed.
        if domain and state.domain != domain:
            continue
        area = snapshot.areas.get(state.area_id or "")
        floor = snapshot.floors.get(area.floor_id or "") if area is not None else None
        entry = snapshot.entities.get(entity_id)
        device_class = state.attributes.get("device_class")
        unit = state.attributes.get("unit_of_measurement")
        candidates.append(
            {
                "kind": "entity",
                "id": entity_id,
                "entity_id": entity_id,
                "name": state.name or "",
                "object_id": state.object_id,
                "area_id": state.area_id or "",
                "area_name": area.name if area is not None else "",
                "floor_name": floor.name if floor is not None else "",
                "device_class": device_class
                if isinstance(device_class, str)
                else (entry.device_class if entry else ""),
                "unit": unit if isinstance(unit, str) else "",
                "domain": state.domain,
                "aliases": entry.aliases if entry is not None else (),
                "entity_category": entry.entity_category if entry is not None else "",
            }
        )
    return tuple(candidates)


def area_candidates(snapshot: HomeSnapshot) -> tuple[CandidateDict, ...]:
    """Yield area selector candidates."""
    return tuple(
        {
            "kind": "area",
            "id": area.area_id,
            "area_id": area.area_id,
            "name": area.name,
            "aliases": area.aliases,
            "floor_id": area.floor_id or "",
            "floor_name": snapshot.floors[area.floor_id].name if area.floor_id in snapshot.floors else "",
        }
        for area in sorted(snapshot.areas.values(), key=lambda item: item.area_id)
    )


def floor_candidates(snapshot: HomeSnapshot) -> tuple[CandidateDict, ...]:
    """Yield floor selector candidates."""
    return tuple(
        {
            "kind": "floor",
            "id": floor.floor_id,
            "floor_id": floor.floor_id,
            "name": floor.name,
            "aliases": floor.aliases,
            "level": floor.level,
        }
        for floor in sorted(snapshot.floors.values(), key=lambda item: item.floor_id)
    )


def label_candidates(snapshot: HomeSnapshot) -> tuple[CandidateDict, ...]:
    """Yield label selector candidates."""
    return tuple(
        {"kind": "label", "id": label.label_id, "label_id": label.label_id, "name": label.name, "aliases": ()}
        for label in sorted(snapshot.labels.values(), key=lambda item: item.label_id)
    )


def device_candidates(snapshot: HomeSnapshot) -> tuple[CandidateDict, ...]:
    """Yield device selector candidates."""
    return tuple(
        {
            "kind": "device",
            "id": device.id,
            "device_id": device.id,
            "name": device.name_by_user or device.name or "",
            "area_id": device.area_id or "",
            "aliases": (),
        }
        for device in sorted(snapshot.devices.values(), key=lambda item: item.id)
    )


def service_candidates(snapshot: HomeSnapshot, domain: str = "") -> tuple[CandidateDict, ...]:
    """Yield service candidates with schema field names for service-data-aware ranking."""
    candidates: list[CandidateDict] = []
    for service_domain, services in sorted(snapshot.services.items()):
        # Service-domain filters keep nearest-service suggestions inside the requested domain when known.
        if domain and service_domain != domain:
            continue
        for service in sorted(services):
            brief = snapshot.services_schema.get(service_domain, {}).get(service)
            candidates.append(
                {
                    "kind": "service",
                    "id": f"{service_domain}.{service}",
                    "domain": service_domain,
                    "service": service,
                    "name": service.replace("_", " "),
                    "fields": _schema_fields(brief),
                    "aliases": (),
                }
            )
    return tuple(candidates)


def sql_table_candidates() -> tuple[CandidateDict, ...]:
    """Yield the static in-memory SQL table/view names."""
    return tuple(
        {"kind": "sql_table", "id": name, "name": name, "aliases": ()} for name in sorted(_SCHEMA_TABLE_NAMES)
    )


def sql_column_candidates(table_name: str) -> tuple[CandidateDict, ...]:
    """Yield known columns for the in-memory SQL table/view surface.

    Column vocabulary is single-sourced from ``home_db.columns_for_table`` so the
    guidance candidates cannot drift from the actual DDL. An unknown table returns
    no candidates (honest absence) instead of masking with another table's columns.
    """
    return tuple(
        {"kind": "sql_column", "id": name, "name": name, "aliases": ()} for name in columns_for_table(table_name)
    )


def code_global_candidates() -> tuple[CandidateDict, ...]:
    """Yield Monty-visible global names from the generated contract surface."""
    from ..contracts import AVAILABLE_GLOBALS

    return tuple({"kind": "code_name", "id": name, "name": name, "aliases": ()} for name in sorted(AVAILABLE_GLOBALS))


def code_attribute_candidates(available_attributes: Iterable[str]) -> tuple[CandidateDict, ...]:
    """Yield an attribute surface from the supplied context."""
    attributes = tuple(available_attributes)
    return tuple({"kind": "code_attribute", "id": name, "name": name, "aliases": ()} for name in sorted(attributes))


def _schema_fields(brief: ServiceSchemaBrief | Mapping[str, object] | None) -> frozenset[str]:
    """Return service field names using the same schema brief helper as target matching."""
    if not isinstance(brief, Mapping):
        return frozenset()
    return frozenset(cast(str, field["name"]) for field in raw_service_field_names(brief))
