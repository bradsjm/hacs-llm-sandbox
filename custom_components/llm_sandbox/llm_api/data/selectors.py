"""Pure snapshot-backed HA target selector expansion."""

from collections.abc import Mapping

from ...snapshot.models import HomeSnapshot

AGGREGATE_SELECTOR_KEYS = ("device_id", "area_id", "label_id", "label", "floor_id")


def _selector_values(value: object) -> tuple[str, ...]:
    """Normalize a target selector value to one or more string IDs."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return (str(value),)


def expand_aggregate_selector(snapshot: HomeSnapshot, selector: str, value: object) -> tuple[str, ...]:
    """Expand one index-backed aggregate selector value to visible entity IDs."""
    indexes = snapshot.indexes
    if selector == "device_id":
        return tuple(indexes.entity_ids_by_device_id.get(str(value), ()))
    if selector == "area_id":
        return tuple(indexes.entity_ids_by_area_id.get(str(value), ()))
    if selector in {"label", "label_id"}:
        return tuple(indexes.entity_ids_by_label.get(str(value), ()))
    if selector == "floor_id":
        entity_ids: list[str] = []
        for area_id in indexes.area_ids_by_floor_id.get(str(value), ()):
            entity_ids.extend(indexes.entity_ids_by_area_id.get(area_id, ()))
        return tuple(entity_ids)
    return ()


def expand_aggregate_selectors(
    snapshot: HomeSnapshot,
    target: Mapping[str, object],
    *,
    selector_keys: tuple[str, ...] = AGGREGATE_SELECTOR_KEYS,
) -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """Expand aggregate selectors, preserving requested values for caller-specific handling."""
    expansions: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    for selector in selector_keys:
        if selector not in target:
            continue
        resolved_values = tuple(
            (requested, resolved)
            for requested in _selector_values(target[selector])
            if (resolved := expand_aggregate_selector(snapshot, selector, requested))
        )
        if resolved_values:
            expansions[selector] = resolved_values
    return expansions
