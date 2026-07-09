"""Pure recorder scoping and window helpers over frozen snapshots."""

from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from ...const import MAX_RECORDER_ENTITY_IDS
from ...snapshot.models import HomeSnapshot
from ..errors import RecoverableToolError
from .selectors import expand_aggregate_selector

ENTITY_NOT_VISIBLE = "entity_not_visible"
SELECTOR_NO_MATCH = "selector_no_match"
TIME_WINDOW_TOO_LARGE = "time_window_too_large"


def _validate_visibility(snapshot: HomeSnapshot, ids: list[str]) -> None:
    """Require all requested IDs to exist in the fresh visible snapshot."""
    for entity_id in ids:
        if entity_id not in snapshot.states:
            raise RecoverableToolError(ENTITY_NOT_VISIBLE, {"entity_id": entity_id})


def _as_list(value: object) -> list[str]:
    """Normalize a scalar/list selector value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


# Location-backed selectors resolved through snapshot indexes; ``domain`` is a
# filter, not a location selector, so it is excluded from selector-presence checks.
_LOCATION_SELECTOR_FIELDS = ("area_id", "device_id", "floor_id", "label_id")


def resolve_entity_ids(snapshot: HomeSnapshot, data: dict[str, object], id_key: str) -> list[str]:
    """Resolve explicit IDs plus HA-native selectors to visible entity IDs.

    Explicit IDs are validated for visibility (an invisible one names itself in
    the error). Location selectors (area/device/floor/label) expand to visible
    entities and union across selector types. A selector that is present but
    matches nothing raises ``selector_no_match`` with candidate ids rather than
    widening (e.g. a typo'd ``area_id`` plus ``domain`` would otherwise silently
    expand to every matching-domain entity in the home). ``domain`` filters the
    resolved set and, when no IDs or selectors are given, expands across all
    visible states of that domain.
    """
    explicit = [entity_id.lower() for entity_id in _as_list(data.get(id_key))]
    # Explicit IDs must each be visible (named in the error so the LLM can correct).
    _validate_visibility(snapshot, explicit)
    domains = {domain.lower() for domain in _as_list(data.get("domain"))}
    provided_selectors = [field for field in _LOCATION_SELECTOR_FIELDS if _as_list(data.get(field))]
    selector_present = bool(provided_selectors)

    selector_ids: list[str] = []
    for selector in _LOCATION_SELECTOR_FIELDS:
        for requested in _as_list(data.get(selector)):
            expanded_ids = expand_aggregate_selector(snapshot, selector, requested)
            # A supplied location selector value resolving to nothing is a naming
            # error, not a cue to silently narrow to the other selector values.
            if not expanded_ids:
                raise RecoverableToolError(
                    SELECTOR_NO_MATCH,
                    {
                        "selectors": selector,
                        "selector_id": requested,
                        "domain": next(iter(domains), ""),
                    },
                )
            selector_ids.extend(expanded_ids)

    def _domain_matches(entity_id: str) -> bool:
        return not domains or entity_id.split(".", 1)[0].lower() in domains

    seen: set[str] = set()
    resolved: list[str] = []
    # Explicit IDs are kept as-is (visibility already validated).
    for entity_id in explicit:
        if entity_id not in seen:
            seen.add(entity_id)
            resolved.append(entity_id)
    # Selector expansion keeps only visible entities honoring the domain filter.
    for entity_id in selector_ids:
        if entity_id in seen or entity_id not in snapshot.states or not _domain_matches(entity_id):
            continue
        seen.add(entity_id)
        resolved.append(entity_id)
    # Pure-domain scope with no IDs and no selectors expands across all visible matching states.
    if not resolved and domains and not selector_present:
        resolved.extend(entity_id for entity_id in snapshot.states if _domain_matches(entity_id))

    if not resolved:
        raise RecoverableToolError(
            "invalid_tool_input",
            {"error": "no visible entity IDs or scope selectors resolved"},
        )
    if len(resolved) > MAX_RECORDER_ENTITY_IDS:
        raise RecoverableToolError(
            "invalid_tool_input",
            {"error": f"scope resolves to {len(resolved)} entities; narrow it to at most {MAX_RECORDER_ENTITY_IDS}"},
        )
    return resolved


def _clamp_window(
    now: datetime,
    start_in: datetime | None,
    end_in: datetime | None,
    *,
    hours: float | None = None,
    default_hours: int,
    max_hours: int,
) -> tuple[datetime, datetime]:
    """Resolve start/end values, honoring an explicit window or a relative ``hours`` size.

    Precedence: explicit ``start``/``end`` win; otherwise a relative ``hours``
    size is applied against ``end``; otherwise the tool default window is used.
    The recorder lookback cap is always enforced.
    """
    end = dt_util.as_utc(end_in or now)
    if start_in is not None:
        start = dt_util.as_utc(start_in)
    elif hours is not None:
        start = end - timedelta(hours=hours)
    else:
        start = end - timedelta(hours=default_hours)
    if start > end:
        raise RecoverableToolError("invalid_tool_input", {"error": "start after end"})
    if end - start > timedelta(hours=max_hours):
        raise RecoverableToolError(TIME_WINDOW_TOO_LARGE, {"max_hours": str(max_hours)})
    return start, end
