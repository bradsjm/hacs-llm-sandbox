"""Action sections and tool description builders for LLM-facing prompts."""

from ...snapshot.models import HomeSnapshot
from ..data.home_db import render_query_schema_prompt

_INVENTORY_AREA_NAME_CAP = 30
_DOMAIN_COUNT_SEPARATOR = "\N{MULTIPLICATION SIGN}"

ERROR_GUIDANCE_PROMPT = (
    "## Error guidance\n"
    "- Recoverable errors may include guidance: {confidence, candidates, reason, next_step, cross_kind}. "
    "confidence controls candidates: exact/high means a strong single suggestion to act on; "
    "ambiguous/listing means choose carefully from the list; none means nothing matched. "
    "next_step is the concrete next action.\n"
    "- candidates are {id, name, match, detail}.\n"
    "- In execute_home_code success, actions[] may include status:'error' even when execution.status:'ok'; "
    "blocked or failed service calls are recorded action outcomes, not code crashes, and top-level notes "
    "summarizes failed actions.\n"
    "- If a target was auto-resolved, success includes resolutions:[{requested, applied}] and action records "
    "carry resolved_from; report the applied id, not the requested alias."
)

ACTIONS_ENABLED_PROMPT = (
    "## Service calls (enabled)\n"
    "- await hass.services.async_call('<domain>', '<service>', service_data, "
    "target={'entity_id': '<entity_id>'}, blocking=True) performs the call for "
    "real.\n"
    "- Keep service data and target separate; put entities in target "
    "(entity_id/device_id/area_id/label_id/floor_id).\n"
    "- blocking=True waits and reports outcome; blocking=False is "
    "fire-and-forget and yields no detailed error. Prefer blocking when you need "
    "to know it succeeded.\n"
    "- return_response=True requires blocking=True and is needed for services "
    "that produce a response.\n"
    "- Calls run sequentially with no rollback; a later failure does not undo "
    "earlier calls.\n"
    "- Action results are compact records with service, target, status, and "
    "resolved_from or error when relevant.\n"
    "- An action with status:'error' under execution.status:'ok' is a recorded "
    "blocked or failed call, not a code crash; read top-level notes for the "
    "summary and any error.guidance for the next step.\n"
)


ACTIONS_DISABLED_PROMPT = (
    "## Service calls (disabled)\n"
    "Service calls are disabled for this assistant. hass.services.async_call is "
    "rejected. Use the service-catalog reads (has_service, "
    "async_services_for_domain, async_services_for_target, supports_response), "
    "states, and registries only.\n"
)


def compose_system_prompt(
    base_prompt: str,
    actions_enabled: bool,
    *,
    location_section: str | None = None,
    inventory_section: str | None = None,
) -> str:
    """Return the base API prompt plus action and optional dynamic sections."""
    # Exactly one service-call section per instance: live-call guidance when
    # actions are enabled, the rejection notice otherwise. The static tool
    # descriptions stay action-neutral.
    section = ACTIONS_ENABLED_PROMPT if actions_enabled else ACTIONS_DISABLED_PROMPT
    prompt = f"{base_prompt}\n\n{section}\n\n{ERROR_GUIDANCE_PROMPT}"
    # Dynamic inventory comes before request-location scope so the location hint
    # stays the final, most specific request context.
    if inventory_section is not None:
        prompt = f"{prompt}\n\n{inventory_section}"
    if location_section is not None:
        prompt = f"{prompt}\n\n{location_section}"
    return prompt


def render_home_inventory(
    snapshot: HomeSnapshot,
    *,
    recorder_available: bool,
    logbook_available: bool,
) -> str | None:
    """Render a compact visible-inventory digest plus data-availability caveats."""
    if not snapshot.states and recorder_available and logbook_available:
        return None

    lines = ["## Home inventory"]
    if snapshot.areas:
        lines.append(_render_areas_by_floor(snapshot))
    if snapshot.states:
        lines.append(_render_domain_counts(snapshot))

    # Availability caveats mirror the dynamic tool list for this request.
    if not recorder_available:
        lines.append(
            "- Recorder is unavailable: get_history, get_statistics, and get_logbook are not offered this request. "
            "Read current state with execute_home_code."
        )
    else:
        if not logbook_available:
            lines.append("- Logbook is unavailable: get_logbook is not offered this request.")
        if not _has_statistics_candidate(snapshot):
            lines.append(
                "- No visible entities expose long-term statistics (state_class); get_statistics will return empty."
            )

    return "\n".join(lines)


def _render_areas_by_floor(snapshot: HomeSnapshot) -> str:
    """Render visible area names grouped by floor with a large-home cap."""
    if len(snapshot.areas) > _INVENTORY_AREA_NAME_CAP:
        return (
            f"- {len(snapshot.areas)} visible areas across {len(snapshot.floors)} floor(s); "
            "use area_registry.async_list_areas() to enumerate."
        )

    names_by_floor: dict[str | None, list[str]] = {}
    for area in snapshot.areas.values():
        # State mutation: accumulate only visibility-filtered area names from the
        # snapshot; entity ids stay behind tools.
        names_by_floor.setdefault(area.floor_id, []).append(area.name)

    parts: list[str] = []
    for floor in sorted(snapshot.floors.values(), key=lambda item: (item.level is None, item.level or 0, item.name)):
        if floor.floor_id not in names_by_floor:
            continue
        names = sorted(names_by_floor.pop(floor.floor_id))
        parts.append(f"{floor.name} ({', '.join(names)})")

    for floor_id, names in sorted(names_by_floor.items(), key=lambda item: (item[0] is not None, item[0] or "")):
        label = "No floor" if floor_id is None else floor_id
        parts.append(f"{label} ({', '.join(sorted(names))})")

    return f"- Areas by floor: {', '.join(parts)}"


def _render_domain_counts(snapshot: HomeSnapshot) -> str:
    """Render per-domain visible state counts."""
    domain_counts: dict[str, int] = {}
    for state in snapshot.states.values():
        # State mutation: count domains over the visibility-filtered state set.
        domain_counts[state.domain] = domain_counts.get(state.domain, 0) + 1
    ordered = sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))
    return "- Entities by domain: " + ", ".join(
        f"{domain}{_DOMAIN_COUNT_SEPARATOR}{count}" for domain, count in ordered
    )


def _has_statistics_candidate(snapshot: HomeSnapshot) -> bool:
    """Return whether any visible state advertises a recorder statistics surface."""
    return any("state_class" in state.attributes for state in snapshot.states.values())


def render_request_location(
    device_id: str | None,
    area_id: str | None,
    area_name: str | None,
    floor_id: str | None,
    floor_name: str | None,
) -> str | None:
    """Render a compact prompt section for the initiating device location."""
    if device_id is None:
        return None

    lines = [
        "## Request location",
        f"- device_id: {device_id}",
    ]
    if area_id is not None and area_name is not None:
        lines.append(f"- area_id: {area_id} ({area_name})")
    if floor_id is not None and floor_name is not None:
        lines.append(f"- floor_id: {floor_id} ({floor_name})")
    if area_id is not None and area_name is not None:
        lines.append(
            "For underspecified local questions, use this area as the default scope. "
            "If the user asks for the whole home or names another area/floor, follow that explicit scope."
        )
    return "\n".join(lines)


def build_execute_home_code_description() -> str:
    """Return the execute_home_code tool description."""
    return (
        "Execute bounded Python/Monty code against a frozen, read-only Home Assistant view. "
        "Read states and registries using the native Home Assistant patterns documented in the API prompt. "
        f"{render_query_schema_prompt(compact=True, include_heading=False).removeprefix('- ')} "
        "The query load can be narrowed with entity_ids or area_id/floor_id/device_id/label_id/domain. "
        "Service-call availability follows the API prompt. "
        "Success returns {execution:{status:'ok'}, output:<data>} with printed only when print() emitted lines "
        "and resolutions only when a remembered missing literal was rewritten to a visible entity id. "
        "If output is empty because a literal entity id is missing, note names the missing id and structured "
        "guidance may describe visible replacements. Errors return {execution:{status:'code_error'|'helper_error'|"
        "'setup_error', kind?, message, guidance?}, output:null}."
    )


def build_get_history_description() -> str:
    """Return the get_history tool description."""
    return (
        "Return state-value history or server-side summaries for visible "
        "entities over a bounded UTC window; use when you need how a state or "
        "attribute changed over time. Prefer aggregate modes for counts, durations, first/last sightings, "
        "or time in state instead of paging raw rows. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Raw success returns {window, entities}, where entities is keyed by entity_id and each value has "
        "rows of [t, state] plus unit when known; rows omit attributes by default — pass attributes "
        "(a list of attribute names, opt-in) to append a {name: value} element per row carrying only the "
        "requested attributes that exist (bounded count). The first page returns the newest rows; when more "
        "remain, next_cursor appears — pass it back as cursor (omit window args) to fetch the next older page. "
        "Pass aggregate=count_transitions|time_in_state|state_counts|first_seen|last_seen|on_duration "
        "to return {window, mode, summary} with no rows, cursor, or attributes; aggregate windows may be "
        "larger than raw history. count_transitions may include from_state/to_state filters; "
        "first_seen/last_seen may include to_state to find when a specific state first or last appeared. "
        "For declarative analytics, pass aggregate={field:[ops]}, group_by=[...], bucket='1h', where=[...], "
        "order_by, or limit to return {window, rows} as list[dict] with no cursor. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )


def build_get_statistics_description() -> str:
    """Return the get_statistics tool description."""
    return (
        "Return pre-aggregated long-term statistics (mean/min/max/state/sum and raw "
        "recorder units) for visible statistic IDs over a bounded UTC window. "
        "These are historical aggregates over a period, not current values; for "
        "a current value or an average of current states, read states in "
        "execute_home_code instead. Each statistic ID must be a currently-visible "
        "entity ID; external or non-entity statistic IDs are rejected. Scope with "
        "statistic_ids or HA-native selectors "
        "(area_id/device_id/floor_id/label_id/domain); size the window with hours=<n> or ISO start/end. "
        "Success returns {window, period, statistics}, where statistics is keyed by statistic/entity id and each "
        "value has fields plus rows of [t, {field: value}]. Pass types (mean/min/max/state/sum) to select "
        "which statistic fields to include; omitted or null fields are left out. The first page returns the newest rows; when more remain, "
        "next_cursor appears — pass it back as cursor (omit window args) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )


def build_get_logbook_description() -> str:
    """Return the get_logbook tool description."""
    return (
        "Return human-readable logbook entries (the activity/events timeline — "
        "what happened and why) for visible entities over a bounded UTC window; "
        "use for 'what happened with X', activity, or a timeline. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Success returns {window, entries}, where entries is a flat list of timeline records each carrying "
        "its entity_id. The first page returns the newest entries; when more remain, next_cursor appears — "
        "pass it back as cursor (omit window args) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )


def build_get_camera_image_description() -> str:
    """Return the get_camera_image tool description."""
    return (
        "Capture a single live frame from a visible camera or image entity and return it as an "
        "inline image for visual analysis. Use this to answer questions about what is currently "
        "visible. Only entities visible under the configured scope can be captured. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )
