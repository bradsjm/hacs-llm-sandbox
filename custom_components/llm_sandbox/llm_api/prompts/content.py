"""Action sections and tool description builders for LLM-facing prompts."""

from homeassistant.helpers import llm

from ...snapshot.models import HomeSnapshot, SafeAreaEntry
from ..data.home_db import render_query_schema_prompt
from .profiles import PromptDetail, PromptProfile

_INVENTORY_AREAS_PER_FLOOR = 10
_INVENTORY_FLOOR_CAP = 10
_DOMAIN_COUNT_SEPARATOR = "\N{MULTIPLICATION SIGN}"


def compose_system_prompt(
    profile: PromptProfile,
    actions_enabled: bool,
    *,
    base_prompt: str | None = None,
    tool_section: str | None = None,
    location_section: str | None = None,
    inventory_section: str | None = None,
) -> str:
    """Return the base API prompt plus action and optional dynamic sections."""
    # Exactly one service-call section per instance: live-call guidance when
    # actions are enabled, the rejection notice otherwise. The static tool
    # descriptions stay action-neutral.
    selected_base_prompt = profile.base_prompt if base_prompt is None else base_prompt
    action_section = _render_action_guidance(profile.detail, actions_enabled)
    error_section = _render_error_guidance(profile.detail)
    prompt_parts = [part for part in (tool_section, selected_base_prompt, action_section, error_section) if part]
    prompt = "\n\n".join(prompt_parts)
    # Dynamic inventory comes before request-location scope so the location hint
    # stays the final, most specific request context.
    if inventory_section is not None:
        prompt = f"{prompt}\n\n{inventory_section}"
    if location_section is not None:
        prompt = f"{prompt}\n\n{location_section}"
    return prompt


def _render_action_guidance(detail: PromptDetail, actions_enabled: bool) -> str:
    """Render exactly one action contract at the requested profile detail."""
    if not actions_enabled:
        return (
            "## Service calls (disabled)\n"
            "Service calls are disabled for this assistant. hass.services.async_call is rejected. Use service-catalog reads "
            "(has_service, async_services_for_domain, async_services_for_target, supports_response), states, and registries only."
        )
    if detail is PromptDetail.GUIDED:
        return """## Service calls (enabled)
- await hass.services.async_call('<domain>', '<service>', service_data, target={'entity_id': '<entity_id>'}, blocking=True) performs a real call.
- Keep service data separate from target; targets use entity_id/device_id/area_id/label_id/floor_id. blocking=True waits and reports the outcome; blocking=False is fire-and-forget without detailed errors. Services advertising optional or required responses are automatically run blocking with response capture.
- Calls run sequentially with no rollback: later failures do not undo earlier calls. Each call is prevalidated against the fresh snapshot for target visibility, target capability, service-data field capability, response mode, and configured action-domain policy before live dispatch.
- Action records are compact {service, target, status, resolved_from? | error?}. status:'error' under execution.status:'ok' is a recorded blocked/failed action, not a code crash; use top-level notes and error.guidance to recover."""
    if detail is PromptDetail.BALANCED:
        return """## Service calls (enabled)
- await hass.services.async_call('<domain>', '<service>', service_data, target={'entity_id': '<entity_id>'}, blocking=True). Keep service_data and target separate; target accepts entity_id/device_id/area_id/label_id/floor_id. blocking=True reports outcomes; blocking=False is fire-and-forget. Services advertising optional or required responses are automatically run blocking with response capture.
- Calls are sequential with no rollback and are prevalidated against the fresh snapshot for target visibility/capability, service-data fields, response mode, and action-domain policy. Records contain service, target, status, and resolved_from or error; status:'error' may occur under execution.status:'ok' with recovery in notes/error.guidance."""
    return """## Service calls (enabled)
- await hass.services.async_call(domain, service, service_data, target={entity_id|device_id|area_id|label_id|floor_id: ...}, blocking=True); keep data and target separate. Optional or required response services are automatically run blocking with response capture.
- Calls are sequential/no rollback; fresh-snapshot validation covers target visibility/capability, fields, response mode, and action policy.
- Outcome: {service,target,status,resolved_from?|error?}; status:'error' can accompany execution.status:'ok', with notes/error.guidance."""


def _render_error_guidance(detail: PromptDetail) -> str:
    """Render structured recovery metadata without changing its stable contract."""
    if detail is PromptDetail.GUIDED:
        return """## Error guidance
- Recoverable errors may include guidance: {confidence, candidates, reason, next_step, cross_kind}. exact/high means a strong single suggestion; ambiguous/listing means choose carefully from candidates; none means nothing matched. next_step is the concrete next action.
- candidates are {id, name, match, detail}. In execute_home_code success, actions[] may include status:'error' while execution.status:'ok': these are recorded blocked/failed action outcomes, not code crashes, and top-level notes summarizes them.
- Auto-resolved targets appear in resolutions:[{requested, applied}] and action records carry resolved_from; report the applied ID, not the requested alias."""
    confidence = (
        "confidence: exact/high is strong, ambiguous/listing needs selection, none has no match. "
        if detail is PromptDetail.BALANCED
        else "confidence distinguishes strong, ambiguous/listing, and no-match guidance. "
    )
    return (
        "## Error guidance\n"
        "- Recoverable errors may include guidance:{confidence,candidates,reason,next_step,cross_kind}; "
        f"{confidence}candidates are {{id,name,match,detail}} and next_step is the next action.\n"
        "- execute_home_code may succeed with actions[].status:'error' and execution.status:'ok': actions are recorded "
        "blocked/failed outcomes, not crashes; see notes/error.guidance. Auto-resolution reports "
        "resolutions:[{requested,applied}] and action resolved_from; report the applied ID."
    )


def render_tool_capabilities(tools: list[llm.Tool]) -> str:
    """Render cross-tool decision guidance not covered by provider tool schemas.

    Each tool's full description is already sent as a provider function schema,
    so this returns only the recorder-routing guidance — which tool to pick for
    a given recorder need — that no single tool description can express alone.
    """
    tool_names = {tool.name for tool in tools}
    if not (tool_names & {"get_history", "get_statistics", "get_logbook"}):
        return ""
    return "\n".join(
        (
            "## Recorder routing",
            "- For direct history, statistics, or logbook retrieval/summarization, use the matching standalone tool.",
            "- For recorder data combined with current state/registries, computation, conditional reasoning, or actions, use one "
            "execute_home_code call with await hass.history(...), hass.query(...), or hass.logbook(...).",
            "- Run independent direct reads in parallel. Scope them with selectors instead of discovery calls, and never retrieve the same evidence twice.",
        )
    )


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
            "- Recorder is unavailable: historical recorder tools are not offered this request. "
            "Read current state with the available code tool."
        )
    else:
        if not logbook_available:
            lines.append(
                "- Logbook is unavailable: standalone and hass.logbook activity/events access are not offered this request."
            )
        if not _has_statistics_candidate(snapshot):
            lines.append(
                "- No visible entities expose long-term statistics (state_class); get_statistics will return empty."
            )

    return "\n".join(lines)


def _render_areas_by_floor(snapshot: HomeSnapshot) -> str:
    """Render visible area names grouped by floor, capped per floor.

    Areas within each floor are ranked by visible entity count (descending) so
    the most relevant areas appear first. The per-floor cap prevents a single
    large floor from displacing all floor names — the failure mode of the
    previous overall-area cap. Floors are capped separately at
    ``_INVENTORY_FLOOR_CAP``.
    """
    # Count visible state-bearing entities per area for importance ranking.
    entity_counts: dict[str, int] = {}
    for state in snapshot.states.values():
        if state.area_id is not None:
            entity_counts[state.area_id] = entity_counts.get(state.area_id, 0) + 1

    areas_by_floor: dict[str | None, list[SafeAreaEntry]] = {}
    for area in snapshot.areas.values():
        # State mutation: accumulate only visibility-filtered area entries from
        # the snapshot; entity ids stay behind tools.
        areas_by_floor.setdefault(area.floor_id, []).append(area)

    def _format_group(label: str, areas: list[SafeAreaEntry]) -> str:
        """Render one floor group with top areas by entity count and a truncation tail."""
        ranked = sorted(areas, key=lambda a: (-entity_counts.get(a.area_id, 0), a.name.lower()))
        shown = ranked[:_INVENTORY_AREAS_PER_FLOOR]
        names = ", ".join(area.name for area in shown)
        if len(ranked) > _INVENTORY_AREAS_PER_FLOOR:
            remaining = len(ranked) - _INVENTORY_AREAS_PER_FLOOR
            names += f", +{remaining} more (area_registry.async_list_areas())"
        return f"{label} ({names})"

    sorted_floors = sorted(snapshot.floors.values(), key=lambda item: (item.level is None, item.level or 0, item.name))
    parts: list[str] = []
    for floor in sorted_floors[:_INVENTORY_FLOOR_CAP]:
        floor_areas = areas_by_floor.pop(floor.floor_id, None)
        if floor_areas is not None:
            parts.append(_format_group(floor.name, floor_areas))

    if len(sorted_floors) > _INVENTORY_FLOOR_CAP:
        remaining_floors = len(sorted_floors) - _INVENTORY_FLOOR_CAP
        parts.append(f"+{remaining_floors} more floors (floor_registry.async_list_floors())")
        # Discard areas belonging to truncated floors so the tail loop renders
        # only genuinely un-floorable areas (None or orphaned floor ids).
        for floor in sorted_floors[_INVENTORY_FLOOR_CAP:]:
            areas_by_floor.pop(floor.floor_id, None)

    # Render areas with no floor (or an orphaned floor id) last.
    for floor_id, areas in sorted(areas_by_floor.items(), key=lambda item: (item[0] is not None, item[0] or "")):
        label = "No floor" if floor_id is None else floor_id
        parts.append(_format_group(label, areas))

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
        "Use await hass.history(...), hass.query(...), or hass.logbook(...) to compose bounded recorder data with "
        "current state, registries, computation, conditional reasoning, or an action in this one call. "
        "Service-call availability follows the API prompt. "
        "Success returns {execution:{status:'ok'}, output:<data>} and may include top-level printed, notes, "
        "actions, resolutions, and overflow; printed appears only when print() emitted lines, actions records service-call "
        "outcomes, notes carries snapshot/action/read guidance, overflow reports truncation metadata, and resolutions "
        "reports auto-applied entity ids. "
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
        "Prefer this standalone tool for direct retrieval; use one execute_home_code call for dependent composition or actions. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Raw success returns {window, entities}, where entities is keyed by entity_id and each value has "
        "rows of [t, state] plus unit when known; rows omit attributes by default — pass attributes "
        "(a list of attribute names, opt-in) to append a {name: value} element per row carrying only the "
        "requested attributes that exist (bounded count). The first page returns the newest rows; when more "
        "remain, next_cursor and overflow appear — pass next_cursor back as cursor to the same tool with the same resolved scope "
        "(omit start, end, and hours) to fetch the next older page. "
        "Pass aggregate=count_transitions|time_in_state|state_counts|first_seen|last_seen|on_duration "
        "to return {window, mode, summary} with no rows, cursor, or attributes; aggregate windows may be "
        "larger than raw history. count_transitions may include from_state/to_state filters; "
        "first_seen/last_seen may include to_state to find when a specific state first or last appeared. "
        "For declarative analytics, pass aggregate={field:[ops]}, group_by=[...], bucket='1h', where=[...], "
        "order_by, or limit to return {window, scope:{entity_ids}, rows} as list[dict] with the resolved visible "
        "entity IDs (including when rows is empty), with no cursor. "
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
        "entity ID; external or non-entity statistic IDs are rejected. "
        "Prefer this standalone tool for direct retrieval; use one execute_home_code call for dependent composition or actions. "
        "Scope with statistic_ids or HA-native selectors "
        "(area_id/device_id/floor_id/label_id/domain); size the window with hours=<n> or ISO start/end. "
        "Success returns {window, period, statistics}, where statistics is keyed by statistic/entity id and each "
        "value has fields plus rows of [t, {field: value}]. Pass types (mean/min/max/state/sum) to select "
        "which statistic fields to include; omitted or null fields are left out. The first page returns the newest rows; when more remain, "
        "next_cursor and overflow appear — pass next_cursor back as cursor to the same tool with the same resolved scope "
        "(omit start, end, and hours) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )


def build_get_logbook_description() -> str:
    """Return the get_logbook tool description."""
    return (
        "Return human-readable logbook entries (the activity/events timeline — "
        "what happened and why) for visible entities over a bounded UTC window; "
        "use for 'what happened with X', activity, or a timeline. "
        "Prefer this standalone tool for direct retrieval; use one execute_home_code call for dependent composition or actions. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Success returns {window, scope:{entity_ids}, entries}, where scope contains the resolved visible entity IDs "
        "and entries is a flat list of timeline records each carrying its entity_id. The first page returns the newest "
        "entries; when more remain, next_cursor and overflow appear — "
        "pass next_cursor back as cursor to the same tool with the same resolved scope "
        "(omit start, end, and hours) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )


def build_get_automation_description() -> str:
    """Return the get_automation tool description."""
    return (
        "Search Home Assistant automations and return summaries, with optional complete content and recent run activity. "
        "Results are authorized by the requesting user's entity read permission, "
        "sorted by entity_id. Search title, description (for administrators), IDs, aliases, assigned metadata, "
        "and referenced entity/device/area/floor/label metadata with query, or narrow with automation entity_ids. "
        "Optionally include complete automation content, which requires a Home Assistant administrator; it is never "
        "partially redacted. Optionally include recent non-trace Logbook automation-triggered runs with hours or ISO "
        "start/end; runs require recorder and logbook runtime support and default to 24 hours. "
        "Pagination uses a cursor-only continuation and returns whole automation records within the compact 16 KiB "
        "UTF-8 response budget, allowing one oversized first record. Runs are trigger entries only and do not prove "
        "that conditions or actions succeeded. Errors return {status:'error', error:{key, message}}."
    )


def build_get_camera_image_description() -> str:
    """Return the get_camera_image tool description."""
    return (
        "Capture a single live frame from a visible camera or image entity and return it as an "
        "inline image for visual analysis. Use this to answer questions about what is currently "
        "visible. Only entities visible under the configured scope can be captured. "
        "Errors return {status:'error', error:{key, message, guidance?}}."
    )
