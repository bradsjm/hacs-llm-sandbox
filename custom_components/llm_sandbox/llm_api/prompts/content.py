"""Action sections and tool description builders for LLM-facing prompts."""

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
    "- Errors include a specific message and may include fix candidates such as "
    "valid domains, services, entities, or fields to correct in one retry.\n"
)


ACTIONS_DISABLED_PROMPT = (
    "## Service calls (disabled)\n"
    "Service calls are disabled for this assistant. hass.services.async_call is "
    "rejected. Use the service-catalog reads (has_service, "
    "async_services_for_domain, async_services_for_target, supports_response), "
    "states, and registries only.\n"
)


def compose_system_prompt(base_prompt: str, actions_enabled: bool, location_section: str | None = None) -> str:
    """Return the base API prompt plus action and optional location sections."""
    # Exactly one service-call section per instance: live-call guidance when
    # actions are enabled, the rejection notice otherwise. The static tool
    # descriptions stay action-neutral.
    section = ACTIONS_ENABLED_PROMPT if actions_enabled else ACTIONS_DISABLED_PROMPT
    prompt = f"{base_prompt}\n\n{section}"
    if location_section is None:
        return prompt
    return f"{prompt}\n\n{location_section}"


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
        "Service-call availability follows the API prompt. "
        "Success returns {execution:{status:'ok'}, output:<data>} with printed only when print() emitted lines. "
        "If output is empty because a literal entity id is missing, note gives one imperative retry hint naming "
        "the missing id and a visible replacement. Errors return {execution:{status:'code_error'|'helper_error'|"
        "'setup_error', kind?, message, fix?}, output:null}; message is the specific failure and fix lists "
        "concrete candidate globals, attributes, services, domains, or entity ids to correct in one retry. "
        "actions appears only when a service call was made and contains compact records "
        "{service, target, status, resolved_from?, error?}."
    )


def build_get_history_description() -> str:
    """Return the get_history tool description."""
    return (
        "Return raw state-value history for visible "
        "entities over a bounded UTC window; use when you need how a state or "
        "attribute changed over time. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Success returns {window, entities}, where entities is keyed by entity_id and each value has "
        "rows of [t, state] plus unit when known; rows omit attributes by default — pass attributes "
        "(a list of attribute names, opt-in) to append a {name: value} element per row carrying only the "
        "requested attributes that exist (bounded count). The first page returns the newest rows; when more "
        "remain, next_cursor appears — pass it back as cursor (omit window args) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, fix?}}; entity_not_visible fix names "
        "concrete visible entity candidates."
    )


def build_get_statistics_description() -> str:
    """Return the get_statistics tool description."""
    return (
        "Return pre-aggregated long-term statistics (mean/min/max and raw "
        "recorder units) for visible statistic IDs over a bounded UTC window. "
        "These are historical aggregates over a period, not current values; for "
        "a current value or an average of current states, read states in "
        "execute_home_code instead. Each statistic ID must be a currently-visible "
        "entity ID; external or non-entity statistic IDs are rejected. Scope with "
        "statistic_ids or HA-native selectors "
        "(area_id/device_id/floor_id/label_id/domain); size the window with hours=<n> or ISO start/end. "
        "Success returns {window, period, statistics}, where statistics is keyed by statistic/entity id and each "
        "value has rows of [t, value]. The first page returns the newest rows; when more remain, "
        "next_cursor appears — pass it back as cursor (omit window args) to fetch the next older page. "
        "Errors return {status:'error', error:{key, message, fix?}}; entity_not_visible fix names "
        "concrete visible entity candidates."
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
        "Errors return {status:'error', error:{key, message, fix?}}; entity_not_visible fix names "
        "concrete visible entity candidates."
    )


def build_get_camera_image_description() -> str:
    """Return the get_camera_image tool description."""
    return (
        "Capture a single live frame from a visible camera or image entity and return it as an "
        "inline image for visual analysis. Use this to answer questions about what is currently "
        "visible. Only entities visible under the configured scope can be captured. "
        "Errors return {status:'error', error:{key, message, fix?}}."
    )
