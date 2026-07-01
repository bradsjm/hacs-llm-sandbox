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
    "- Each async_call costs one helper call; reads cost zero. Respect the "
    "budget.\n"
    "- Errors are returned as helper_error. If the service name is wrong, the "
    "response lists valid services for the domain with brief schemas; correct and "
    "retry.\n"
)


ACTIONS_DISABLED_PROMPT = (
    "## Service calls (disabled)\n"
    "Service calls are disabled for this assistant. hass.services.async_call is "
    "rejected. Use the service-catalog reads (has_service, "
    "async_services_for_domain, supports_response), states, and registries "
    "only.\n"
)


def build_execute_home_code_description() -> str:
    """Return the execute_home_code tool description."""
    return (
        "Execute bounded Python/Monty code against a frozen, read-only Home Assistant view. "
        "Read states and registries using the native Home Assistant patterns documented in the API prompt. "
        "Service-call availability follows the API prompt. "
        "Returns {execution, output, printed, actions}. execution.status is "
        "ok | code_error | helper_error | setup_error; use output only when status is ok. "
        "printed holds captured print() lines. actions lists service calls with status, "
        "response, and error details."
    )


def build_get_history_description() -> str:
    """Return the get_history tool description."""
    return (
        "Return raw state-value history (each recorded state row) for visible "
        "entities over a bounded UTC window; use when you need how a state or "
        "attribute changed over time. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Returns {status, window, entities, truncated}."
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
        "Returns {status, window, period, statistics, truncated}."
    )


def build_get_logbook_description() -> str:
    """Return the get_logbook tool description."""
    return (
        "Return human-readable logbook entries (the activity/events timeline — "
        "what happened and why) for visible entities over a bounded UTC window; "
        "use for 'what happened with X', activity, or a timeline. "
        "Scope with entity_ids or HA-native selectors (area_id/device_id/floor_id/label_id/domain); "
        "size the window with hours=<n> or ISO start/end. "
        "Returns {status, window, entries, truncated}."
    )


def build_get_camera_image_description() -> str:
    """Return the get_camera_image tool description."""
    return (
        "Capture a single live frame from a visible camera or image entity and return it as an "
        "inline image for visual analysis. Use this to answer questions about what is currently "
        "visible. Only entities visible under the configured scope can be captured."
    )
