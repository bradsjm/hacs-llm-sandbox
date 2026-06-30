"""Static LLM-facing prompts for the LLM Sandbox facade runtime.

This module is the single home for prose that is sent to the model and that
does not depend on runtime state. Runtime composition (conditional sections)
stays in ``api.py`` next to the state it renders from.
"""

BASE_API_PROMPT = (
    "## Tools\n"
    "LLM Sandbox exposes four tools. `execute_home_code` runs bounded "
    "Python/Monty against a frozen read-only view of your Home Assistant. "
    "`get_history`, `get_statistics`, and `get_logbook` run bounded "
    "recorder queries (entity history, long-term statistics, logbook events) "
    "over a UTC time window you choose; they require the recorder and reject "
    "entities that are not visible to the sandbox.\n"
    "\n"
    "## Globals\n"
    "The following globals are pre-bound (no imports needed):\n"
    "- hass: the Home Assistant root. Exposes hass.states (StateMachine) and "
    "hass.services (service catalog + async_call), and hass.config. "
    "hass.config exposes frozen instance metadata: location_name, latitude, "
    "longitude, elevation, time_zone, language, country, currency, "
    "internal_url, external_url, and units such as units.temperature_unit and "
    "units.length_unit.\n"
    "- states: hass.states (same object). Read entity state.\n"
    "- er, dr, ar, fr, lr, cr: module facades. Pass hass only to these "
    "facades: er.async_get(hass), dr.async_get(hass), ar.async_get(hass), "
    "fr.async_get(hass), lr.async_get(hass), and cr.async_get(hass) return "
    "registry instances.\n"
    "- entity_registry, device_registry, area_registry, floor_registry: the "
    "already-resolved registry instances. Do not call async_get(hass) on these "
    "globals; their methods take entity IDs, device IDs, area IDs, floor IDs, "
    "or names, not hass.\n"
    "- now: the frozen view creation time as an ISO string. Use it as the "
    "reference time for comparisons with State timestamp strings.\n"
    "- date: frozen date facade. date.today() returns the snapshot date in "
    "the HA timezone; date.fromisoformat(s) parses an ISO date string. "
    "Results expose iso (YYYY-MM-DD), year, month, day, weekday, and "
    "isoformat().\n"
    "- datetime: frozen datetime facade. datetime.now() returns the snapshot "
    "time in the HA timezone; datetime.utcnow() returns UTC; "
    "datetime.fromisoformat(s) parses an ISO datetime string. Results expose "
    "iso, year, month, day, hour, minute, second, microsecond, weekday, "
    "date(), and isoformat(). These are frozen snapshot values, not live "
    "wall-clock time.\n"
    "- llm_context: the current request context (platform, language, assistant, "
    "device_id, area_id, area_name, floor_id, floor_name, context).\n"
    "\n"
    "## Reading state\n"
    "- hass.states.get('light.bedroom') returns the State or None.\n"
    "- hass.states.async_all() returns all states; pass a domain string to "
    "filter.\n"
    "- hass.states.is_state('light.bedroom', 'on') returns a bool.\n"
    "- State objects expose: entity_id, domain, object_id, name, state, "
    "attributes, last_changed (ISO string), last_updated (ISO string), "
    "last_reported (ISO string or None), context.\n"
    "\n"
    "## Reading registries\n"
    "Choose one registry access style and do not mix them:\n"
    "- Instance style (simplest): use the pre-bound registry instances directly, "
    "for example device_registry.async_get('<device_id>').\n"
    "- Module style: call async_get(hass) only on er/dr/ar/fr/lr/cr, for example "
    "dr.async_get(hass).async_get('<device_id>').\n"
    "- Wrong: device_registry.async_get(hass). The global device_registry is "
    "already an instance, and async_get on it expects a device_id string.\n"
    "- list(entity_registry.entities.values()) lists all entity registry entries.\n"
    "- list(device_registry.devices.values()) lists all device entries.\n"
    "- er.async_entries_for_area(er.async_get(hass), area_id) returns entity "
    "entries in an area. Effective area = entity.area_id or device.area_id.\n"
    "- er.async_entries_for_device(er.async_get(hass), device_id) returns "
    "entity entries for a device.\n"
    "- er.async_entries_for_label(er.async_get(hass), label_id) returns entity "
    "entries with a label.\n"
    "- er.async_get_entity(er.async_get(hass), domain, platform, unique_id) "
    "returns the matching entity_id or None.\n"
    "- er.async_entries(er.async_get(hass)) returns all entity registry entries.\n"
    "- dr.async_entries_for_area(dr.async_get(hass), area_id) returns devices "
    "in an area.\n"
    "- dr.async_entries_for_label(dr.async_get(hass), label_id) returns "
    "devices with a label.\n"
    "- area_registry.async_get_area_by_name('Bedroom') returns the matching "
    "area (by name or alias) or None.\n"
    "- area_registry.async_list_areas() returns area entries.\n"
    "- Area objects expose: id, name, aliases, floor_id, labels, icon, "
    "picture, humidity_entity_id, temperature_entity_id.\n"
    "- Use area.id as the area_id argument for er/dr area lookup helpers.\n"
    "- floor_registry.async_get_floor_by_name('Ground Floor') returns the "
    "matching floor or None.\n"
    "- floor_registry.async_list_floors() returns floor entries.\n"
    "- Floor objects expose: floor_id, name, aliases, level, icon, created_at, "
    "modified_at.\n"
    "- lr, cr: module facades for labels and categories. Pass hass only: "
    "lr.async_get(hass), cr.async_get(hass).\n"
    "- label_registry: the already-resolved label registry instance.\n"
    "- label_registry.async_get_label_by_name('Favourites') returns the matching "
    "label (normalized-name match: case- and whitespace-insensitive) or None.\n"
    "- label_registry.async_list_labels() returns label entries.\n"
    "- Label objects expose: label_id, name, normalized_name, description, color, "
    "icon, created_at, modified_at.\n"
    "- Use label.label_id as the label_id argument for er/dr label lookup helpers "
    "and for service target={'label_id': ...}.\n"
    "- category_registry: the already-resolved category registry instance. "
    "Categories are scoped (per integration); pass scope as a keyword: "
    "category_registry.async_list_categories(scope='<scope>') and "
    "category_registry.async_get_category(scope='<scope>', category_id='<id>').\n"
    "- Category objects expose: category_id, scope, name, icon, created_at, "
    "modified_at.\n"
    "\n"
    "## Repairs\n"
    "- repairs: read-only view of Home Assistant repairs issues (issue registry).\n"
    "- repairs.async_issues() returns all issues; repairs.async_active_issues() "
    "returns active ones; repairs.async_dismissed_issues() returns dismissed ones.\n"
    "- repairs.async_issues_for_domain('light') and "
    "repairs.async_issues_by_severity('warning') filter the list.\n"
    "- repairs.async_get_issue('light', 'my_issue') returns one issue or None.\n"
    "- Issue objects expose: issue_id, domain, severity (critical/error/warning or "
    "None), active, dismissed_version, translation_key, "
    "translation_placeholders, created.\n"
    "- Issue severity and state reflect the frozen snapshot taken before code ran.\n"
    "\n"
    "## Config entries\n"
    "- config_entries: read-only, secret-stripped view of Home Assistant config "
    "entries.\n"
    "- config_entries.async_entries() returns all entries; pass a domain string to "
    "filter, e.g. config_entries.async_entries('light').\n"
    "- config_entries.async_get_entry('<entry_id>') returns one entry or None.\n"
    "- Config-entry objects expose ONLY: entry_id, domain, title, source, state "
    "(loaded/setup_error/not_loaded/...), unique_id, disabled_by, reason. "
    "Credentials in entry data/options are never exposed.\n"
    "\n"
    "## Recorder tools\n"
    "- get_history requires entity_ids and accepts optional ISO-8601 start/end "
    "timestamps. Omitted timestamps default to the last 1 hour.\n"
    "- get_statistics requires statistic_ids, accepts optional ISO-8601 "
    "start/end timestamps, and accepts period '5minute', 'hour', or 'day' "
    "(default 'hour'). Omitted timestamps default to the last 24 hours.\n"
    "- get_logbook requires entity_ids and accepts optional ISO-8601 start/end "
    "timestamps. Omitted timestamps default to the last 24 hours.\n"
    "- Recorder windows are UTC and capped at 24 hours for history/logbook "
    "and 30 days for statistics.\n"
    "- Recorder results use ISO-8601 UTC timestamps and include a truncated "
    "boolean when output rows are capped.\n"
    "\n"
    "## Service calls\n"
    "- hass.services.has_service('light', 'turn_on') checks if a service exists.\n"
    "- hass.services.async_services() returns this metadata shape for all "
    "domains.\n"
    "- hass.services.async_services_for_domain('light') returns one domain's "
    "per-service metadata: supports_response (none, optional, or only), fields "
    "(a list of {name, required, type_hint, description} parameter briefs), "
    "and dynamic (bool; when true, listed fields are non-exhaustive). It "
    "returns {} for an unknown domain.\n"
    "- hass.services.supports_response('light', 'turn_on') returns response-mode "
    "metadata (none, optional, or only); it is not a service-existence check.\n"
    "- await hass.services.async_call('light', 'turn_on', "
    "{'brightness_pct': 80}, target={'entity_id': 'light.bedroom'}) performs "
    "the service call for real. Keep service data and target separate; put "
    "entities in target={'entity_id': ...}.\n"
    "- blocking=True waits for completion. blocking=False is fire-and-forget "
    "and yields no result and no detailed outcome error, so prefer blocking "
    "when you need to know whether the action succeeded.\n"
    "- return_response=True is required for services that produce a response, "
    "returns the response dict, and requires blocking=True.\n"
    "- The frozen snapshot is taken before actions run. A follow-up read in the "
    "same execute_home_code call will not reflect the change; call "
    "execute_home_code again to observe new state.\n"
    "- Actions apply sequentially with no rollback. If a later call fails, "
    "earlier calls have already happened.\n"
    "- Service-call errors are captured as helper_error with failed-call "
    "details. If the service name is wrong, the response includes valid "
    "services for that domain with brief parameter schemas; correct and retry.\n"
    "- async_call costs one helper call. Reads cost zero; respect the budget.\n"
    "\n"
    "## Execution rules\n"
    "- Assign the final answer to result, or end the code with a bare expression "
    "to have it promoted automatically.\n"
    "- print() output is captured into the printed field.\n"
    "- Keep code self-contained. Use the pre-bound globals and plain Python "
    "loops/comprehensions instead of importing helpers.\n"
    "- Standard builtins are available, including len, sum, min, max, sorted, "
    "dict, list, set, tuple, enumerate, zip, round, range, abs, any, all, map, "
    "filter, int, float, str, bool, and bytes.\n"
    "- Imports are limited. json, math, and re work for basic data handling; "
    "avoid other stdlib modules unless already proven necessary.\n"
    "- Do not import collections. Counter/defaultdict are unavailable; count "
    "with a dict loop like counts[key] = counts.get(key, 0) + 1.\n"
    "- Avoid filesystem, network, OS/process, and pathlib/open calls.\n"
    "- date.today(), datetime.now(), datetime.utcnow(), and "
    "datetime.fromisoformat() are supported through frozen facades (see the "
    "globals section). timedelta, timezone, strftime, strptime, and datetime "
    "arithmetic are not available.\n"
    "- View timestamps (State.last_changed, last_updated, etc.) are ISO "
    "strings; compare them directly or parse with datetime.fromisoformat().\n"
    "- Reflection/introspection builtins are partially unavailable: dir, vars, "
    "setattr, and delattr are not available in the sandbox.\n"
    "- The live hass object, event bus, config, auth, filesystem, network, and "
    "OS/process APIs are not exposed.\n"
)


ACTIONS_DISABLED_PROMPT = (
    "## Service calls (disabled)\n"
    "Service calls are disabled for this assistant. Do not call "
    "hass.services.async_call; it will be rejected. Read states and "
    "registries only.\n"
)


EXECUTE_HOME_CODE_OUTPUT = (
    "Returns {execution, output, printed, actions}. execution.status "
    "is ok, code_error, helper_error, or setup_error. Use output only when "
    "status is ok. printed holds captured print() lines. actions lists "
    "service calls from hass.services.async_call with status, response, and "
    "error details."
)


def build_execute_home_code_description() -> str:
    """Return the execute_home_code tool description."""
    return "\n".join(
        [
            "Execute bounded Python/Monty code against a frozen read-only Home Assistant view.",
            "Read states and registries using native Home Assistant API patterns.",
            "Perform service calls via hass.services.async_call with structured action results.",
            EXECUTE_HOME_CODE_OUTPUT,
        ]
    )


def build_get_history_description() -> str:
    """Return the get_history tool description."""
    return "\n".join(
        [
            "Return recorded state history for one or more visible entities over a bounded UTC window (default last 1h, max 24h).",
            "Pass entity_ids and optional ISO-8601 start/end.",
            "Returns {status, window, entities: {entity_id: [{state, attributes, last_changed, last_updated}]}, truncated}.",
        ]
    )


def build_get_statistics_description() -> str:
    """Return the get_statistics tool description."""
    return "\n".join(
        [
            "Return long-term recorder statistics for one or more visible statistic IDs over a bounded UTC window (default last 24h, max 30 days).",
            "Pass statistic_ids, optional start/end, and period ('5minute'|'hour'|'day', default 'hour').",
            "Returns {status, window, period, statistics: {statistic_id: [{start, mean, min, max, sum, state, last_reset?}]}, truncated}. Units are raw recorder units.",
        ]
    )


def build_get_logbook_description() -> str:
    """Return the get_logbook tool description."""
    return "\n".join(
        [
            "Return logbook events for one or more visible entities over a bounded UTC window (default last 24h, max 24h).",
            "Pass entity_ids (required) and optional ISO-8601 start/end.",
            "Returns {status, window, entries: [{when, name, message, entity_id, state, ...}], truncated}.",
        ]
    )
