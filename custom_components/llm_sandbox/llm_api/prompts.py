"""Static LLM-facing prompts for the LLM Sandbox facade runtime.

This module is the single home for prose that is sent to the model and that
does not depend on runtime state. Runtime composition (conditional sections)
stays in ``api.py`` next to the state it renders from.
"""

BASE_API_PROMPT = (
    "## Tools\n"
    "LLM Sandbox exposes one tool: execute_home_code. It runs bounded "
    "Python/Monty against a frozen snapshot of your Home Assistant.\n"
    "\n"
    "## Globals\n"
    "The following globals are pre-bound (no imports needed):\n"
    "- hass: the Home Assistant root. Exposes hass.states (StateMachine) and "
    "hass.services (service catalog + propose-only async_call).\n"
    "- states: hass.states (same object). Read entity state.\n"
    "- er, dr, ar, fr: module facades. Pass hass only to these facades: "
    "er.async_get(hass), dr.async_get(hass), ar.async_get(hass), and "
    "fr.async_get(hass) return registry instances.\n"
    "- entity_registry, device_registry, area_registry, floor_registry: the "
    "already-resolved registry instances. Do not call async_get(hass) on these "
    "globals; their methods take entity IDs, device IDs, area IDs, floor IDs, "
    "or names, not hass.\n"
    "- now: the frozen snapshot creation time as an ISO string. Use it as the "
    "reference time for comparisons with State timestamp strings.\n"
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
    "- Module style: call async_get(hass) only on er/dr/ar/fr, for example "
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
    "- dr.async_entries_for_area(dr.async_get(hass), area_id) returns devices "
    "in an area.\n"
    "- area_registry.async_get_area_by_name('Bedroom') returns the matching "
    "area (by name or alias) or None.\n"
    "- area_registry.async_list_areas() returns area entries; use area.id, "
    "not area.area_id.\n"
    "- Area objects expose: id, name, aliases, floor_id, labels, icon, "
    "picture, humidity_entity_id, temperature_entity_id.\n"
    "- Use area.id as the area_id argument for er/dr area lookup helpers.\n"
    "- floor_registry.async_get_floor_by_name('Ground Floor') returns the "
    "matching floor or None.\n"
    "\n"
    "## Service calls (propose only)\n"
    "- hass.services.has_service('light', 'turn_on') checks if a service exists.\n"
    "- hass.services.async_services() returns the service catalog with "
    "supports_response values (none, optional, or only).\n"
    "- await hass.services.async_call('light', 'turn_on', "
    "{'brightness_pct': 80}, target={'entity_id': 'light.bedroom'}) proposes a "
    "service call. The call is NOT executed; it is recorded as a proposed action "
    "and returned in the response for the caller to confirm or execute later.\n"
    "- async_call costs one helper call. Reads cost zero.\n"
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
    "- Avoid filesystem, network, OS/process, pathlib/open, and current-time "
    "calls such as datetime.now() or date.today(). Snapshot timestamps are ISO "
    "strings; compare those strings directly when possible.\n"
    "- Reflection/introspection builtins are unavailable (getattr, setattr, "
    "hasattr, delattr, dir, vars, type).\n"
    "- The live hass object, event bus, config, auth, filesystem, network, and "
    "OS/process APIs are not exposed.\n"
)


EXECUTE_HOME_CODE_OUTPUT = (
    "Returns {execution, output, printed, proposed_actions}. execution.status "
    "is ok, code_error, helper_error, or setup_error. Use output only when "
    "status is ok. printed holds captured print() lines. proposed_actions lists "
    "service calls recorded by hass.services.async_call (not yet executed)."
)


def build_execute_home_code_description() -> str:
    """Return the execute_home_code tool description."""
    return "\n".join(
        [
            "Execute bounded Python/Monty code against a frozen Home Assistant snapshot.",
            "Read states and registries using native Home Assistant API patterns.",
            "Propose service calls via hass.services.async_call (recorded, not executed).",
            EXECUTE_HOME_CODE_OUTPUT,
        ]
    )
