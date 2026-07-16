"""Canonical capability catalog rendered into each shipped prompt profile."""

from ..data.home_db import render_query_schema_prompt

_HISTORY_CONTRACT = (
    "hass.history accepts exact start/end, entity IDs or snapshot-backed area/device/floor/label/domain scope, "
    "and analytics arguments; results stay flat with no cursor/window."
)


_CONFIG_UNITS: tuple[str, ...] = (
    "temperature_unit",
    "length_unit",
    "mass_unit",
    "pressure_unit",
    "volume_unit",
    "area_unit",
    "wind_speed_unit",
    "accumulated_precipitation_unit",
)
_CONTEXT_FIELDS: tuple[str, ...] = (
    "platform",
    "language",
    "assistant",
    "device_id",
    "area_id",
    "area_name",
    "floor_id",
    "floor_name",
    "context",
)
_STATE_FIELDS: tuple[str, ...] = (
    "entity_id",
    "domain",
    "object_id",
    "name",
    "state",
    "attributes",
    "last_changed",
    "last_changed_timestamp",
    "last_updated",
    "last_updated_timestamp",
    "last_reported",
    "last_reported_timestamp",
    "context",
    "area_id",
    "floor_id",
    "device_id",
    "platform",
    "unique_id",
)
_REGISTRY_ALIASES: tuple[str, ...] = (
    "er/entity_registry",
    "dr/device_registry",
    "ar/area_registry",
    "fr/floor_registry",
    "lr/label_registry",
    "cr/category_registry",
)
_AREA_FIELDS: tuple[str, ...] = (
    "id",
    "area_id",
    "name",
    "aliases",
    "floor_id",
    "labels",
    "icon",
    "picture",
    "humidity_entity_id",
    "temperature_entity_id",
    "created_at",
    "modified_at",
)
_FLOOR_FIELDS: tuple[str, ...] = ("floor_id", "id", "name", "aliases", "level", "icon", "created_at", "modified_at")
_LABEL_FIELDS: tuple[str, ...] = (
    "label_id",
    "name",
    "normalized_name",
    "description",
    "color",
    "icon",
    "created_at",
    "modified_at",
)
_CATEGORY_FIELDS: tuple[str, ...] = ("category_id", "scope", "name", "icon", "created_at", "modified_at")
_DEVICE_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "name_by_user",
    "manufacturer",
    "model",
    "model_id",
    "sw_version",
    "hw_version",
    "serial_number",
    "area_id",
    "labels",
    "identifiers",
    "connections",
    "configuration_url",
    "entry_type",
    "config_entries",
    "via_device_id",
    "disabled_by",
)
_ENTITY_ENTRY_FIELDS: tuple[str, ...] = (
    "domain",
    "entity_id",
    "unique_id",
    "platform",
    "config_entry_id",
    "device_id",
    "area_id",
    "name",
    "original_name",
    "aliases",
    "labels",
    "disabled_by",
    "hidden_by",
    "entity_category",
    "device_class",
    "original_device_class",
    "capabilities",
    "supported_features",
    "translation_key",
    "has_entity_name",
)
_ISSUE_FIELDS: tuple[str, ...] = (
    "issue_id",
    "domain",
    "severity",
    "active",
    "dismissed_version",
    "translation_key",
    "translation_placeholders",
    "created",
)
_NOTIFICATION_FIELDS: tuple[str, ...] = ("notification_id", "title", "message", "created_at")
_CONFIG_ENTRY_FIELDS: tuple[str, ...] = (
    "entry_id",
    "domain",
    "title",
    "source",
    "state",
    "unique_id",
    "disabled_by",
    "reason",
)
_BUILTINS: tuple[str, ...] = (
    "len",
    "sum",
    "min",
    "max",
    "sorted",
    "dict",
    "list",
    "set",
    "tuple",
    "enumerate",
    "zip",
    "round",
    "range",
    "abs",
    "any",
    "all",
    "map",
    "filter",
    "int",
    "float",
    "str",
    "bool",
)
_ALLOWED_IMPORTS: tuple[str, ...] = ("json", "math", "re")


def _items(values: tuple[str, ...]) -> str:
    """Render an authored capability tuple consistently."""
    return ", ".join(values)


def render_capability_catalog(*, compact: bool) -> str:
    """Render every sandbox capability without selecting a profile-specific subset."""
    if compact:
        return _render_compact_catalog()
    return _render_readable_catalog()


def _render_readable_catalog() -> str:
    """Render the readable form used by Guided and Balanced."""
    return "\n\n".join(
        (
            "## Snapshot and output\n"
            "- Each tool call receives a fresh, frozen visible snapshot. Service calls do not change reads in the same "
            "execute_home_code call; call again to observe changes. Discover real IDs and names from visible data; "
            "instruction placeholders such as <entity_id> are not real IDs.\n"
            "- Await hass.history(...), hass.query(...), hass.logbook(...), and enabled hass.services.async_call(...). "
            "State, registry, config, repairs, notification, and service-catalog reads are synchronous despite async_-style names. "
            "Final output must be serializable: assign result or end with a bare expression. print() is captured in printed.",
            "## Globals and frozen time\n"
            "- Pre-bound globals: hass (frozen facade with states, services, and config metadata), states (hass.states), "
            "now (ISO snapshot time), date, datetime, llm_context, "
            f"{_items(_REGISTRY_ALIASES)}, repairs, persistent_notifications, config_entries. hass.config exposes "
            "location_name, latitude, longitude, elevation, time_zone, language, country, currency, internal_url, "
            f"external_url, and units: {_items(_CONFIG_UNITS)}; it is not live Home Assistant.\n"
            "- llm_context fields: "
            f"{_items(_CONTEXT_FIELDS)}; use attributes or llm_context.get('<key>'). date/datetime support "
            "date.today(), date.fromisoformat(s), datetime.now(), datetime.utcnow(), datetime.fromisoformat(s), and "
            "SafeDateTime.timestamp. There is no live wall clock, timedelta, timezone, strftime, strptime, or "
            "SafeDateTime subtraction; subtract numeric timestamps for durations.",
            "## State, history, and logbook\n"
            "- Await hass.history(...) for bounded recorder-backed state history and hass.logbook(...) for bounded activity "
            "entries. hass.history(...) returns a flat list of {entity_id, when, state, value} rows (or flat analytics "
            "result dicts); it does not return the standalone get_history envelope, cursor, or window. Raw results above "
            "1000 rows are capped and reported through the top-level overflow.history field and a note. "
            f"{_HISTORY_CONTRACT}\n"
            "hass.states.get('<entity_id>') -> State | None; hass.states.async_all('<domain>') lists visible "
            "states, optionally by domain; hass.states.is_state('<entity_id>', '<state>') -> bool.\n"
            f"- State fields: {_items(_STATE_FIELDS)}.",
            "## Registries and record fields\n"
            "- Short and long registry globals are equivalent. Resolve with er.async_get(hass), dr.async_get(hass), "
            "ar.async_get(hass), fr.async_get(hass), lr.async_get(hass), cr.async_get(hass). Look up with "
            "entity_registry.async_get('<entity_id>') or device_registry.async_get('<device_id>').\n"
            "- Name/list methods: area_registry.async_get_area_by_name('<area_name>'), async_list_areas(); "
            "floor_registry.async_get_floor_by_name('<floor_name>'), async_list_floors(); "
            "label_registry.async_get_label_by_name('<label_name>'), async_list_labels(); "
            "category_registry.async_list_categories(scope='<scope>'), async_get_category(scope='<scope>', category_id='<id>'). "
            "List entities/devices with list(entity_registry.entities.values()) and list(device_registry.devices.values()).\n"
            "- Traverse with er.async_entries_for_area('<area_id>'), async_entries_for_device('<device_id>'), "
            "async_entries_for_label('<label_id>'), async_get_entity(er.async_get(hass), '<domain>', '<platform>', '<unique_id>'), "
            "async_entries(), dr.async_entries_for_area('<area_id>'), and async_entries_for_label('<label_id>'). "
            "HA two-argument forms that pass er.async_get(hass) first are accepted. Effective area is entity.area_id or "
            "device.area_id; use area.id, label.label_id, and floor.floor_id as IDs.\n"
            f"- Area: {_items(_AREA_FIELDS)}.\n"
            f"- Floor: {_items(_FLOOR_FIELDS)}.\n"
            f"- Label: {_items(_LABEL_FIELDS)}.\n"
            f"- Category: {_items(_CATEGORY_FIELDS)}.\n"
            f"- Device: {_items(_DEVICE_FIELDS)}.\n"
            f"- Entity entry: {_items(_ENTITY_ENTRY_FIELDS)}.",
            "## Repairs\n"
            "- repairs is read-only: repairs.async_issues(), repairs.async_active_issues(), repairs.async_dismissed_issues(), "
            "repairs.async_issues_for_domain('<domain>'), repairs.async_issues_by_severity('<severity>'), "
            "repairs.async_get_issue('<domain>', '<issue_id>').\n"
            f"- Issue fields: {_items(_ISSUE_FIELDS)}; severity is critical/error/warning or None.",
            "## Persistent notifications\n"
            "- persistent_notifications is a read-only notification-store view: persistent_notifications.async_get_notifications() and "
            "persistent_notifications.async_get_notification('<notification_id>'). Prefer it over hass.states.async_all('persistent_notification'), "
            "whose states may be hidden by scope.\n"
            f"- Notification fields: {_items(_NOTIFICATION_FIELDS)}.",
            "## Config entries\n"
            "- config_entries is read-only and secret-stripped: config_entries.async_entries('<domain>') and "
            "config_entries.async_get_entry('<entry_id>'). Credentials are never exposed.\n"
            f"- Entry fields: {_items(_CONFIG_ENTRY_FIELDS)}.",
            "## Service catalog reads\n"
            "- hass.services.has_service('<domain>', '<service>') -> bool. "
            "async_services_for_domain('<domain>') returns per-service supports_response (none/optional/only), fields "
            "[{name, required, type_hint, description}], and dynamic (true means fields are non-exhaustive); unknown domains "
            "return {}. async_services_for_target({'entity_id'|'device_id'|'area_id'|'label_id'|'floor_id': ...}) returns "
            "services targeting resolved entities with response mode and field names. supports_response('<domain>', '<service>') "
            "returns response mode, not existence.",
            render_query_schema_prompt(),
            "## Execution surface and unavailable APIs\n"
            "- Keep code self-contained and use pre-bound globals/plain Python. Builtins: "
            f"{_items(_BUILTINS)}; map()/filter() return lists. Imports only: {_items(_ALLOWED_IMPORTS)}.\n"
            "- No filesystem, network, OS/process, pathlib/open, dir, vars, setattr, delattr, collections, statistics, or itertools. "
            "The live hass object, registries, event bus, auth, config internals, filesystem, network, and OS/process APIs are not exposed.\n"
            "- Duration example: datetime.now().timestamp - hass.states['<entity_id>'].last_changed_timestamp.",
        )
    )


def _render_compact_catalog() -> str:
    """Render the Frontier form from the same authored capability tuples."""
    return "\n\n".join(
        (
            "## Snapshot/output\n"
            "- Fresh frozen visible snapshot per call; service calls do not update same-call reads. Discover visible IDs/names; "
            "<entity_id> is a placeholder. Await hass.history(...),hass.query(...),hass.logbook(...),enabled "
            "hass.services.async_call(...); state/registry/config/repairs/notification/service-catalog reads sync. "
            "Serializable result or final bare expression; print()->printed.",
            "## Globals/time\n"
            "- hass=frozen states/services/config facade; states=hass.states; now=ISO snapshot time; date/datetime; llm_context; "
            f"{_items(_REGISTRY_ALIASES)}; repairs; persistent_notifications; config_entries. config: location_name,latitude,longitude,"
            "elevation,time_zone,language,country,currency,internal_url,external_url,units."
            f"{_items(_CONFIG_UNITS)}. Context: {_items(_CONTEXT_FIELDS)} via attr/get(). date/datetime: today/fromisoformat/"
            "now/utcnow/timestamp only; no live clock,timedelta,timezone,strftime,strptime,SafeDateTime subtraction; use numeric timestamps.",
            "## State/history/logbook\n"
            "- await hass.history(...), hass.logbook(...). hass.history(...) returns a flat list of "
            "{entity_id,when,state,value} rows or flat analytics result dicts; it has no standalone get_history "
            "envelope/cursor/window. Raw results above 1000 rows are capped and reported through top-level "
            "overflow.history plus a note. states.get(id)->State|None; async_all(domain); is_state(id,state). "
            f"{_HISTORY_CONTRACT} "
            f"State: {_items(_STATE_FIELDS)}.",
            "## Registries/records\n"
            "- Resolve er/dr/ar/fr/lr/cr with async_get(hass); aliases: "
            f"{_items(_REGISTRY_ALIASES)}. Lookup entity_registry.async_get(id),device_registry.async_get(id). "
            "Name/list: area/floor/label async_get_*_by_name + async_list_*; category async_list_categories(scope=) / "
            "async_get_category(scope=,category_id=); lists: entity_registry.entities.values(),device_registry.devices.values(). "
            "Traverse er.async_entries_for_area/device/label,async_get_entity(er.async_get(hass),domain,platform,unique_id),"
            "async_entries; dr.async_entries_for_area/label; HA two-arg forms accepted. Effective area=entity.area_id or device.area_id; "
            "use area.id,label.label_id,floor.floor_id.\n"
            f"- Area({_items(_AREA_FIELDS)}); Floor({_items(_FLOOR_FIELDS)}); Label({_items(_LABEL_FIELDS)}); "
            f"Category({_items(_CATEGORY_FIELDS)}); Device({_items(_DEVICE_FIELDS)}); Entity({_items(_ENTITY_ENTRY_FIELDS)}).",
            "## Repairs\n"
            "- repairs.async_issues()/repairs.async_active_issues()/repairs.async_dismissed_issues()/repairs.async_issues_for_domain(domain)/"
            "repairs.async_issues_by_severity(severity)/repairs.async_get_issue(domain,issue_id). "
            f"Issue: {_items(_ISSUE_FIELDS)}; severity=critical/error/warning|None.",
            "## Notifications\n"
            "- persistent_notifications.async_get_notifications()/persistent_notifications.async_get_notification(id); prefer to hidden persistent_notification "
            f"states. Notification: {_items(_NOTIFICATION_FIELDS)}.",
            "## Config entries\n"
            "- Secret-stripped read-only config_entries.async_entries(domain)/async_get_entry(id); no credentials. "
            f"Entry: {_items(_CONFIG_ENTRY_FIELDS)}.",
            "## Service catalog\n"
            "- hass.services.has_service(domain,service); async_services_for_domain(domain)->supports_response(none/optional/only),"
            "fields[{name,required,type_hint,description}],dynamic (true=non-exhaustive),{} unknown; "
            "async_services_for_target({entity_id|device_id|area_id|label_id|floor_id:...})->resolved-target services,response/fields; "
            "supports_response(domain,service)=response mode, not existence.",
            render_query_schema_prompt(compact=True),
            "## Execution/unavailable\n"
            f"- Self-contained pre-bound globals/plain Python. Builtins: {_items(_BUILTINS)}; map/filter return lists. Imports: {_items(_ALLOWED_IMPORTS)}. "
            "No filesystem/network/OS/process/pathlib/open/dir/vars/setattr/delattr/collections/statistics/itertools; no live hass, "
            "registries,event bus,auth,config internals. Duration: datetime.now().timestamp - hass.states['<entity_id>'].last_changed_timestamp.",
        )
    )
