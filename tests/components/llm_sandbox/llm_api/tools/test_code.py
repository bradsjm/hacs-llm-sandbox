"""End-to-end tests for the Monty executor with HA-native facades."""

import json
from collections.abc import Awaitable
from typing import Any

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.llm_api import executor
from custom_components.llm_sandbox.llm_api.tools.code import _execute
from homeassistant.core import Context, HomeAssistant, SupportsResponse
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def _run_code(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    code: str,
) -> dict[str, object]:
    """Run Monty code through the execute_home_code tool path."""
    llm_context = llm.LLMContext(
        platform="test",
        context=Context(),
        language="en",
        assistant=None,
        device_id=None,
    )
    return await _execute(hass, entry.entry_id, {"code": code}, llm_context)  # type: ignore[return-value]


async def test_read_state_and_registry_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify Monty can read states and registries through HA-native facades."""
    code = """
bedroom = area_registry.async_get_area_by_name("Bedroom")
er = entity_registry
bedroom_states = []
if bedroom is not None:
    bedroom_entries = er.async_entries_for_area(er.async_get(hass), bedroom.id)
    for entry in bedroom_entries:
        if entry.entity_id.split(".")[0] != "light":
            continue
        st = hass.states.get(entry.entity_id)
        if st is not None:
            bedroom_states.append({
                "entity_id": entry.entity_id,
                "state": st.state,
                "attributes": {k: v for k, v in st.attributes.items() if k in ("friendly_name", "brightness")},
            })
result = bedroom_states
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == [
        {
            "entity_id": "light.bedroom",
            "state": "on",
            "attributes": {"friendly_name": "Bedroom Light"},
        }
    ]


async def test_state_machine_sugar_subscript_and_contains(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify optional subscript sugar works alongside strict methods."""
    code = """
result = {
    "get": hass.states.get("light.bedroom").state,
    "subscript": states["light.bedroom"].state,
    "contains": "light.bedroom" in states,
    "len": len(states),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["get"] == "on"
    assert output["subscript"] == "on"
    assert output["contains"] is True
    assert output["len"] >= 2


async def test_missing_entity_read_attaches_available_hint(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """An absent states.get yielding an empty result names visible same-domain entities."""
    code = """
result = hass.states.get("light.kitchen_main")
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] is None
    assert result["execution"]["referenced_missing"] == ["light.kitchen_main"]
    hint = result["execution"]["available_hint"]
    assert isinstance(hint, str)
    assert "light.bedroom" in hint
    assert "light.living_room" in hint


async def test_present_entity_read_attaches_no_hint(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """A successful read produces no referenced_missing tracking or available_hint."""
    code = """
state = hass.states.get("light.bedroom")
result = state.state if state is not None else None
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "on"
    assert result["execution"]["referenced_missing"] == []
    assert "available_hint" not in result["execution"]


async def test_map_filter_normalize_and_run(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """map()/filter() run natively and return lists end-to-end."""
    code = """
result = {
    "map": map(lambda x: x * 2, [1, 2, 3]),
    "filter": filter(lambda x: x > 1, [1, 2, 3]),
    "filter_none": filter(None, [0, 1, 2]),
    "map_multi": map(lambda a, b: a + b, [1, 2], [10, 20]),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["map"] == [2, 4, 6]
    assert output["filter"] == [2, 3]
    assert output["filter_none"] == [1, 2]
    assert output["map_multi"] == [11, 22]


@pytest.mark.parametrize(
    ("method", "id_expr"),
    [
        pytest.param("async_entries_for_area", "bedroom_id", id="area"),
        pytest.param("async_entries_for_device", "device_id", id="device"),
    ],
)
async def test_registry_traversal_one_and_two_arg_forms_match(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    method: str,
    id_expr: str,
) -> None:
    """One-arg and two-arg registry traversal forms resolve identically."""
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=loaded_entry.entry_id,
        identifiers={("test", "traversal-device")},
    )
    er.async_get(hass).async_update_entity("light.bedroom", device_id=device.id)
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")

    code = f"""
bedroom_id = "{bedroom.id}"
device_id = "{device.id}"
one_arg = [e.entity_id for e in er.{method}({id_expr})]
two_arg = [e.entity_id for e in er.{method}(er.async_get(hass), {id_expr})]
result = {{"one_arg": one_arg, "two_arg": two_arg}}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["one_arg"] == output["two_arg"]
    assert output["one_arg"] == ["light.bedroom"]


async def test_entity_entry_domain_field_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The derived domain field is readable through the entity registry facade."""
    code = """
entry = entity_registry.async_get("light.bedroom")
result = entry.domain
"""
    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "light"


async def test_service_catalog_reads(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify the service catalog snapshot is accessible."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    code = """
result = {
    "has_turn_on": hass.services.has_service("light", "turn_on"),
    "has_missing": hass.services.has_service("light", "nonexistent"),
    "domain_count": len(hass.services.async_services()),
    "services": hass.services.async_services(),
    "light_services": hass.services.async_services_for_domain("light"),
    "missing_services": hass.services.async_services_for_domain("missing"),
    "turn_on_response": hass.services.supports_response("light", "turn_on"),
    "required_response": hass.services.supports_response("test_response", "required"),
    "missing_response": hass.services.supports_response("missing", "missing"),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["has_turn_on"] is True
    assert output["has_missing"] is False
    assert output["domain_count"] >= 1
    assert output["services"]["light"]["turn_on"]["supports_response"] == "none"
    assert output["services"]["light"]["turn_on"]["fields"] == []
    assert isinstance(output["services"]["light"]["turn_on"]["fields"], list)
    assert isinstance(output["services"]["light"]["turn_on"]["dynamic"], bool)
    assert output["services"]["test_response"]["required"]["supports_response"] == "only"
    assert output["light_services"]["turn_on"]["supports_response"] == "none"
    assert output["light_services"]["turn_on"]["fields"] == []
    assert isinstance(output["light_services"]["turn_on"]["fields"], list)
    assert isinstance(output["light_services"]["turn_on"]["dynamic"], bool)
    assert output["missing_services"] == {}
    assert output["turn_on_response"] == "none"
    assert output["required_response"] == "only"
    assert output["missing_response"] == "none"


async def test_service_schema_fields_flow_through_read_path(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify service schema fields are visible through the Monty read facade."""
    hass.services.async_register(
        "schema_test",
        "do_thing",
        lambda call: None,
        schema=vol.Schema(
            {
                vol.Required("count"): vol.Coerce(int),
                vol.Optional("label"): str,
            }
        ),
    )
    await hass.async_block_till_done()
    code = """
result = hass.services.async_services_for_domain("schema_test")
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    service = result["output"]["do_thing"]
    assert service["dynamic"] is False
    assert len(service["fields"]) == 2
    fields = {field["name"]: field for field in service["fields"]}
    assert fields["count"]["required"] is True
    assert fields["label"]["required"] is False


async def test_config_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify safe hass.config values are readable through the Monty root."""
    hass.config.location_name = "Test Home"
    hass.config.elevation = 42
    hass.config.country = "NL"
    code = """
result = {
    "location_name": hass.config.location_name,
    "elevation": hass.config.elevation,
    "time_zone": hass.config.time_zone,
    "country": hass.config.country,
    "temperature_unit": hass.config.units.temperature_unit,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["location_name"] == "Test Home"
    assert output["elevation"] == 42
    assert output["country"] == "NL"
    assert isinstance(output["time_zone"], str)
    assert isinstance(output["temperature_unit"], str)


async def test_device_label_and_entity_lookup_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify device label and entity module lookup helpers use the snapshot."""
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=loaded_entry.entry_id,
        identifiers={("test", "dev1")},
        name="Labelled Device",
    )
    device_registry.async_update_device(device.id, labels={"fav"})
    # Link the visible Bedroom light to this device so the device survives the
    # snapshot's visibility filtering (entity-less devices are dropped unless
    # they are the anchor device).
    er.async_get(hass).async_update_entity("light.bedroom", device_id=device.id)
    code = """
result = {
    "device_label_count": len(dr.async_entries_for_label(dr.async_get(hass), "fav")),
    "entity_by_id": er.async_get_entity(er.async_get(hass), "light", "test", "bedroom"),
    "entity_entries_count": len(er.async_entries(er.async_get(hass))),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["device_label_count"] == 1
    assert output["entity_by_id"] == "light.bedroom"
    assert output["entity_entries_count"] >= 2


async def test_label_registry_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    from homeassistant.helpers import label_registry as lr

    lr.async_get(hass).async_create("Favourites")
    code = """
by_name = label_registry.async_get_label_by_name("FAV OURITES")
result = {
    "count": len(label_registry.async_list_labels()),
    "label_id": by_name.label_id if by_name else None,
    "module_count": len(lr.async_get(hass).async_list_labels()),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["count"] >= 1
    assert output["label_id"] is not None
    assert output["module_count"] == output["count"]


async def test_category_registry_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    from homeassistant.helpers import category_registry as cr_core

    reg = cr_core.async_get(hass)
    reg.async_create(name="High Priority", scope="todo")
    code = """
cats = category_registry.async_list_categories(scope="todo")
result = {
    "count": len(cats),
    "module_count": len(cr.async_get(hass).async_list_categories(scope="todo")),
    "missing_scope": len(category_registry.async_list_categories(scope="nope")),
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["count"] >= 1
    assert output["module_count"] == output["count"]
    assert output["missing_scope"] == 0


async def test_repairs_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue

    async_create_issue(
        hass,
        domain="light",
        issue_id="warn_thing",
        is_fixable=False,
        is_persistent=False,
        severity=IssueSeverity.WARNING,
        translation_key="warn_thing",
    )
    code = """
result = {
    "total": len(repairs.async_issues()),
    "active": len(repairs.async_active_issues()),
    "by_severity": len(repairs.async_issues_by_severity("warning")),
    "one": repairs.async_get_issue("light", "warn_thing") is not None,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["total"] >= 1
    assert output["active"] >= 1
    assert output["by_severity"] >= 1
    assert output["one"] is True


async def test_persistent_notifications_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    from homeassistant.components.persistent_notification import async_create

    async_create(
        hass,
        "Disk almost full",
        title="Storage warning",
        notification_id="disk_full",
    )
    code = """
notification = persistent_notifications.async_get_notification("disk_full")
result = {
    "total": len(persistent_notifications.async_get_notifications()),
    "one": notification is not None,
    "title": notification.title if notification else None,
    "message": notification.message if notification else None,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["total"] >= 1
    assert output["one"] is True
    assert output["title"] == "Storage warning"
    assert output["message"] == "Disk almost full"


async def test_config_entries_read_end_to_end(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    code = """
mine = config_entries.async_get_entry("<ENTRY_ID>")
all_count = len(config_entries.async_entries())
domain_count = len(config_entries.async_entries("llm_sandbox"))
result = {
    "found": mine is not None,
    "domain": mine.domain if mine else None,
    "all_count": all_count,
    "domain_count": domain_count,
}
""".replace("<ENTRY_ID>", loaded_entry.entry_id)

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["found"] is True
    assert output["domain"] == "llm_sandbox"
    assert output["domain_count"] >= 1


async def test_service_action_executes_and_records_outcome(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify async_call executes and records a successful action outcome."""
    events: list[str] = []
    hass.bus.async_listen("call_service", lambda event: events.append(event.data.get("service", "")))

    code = """
ret = await hass.services.async_call(
    "light",
    "turn_on",
    {"brightness_pct": 80},
    target={"entity_id": "light.bedroom"},
)
result = ret
"""

    result = await _run_code(hass, loaded_entry, code)
    await hass.async_block_till_done()

    assert result["execution"]["status"] == "ok"
    # Return value matches HA: None (not return_response).
    assert result["output"] is None
    actions = result["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["domain"] == "light"
    assert action["service"] == "turn_on"
    assert action["service_data"]["brightness_pct"] == 80
    assert action["target"]["entity_id"] == ["light.bedroom"]
    assert action["status"] == "ok"
    assert action["response"] is None
    assert action["error"] is None
    assert events == ["turn_on"]


async def test_action_payload_is_json_safe(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify nested non-primitive service action data is serialized safely."""
    code = """
state = hass.states.get("light.bedroom")
await hass.services.async_call(
    "light",
    "turn_on",
    {
        7: {
            "levels": (1, 2),
            "labels": {"cozy", "night"},
            "state": state,
        },
    },
    target={"entity_id": ("light.bedroom",)},
)
result = "ok"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    json.dumps(result["actions"])
    action = result["actions"][0]
    nested = action["service_data"]["7"]
    assert nested["levels"] == [1, 2]
    assert set(nested["labels"]) == {"cozy", "night"}
    assert nested["state"]["entity_id"] == "light.bedroom"
    assert action["target"]["entity_id"] == ["light.bedroom"]


async def test_large_allocation_fails_with_monty_resource_limit(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify Monty memory limits bound runaway allocation before tool success."""
    result = await _run_code(hass, loaded_entry, "result = [0] * 10000000")

    assert result["execution"]["status"] == "code_error"
    assert result["execution"]["kind"] == "MemoryError"


async def test_timeout_returns_code_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify execution timeouts are runtime code errors, not setup errors."""

    async def _raise_timeout(_awaitable: Awaitable[Any], **_kwargs: object) -> Any:
        close = getattr(_awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    monkeypatch.setattr(executor.asyncio, "wait_for", _raise_timeout)

    result = await _run_code(hass, loaded_entry, "result = 1")

    assert result["execution"]["status"] == "code_error"
    assert result["execution"]["kind"] == "TimeoutError"
    assert result["output"] is None


async def test_service_not_found_records_error_action_and_continues(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify unknown service/domain records an errored action and continues."""
    code = """
await hass.services.async_call("nonexistent", "do_thing")
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "should not reach"
    assert result["actions"][0]["status"] == "error"
    assert result["actions"][0]["error"]["key"] == "service_not_found"


async def test_response_required_service_without_return_response_records_error_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify ONLY response services record an error when return_response is absent."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    code = """
await hass.services.async_call("test_response", "required")
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "should not reach"
    assert result["actions"][0]["status"] == "error"
    assert result["actions"][0]["error"]["key"] == "service_lacks_response_request"


async def test_no_response_service_with_return_response_records_error_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify NONE response services record an error for return_response=True."""
    code = """
await hass.services.async_call("light", "turn_on", blocking=True, return_response=True)
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "should not reach"
    assert result["actions"][0]["status"] == "error"
    assert result["actions"][0]["error"]["key"] == "service_response_not_supported"


async def test_return_response_without_blocking_records_error_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify return_response=True records an error when blocking is absent."""
    hass.services.async_register(
        "test_response",
        "optional",
        lambda call: {},
        supports_response=SupportsResponse.OPTIONAL,
    )
    code = """
await hass.services.async_call("test_response", "optional", return_response=True)
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "should not reach"
    assert result["actions"][0]["status"] == "error"
    assert result["actions"][0]["error"]["key"] == "service_response_requires_blocking"


async def test_response_required_service_with_return_response_records_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify return_response=True records an executed ONLY service action."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    code = """
result = await hass.services.async_call(
    "test_response",
    "required",
    blocking=True,
    return_response=True,
)
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    actions = result["actions"]
    assert len(actions) == 1
    assert actions[0]["domain"] == "test_response"
    assert actions[0]["service"] == "required"
    assert actions[0]["return_response"] is True
    assert actions[0]["status"] == "ok"
    assert actions[0]["response"] == result["output"]


async def test_positional_response_service_call_records_action(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify HA-style positional async_call arguments are accepted."""
    hass.services.async_register(
        "test_response",
        "required",
        lambda call: {},
        supports_response=SupportsResponse.ONLY,
    )
    events: list[str] = []
    hass.bus.async_listen("call_service", lambda event: events.append(event.data.get("service", "")))
    code = """
result = await hass.services.async_call(
    "test_response",
    "required",
    None,
    True,
    None,
    {"entity_id": "light.bedroom"},
    True,
)
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == {}
    actions = result["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["domain"] == "test_response"
    assert action["service"] == "required"
    assert action["service_data"] is None
    assert action["blocking"] is True
    assert action["target"]["entity_id"] == ["light.bedroom"]
    assert action["return_response"] is True
    assert action["status"] == "ok"
    assert action["response"] == result["output"]
    assert events == ["required"]


async def test_llm_context_device_id_accessible(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify llm_context is accessible inside Monty code."""
    code = """
result = {
    "platform": llm_context.platform,
    "language": llm_context.language,
    "device_id": llm_context.device_id,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["platform"] == "test"
    assert output["language"] == "en"


async def test_missing_await_normalized_on_async_call(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify the forgiveness layer adds missing await on async_call."""
    code = """
hass.services.async_call("light", "turn_off")
result = "ok"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert "awaited_async_calls" in result["execution"]["normalizations"]


async def test_forgiveness_layer_collapses_home_tour_to_single_call(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """R3: alias fields + reflection builtins resolve in one successful call.

    Reproduces the "tell me about my home" patterns that previously took five
    calls and three errors, now collapsing to one ok result.
    """
    from homeassistant.helpers import area_registry as ar
    from homeassistant.helpers import floor_registry as fr

    floor = fr.async_get(hass).async_create("Ground Floor")
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    ar.async_get(hass).async_update(bedroom.id, floor_id=floor.floor_id)

    code = """
floors = [(f.name, f.floor_id, f.id) for f in floor_registry.async_list_floors()]
areas = [(a.name, a.id, a.area_id) for a in area_registry.async_list_areas()]
first_floor = floor_registry.async_list_floors()[0]
result = {
    "floor_count": len(floors),
    "floors": floors,
    "areas": areas,
    "has_name": hasattr(first_floor, "name"),
    "has_bogus": hasattr(first_floor, "nope"),
    "floor_name_via_getattr": getattr(first_floor, "name", None),
    "type_name": type(floor_registry).__name__,
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["floor_count"] == 1
    # Alias: f.id equals canonical f.floor_id.
    assert output["floors"][0][1] == output["floors"][0][2]
    # Alias: a.area_id equals canonical a.id.
    assert output["areas"][0][1] == output["areas"][0][2]
    assert output["has_name"] is True
    assert output["has_bogus"] is False
    assert output["floor_name_via_getattr"] == "Ground Floor"
    assert output["type_name"] == "SafeFloorRegistry"
    normalizations = result["execution"]["normalizations"]
    assert "type_name_resolved" in normalizations


async def test_datetime_now_isoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """datetime.now() returns a frozen snapshot datetime in HA timezone."""
    code = "result = datetime.now().isoformat()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert isinstance(output, str)
    assert "T" in output


async def test_datetime_utcnow_isoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """datetime.utcnow() returns a frozen UTC snapshot datetime."""
    code = "result = datetime.utcnow().isoformat()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert "T" in result["output"]


async def test_date_today_isoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """date.today() returns the frozen snapshot date."""
    code = "result = date.today().isoformat()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert isinstance(output, str)
    assert len(output) == 10  # YYYY-MM-DD format


async def test_date_today_fields(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """date.today() exposes year, month, day, weekday fields."""
    code = """
d = date.today()
result = {
    "year": d.year,
    "month": d.month,
    "day": d.day,
    "weekday": d.weekday,
    "year_matches_snapshot": d.year == int(now[:4]),
}
"""
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["year_matches_snapshot"] is True
    assert isinstance(output["month"], int)
    assert isinstance(output["day"], int)
    assert isinstance(output["weekday"], int)


async def test_datetime_fromisoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """datetime.fromisoformat parses an ISO string and exposes fields."""
    code = "result = datetime.fromisoformat('2025-01-15T08:30:00+00:00').year"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert result["output"] == 2025


async def test_date_fromisoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """date.fromisoformat parses an ISO date string and exposes fields."""
    code = "result = date.fromisoformat('2025-03-20').month"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert result["output"] == 3


async def test_datetime_now_date_method(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """datetime.now().date() returns a SafeDate value."""
    code = "result = datetime.now().date().isoformat()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert len(result["output"]) == 10


async def test_return_date_object_directly(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Returning a SafeDate directly produces an ISO string."""
    code = "result = date.today()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert len(result["output"]) == 10


async def test_return_datetime_object_directly(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Returning a SafeDateTime directly produces an ISO string."""
    code = "result = datetime.now()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert "T" in result["output"]


async def test_from_datetime_import_normalized(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """from datetime import datetime is normalized to the sandbox facade."""
    code = "from datetime import datetime\nresult = datetime.now().isoformat()[:4] == now[:4]"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert result["output"] is True
    assert "datetime_imports_resolved" in result["execution"]["normalizations"]


async def test_import_datetime_as_dt_normalized(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """import datetime as dt is normalized with attribute rewriting."""
    code = "import datetime as dt\nresult = dt.datetime.now().year == int(now[:4])"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert result["output"] is True
    assert "datetime_imports_resolved" in result["execution"]["normalizations"]


async def test_from_datetime_import_date_as_d_normalized(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """from datetime import date as d is normalized to the sandbox facade."""
    code = "from datetime import date as d\nresult = d.today().isoformat()"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert len(result["output"]) == 10


async def test_parse_state_timestamp_with_fromisoformat(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """datetime.fromisoformat parses State timestamp strings."""
    code = """
s = hass.states.get('light.bedroom')
result = datetime.fromisoformat(s.last_changed).year == int(now[:4])
"""
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert result["output"] is True


async def test_now_global_unchanged(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """The now global remains an ISO string, not a facade object."""
    code = "result = now"
    result = await _run_code(hass, loaded_entry, code)
    assert result["execution"]["status"] == "ok"
    assert isinstance(result["output"], str)
    assert "T" in result["output"]
