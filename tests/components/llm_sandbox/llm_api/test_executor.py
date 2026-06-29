"""End-to-end tests for the Monty executor with HA-native facades."""

import json
from collections.abc import Awaitable
from typing import Any

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.llm_api import executor
from custom_components.llm_sandbox.llm_api.api import _execute
from homeassistant.core import Context, HomeAssistant, SupportsResponse
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
result = "No match"
if bedroom is not None:
    for entry in er.async_entries_for_area(er.async_get(hass), bedroom.id):
        if entry.entity_id.split(".")[0] != "light":
            continue
        st = hass.states.get(entry.entity_id)
        if st is not None and st.state == "on":
            result = "Yes"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    assert result["output"] == "Yes"


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


async def test_service_not_found_raises_helper_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify unknown service/domain raises a structured helper error."""
    code = """
await hass.services.async_call("nonexistent", "do_thing")
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "helper_error"
    assert result["output"] is None
    assert result["execution"]["code"] == "service_not_found"
    assert result["actions"][0]["status"] == "error"
    assert result["actions"][0]["error"]["key"] == "service_not_found"


async def test_response_required_service_without_return_response_raises_helper_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify ONLY response services require return_response=True."""
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

    assert result["execution"]["status"] == "helper_error"
    assert result["output"] is None
    assert result["execution"]["code"] == "service_lacks_response_request"


async def test_no_response_service_with_return_response_raises_helper_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify NONE response services reject return_response=True."""
    code = """
await hass.services.async_call("light", "turn_on", blocking=True, return_response=True)
result = "should not reach"
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "helper_error"
    assert result["output"] is None
    assert result["execution"]["code"] == "service_response_not_supported"


async def test_return_response_without_blocking_raises_helper_error(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify return_response=True requires blocking=True."""
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

    assert result["execution"]["status"] == "helper_error"
    assert result["output"] is None
    assert result["execution"]["code"] == "service_response_requires_blocking"


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
result = {
    "floor_count": len(floors),
    "floors": floors,
    "areas": areas,
    "has_list": hasattr(floor_registry, "async_list_floors"),
    "has_bogus": hasattr(floor_registry, "nope"),
    "area_count_via_getattr": len(getattr(area_registry, "async_list_areas")()),
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
    assert output["has_list"] is True
    assert output["has_bogus"] is False
    assert output["area_count_via_getattr"] == 1
    assert output["type_name"] == "SafeFloorRegistry"
    normalizations = result["execution"]["normalizations"]
    assert "hasattr_resolved" in normalizations
    assert "getattr_resolved" in normalizations
    assert "type_name_resolved" in normalizations
