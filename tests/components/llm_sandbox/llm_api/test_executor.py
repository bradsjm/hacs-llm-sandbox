"""End-to-end tests for the Monty executor with HA-native facades."""

import json
from collections.abc import Awaitable
from typing import Any

import pytest
from custom_components.llm_sandbox.llm_api import executor
from custom_components.llm_sandbox.llm_api.api import _execute
from homeassistant.core import Context, HomeAssistant, SupportsResponse
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
}
"""

    result = await _run_code(hass, loaded_entry, code)

    assert result["execution"]["status"] == "ok"
    output = result["output"]
    assert output["has_turn_on"] is True
    assert output["has_missing"] is False
    assert output["domain_count"] >= 1
    assert output["services"]["light"]["turn_on"]["supports_response"] == "none"
    assert output["services"]["test_response"]["required"]["supports_response"] == "only"


async def test_proposed_action_recorded_without_execution(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify async_call records a proposed action and does not mutate state.

    This is the core Option A assertion: the LLM code proposes a service call,
    it shows up in proposed_actions, the return matches HA's shape, and no
    real service event fires.
    """
    call_log: list[str] = []

    async def _mock_async_call(domain, service, *args, **kwargs):
        call_log.append(f"{domain}.{service}")

    # Track whether the real service bus fires by listening for call_service events.
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

    assert result["execution"]["status"] == "ok"
    # Return value matches HA: None (not return_response).
    assert result["output"] is None
    # One proposed action recorded.
    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    action = proposed[0]
    assert action["domain"] == "light"
    assert action["service"] == "turn_on"
    assert action["service_data"]["brightness_pct"] == 80
    assert action["target"]["entity_id"] == "light.bedroom"
    # No real service call fired.
    assert call_log == []
    assert events == []


async def test_proposed_action_payload_is_json_safe(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
) -> None:
    """Verify nested non-primitive proposed service data is serialized safely."""
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
    json.dumps(result["proposed_actions"])
    action = result["proposed_actions"][0]
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


async def test_proposed_action_service_not_found_raises_helper_error(
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
    """Verify return_response=True records an ONLY service proposal."""
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
    assert result["output"] == {}
    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    assert proposed[0]["domain"] == "test_response"
    assert proposed[0]["service"] == "required"
    assert proposed[0]["return_response"] is True


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
