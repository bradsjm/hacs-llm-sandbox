"""Behavior tests for live service invocation through the safe facade."""

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import cast

import pytest
import voluptuous as vol
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.errors import (
    CodeErrorPayload,
    HelperErrorPayload,
    HelperExecutionError,
)
from custom_components.llm_sandbox.llm_api.executor_support import (
    ExecutionState,
    helper_error_payload_for_state,
    validation_error,
)
from custom_components.llm_sandbox.llm_api.facade_views import (
    SafeHass,
    SafeServiceRegistry,
    build_facades,
)
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from custom_components.llm_sandbox.llm_api.runtime import (
    RuntimeContext,
    activate_runtime,
    clear_runtime,
)
from custom_components.llm_sandbox.runtime import SandboxSettings
from custom_components.llm_sandbox.snapshot.models import (
    DEFAULT_SCOPE,
    HomeSnapshot,
    SafeConfig,
    SafeContext,
    SafeState,
    SafeUnitSystem,
    ServiceSchemaBrief,
    SnapshotIndexes,
)
from custom_components.llm_sandbox.types import ActionRecord, ProposedAction
from homeassistant.core import Context, SupportsResponse

LIGHT_TURN_ON_BRIEF: ServiceSchemaBrief = {
    "fields": [
        {
            "name": "brightness_pct",
            "required": False,
            "type_hint": "integer",
            "description": None,
        }
    ],
    "dynamic": False,
}
LIGHT_GET_STATE_BRIEF: ServiceSchemaBrief = {"fields": [], "dynamic": False}
SWITCH_TURN_ON_BRIEF: ServiceSchemaBrief = {"fields": [], "dynamic": False}
TEST_OPTIONAL_BRIEF: ServiceSchemaBrief = {"fields": [], "dynamic": False}
TEST_REQUIRED_BRIEF: ServiceSchemaBrief = {"fields": [], "dynamic": False}


@dataclass(slots=True)
class RecordingInvoker:
    """Configurable service invoker that records validated action payloads."""

    responses: list[object] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)
    calls: list[ProposedAction] = field(default_factory=list)

    async def __call__(self, action: ProposedAction) -> object:
        """Record the action and then return or raise the configured outcome."""
        self.calls.append(_copy_action(action))
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return None


@dataclass(frozen=True, slots=True)
class ServiceHarness:
    """Facade runtime pieces needed to exercise one service invocation run."""

    snapshot: HomeSnapshot
    runtime: RuntimeContext
    services: SafeServiceRegistry
    invoker: RecordingInvoker


@pytest.fixture(autouse=True)
def _clear_active_runtime() -> Iterator[None]:
    """Keep context-local facade runtime isolated between tests."""
    clear_runtime()
    yield
    clear_runtime()


async def test_live_invocation_records_cleaned_resolved_ok_action() -> None:
    """Validated calls strip selectors, rewrite targets, and record one success."""
    harness = _service_harness()
    supplied_context = Context(user_id="llm-supplied-user")

    result = await harness.services.async_call(
        "light",
        "turn_on",
        {"brightness_pct": 80, "entity_id": "light.hidden"},
        blocking=False,
        context=supplied_context,
        target={"entity_id": "light.bedroom"},
    )

    assert result is None
    assert harness.invoker.calls == [
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"brightness_pct": 80},
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": False,
            "return_response": False,
        }
    ]
    assert "context" not in harness.invoker.calls[0]
    assert harness.runtime.state.actions == [
        {
            "service": "light.turn_on",
            "target": {"entity_id": ["light.bedroom"]},
            "status": "ok",
        }
    ]


async def test_return_response_records_service_response_on_success() -> None:
    """Response-capable blocking calls expose and record the live service response."""
    service_response = {"state": "on", "changed": True}
    harness = _service_harness(invoker=RecordingInvoker(responses=[service_response]))

    result = await harness.services.async_call(
        "light",
        "get_state",
        blocking=True,
        target={"entity_id": "light.bedroom"},
        return_response=True,
    )

    assert result == service_response
    assert harness.invoker.calls == [
        {
            "domain": "light",
            "service": "get_state",
            "service_data": None,
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": True,
            "return_response": True,
        }
    ]
    assert harness.runtime.state.actions == [
        {
            "service": "light.get_state",
            "target": {"entity_id": ["light.bedroom"]},
            "status": "ok",
            "response": service_response,
        }
    ]


@pytest.mark.parametrize(
    (
        "domain",
        "service",
        "blocking",
        "return_response",
        "service_response",
        "expected_result",
        "expected_status",
        "expected_error_key",
        "expected_invoker_calls",
    ),
    [
        pytest.param(
            "test_response",
            "required",
            False,
            False,
            {"required": True},
            {"required": True},
            "ok",
            None,
            [
                {
                    "domain": "test_response",
                    "service": "required",
                    "service_data": None,
                    "target": None,
                    "blocking": True,
                    "return_response": True,
                }
            ],
            id="only-accommodated",
        ),
        pytest.param(
            "test_response",
            "optional",
            False,
            False,
            {"optional": True},
            {"optional": True},
            "ok",
            None,
            [
                {
                    "domain": "test_response",
                    "service": "optional",
                    "service_data": None,
                    "target": None,
                    "blocking": True,
                    "return_response": True,
                }
            ],
            id="optional-accommodated",
        ),
        pytest.param(
            "light",
            "turn_on",
            True,
            True,
            None,
            None,
            "error",
            "service_response_not_supported",
            [],
            id="none-rejected",
        ),
    ],
)
async def test_response_mode_policy(
    domain: str,
    service: str,
    blocking: bool,
    return_response: bool,
    service_response: dict[str, object] | None,
    expected_result: object,
    expected_status: str,
    expected_error_key: str | None,
    expected_invoker_calls: list[ProposedAction],
) -> None:
    """Response-mode policy accommodates ONLY/OPTIONAL and blocks NONE before invocation."""
    responses = [] if service_response is None else [service_response]
    harness = _service_harness(invoker=RecordingInvoker(responses=responses))

    result = await _ok_call(harness, domain, service, blocking=blocking, return_response=return_response)

    assert result == expected_result
    assert harness.invoker.calls == expected_invoker_calls
    assert harness.runtime.state.actions[0]["status"] == expected_status
    if expected_error_key is None:
        assert harness.runtime.state.actions[0]["response"] == service_response
    else:
        assert cast(dict[str, object], harness.runtime.state.actions[0]["error"])["key"] == expected_error_key


async def test_service_not_found_records_blocked_action_and_returns_none() -> None:
    """Unknown services record a blocked action with domain-local schema hints."""
    harness = _service_harness()

    result = await _ok_call(harness, "light", "missing")

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["service_not_found"]
    action_error = cast(dict[str, object], harness.runtime.state.actions[0]["error"])
    assert isinstance(action_error["message"], str)
    assert action_error["message"]
    assert action_error["message"] != action_error["key"]
    assert {"light.get_state", "light.turn_on"} <= _guidance_candidate_ids(action_error["guidance"])
    assert harness.invoker.calls == []


async def test_service_validation_error_uses_translation_key_and_message() -> None:
    """Live validation failures keep HA's translation key and a specific message."""
    harness = _service_harness(
        invoker=RecordingInvoker(errors=[validation_error("invalid_light_target", {"entity_id": "light.bedroom"})])
    )

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert isinstance(payload["execution"]["message"], str)
    assert payload["execution"]["message"]
    assert payload["execution"]["message"] != "invalid_light_target"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["invalid_light_target"]


async def test_voluptuous_invalid_is_service_call_failed_action_error() -> None:
    """Schema failures without HA translation metadata use the generic call-failed key."""
    harness = _service_harness(invoker=RecordingInvoker(errors=[vol.Invalid("bad value")]))

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert isinstance(payload["execution"]["message"], str)
    assert payload["execution"]["message"]
    assert payload["execution"]["message"] != "service_call_failed"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["service_call_failed"]


async def test_expired_per_call_deadline_records_timeout_action_error() -> None:
    """A service call with no remaining runtime budget fails as a service timeout."""
    harness = _service_harness(deadline=time.monotonic() - 1)

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert isinstance(payload["execution"]["message"], str)
    assert payload["execution"]["message"]
    assert payload["execution"]["message"] != "service_call_timeout"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["service_call_timeout"]
    assert harness.invoker.calls == []


async def test_explicit_hidden_entity_target_resolves_to_unique_visible_entity() -> None:
    """A unique same-domain fuzzy target match resolves and executes."""
    harness = _service_harness()

    result = await _ok_call(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "bedroom"},
    )

    assert result is None
    assert harness.runtime.state.actions[0]["resolved_from"] == "bedroom"
    assert harness.invoker.calls == [
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": None,
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": False,
            "return_response": False,
        }
    ]
    assert harness.runtime.state.actions[0]["target"] == {"entity_id": ["light.bedroom"]}
    assert harness.runtime.state.actions[0]["status"] == "ok"


async def test_service_data_entity_selector_bypass_is_cleaned_and_resolved() -> None:
    """Entity selectors smuggled in service_data are cleaned before target resolution."""
    harness = _service_harness()

    result = await _ok_call(
        harness,
        "light",
        "turn_on",
        service_data={"brightness_pct": 25, "entity_id": "bedroom"},
    )

    assert result is None
    action = harness.runtime.state.actions[0]
    assert action["target"] == {"entity_id": ["light.bedroom"]}
    assert action["status"] == "ok"
    assert action["resolved_from"] == "bedroom"
    assert harness.invoker.calls == [
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"brightness_pct": 25},
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": False,
            "return_response": False,
        }
    ]


async def test_device_target_resolves_to_visible_entity_target_for_invocation() -> None:
    """Aggregate device targets are expanded to visible entity_id targets."""
    harness = _service_harness()

    result = await harness.services.async_call(
        "light",
        "turn_on",
        target={"device_id": "device-bedroom"},
    )

    assert result is None
    adjustments = harness.runtime.state.actions[0]["adjustments"]
    assert isinstance(adjustments, list)
    assert len(adjustments) == 1
    adjustment = adjustments[0]
    assert adjustment["key"] == "target_selector_expanded"
    assert adjustment["status"] == "applied"
    assert adjustment["retry_needed"] is False
    assert isinstance(adjustment["message"], str)
    assert adjustment["message"]
    assert adjustment["requested"] == {"device_id": "device-bedroom"}
    assert adjustment["applied"] == {"entity_id": ["light.bedroom"]}
    assert harness.invoker.calls == [
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": None,
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": False,
            "return_response": False,
        }
    ]
    assert harness.runtime.state.actions[0]["target"] == {"entity_id": ["light.bedroom"]}
    assert harness.runtime.state.actions[0]["status"] == "ok"


async def test_empty_entity_target_is_rejected_before_invocation() -> None:
    """Recognized target selectors that resolve to no entities are not invoked."""
    harness = _service_harness()

    result = await _ok_call(
        harness,
        "light",
        "turn_on",
        target={"entity_id": []},
    )

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["service_target_not_visible"]
    assert harness.invoker.calls == []


async def test_ambiguous_entity_target_blocks_with_candidates() -> None:
    """Ambiguous same-domain target matches record candidates and do not invoke live HA."""
    harness = _service_harness()

    result = await _ok_call(harness, "switch", "turn_on", target={"entity_id": "switch.outlet_kitchen"})

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["service_target_not_visible"]
    action_error = cast(dict[str, object], harness.runtime.state.actions[0]["error"])
    assert isinstance(action_error["message"], str)
    assert action_error["message"]
    assert {"switch.outlet", "switch.kitchen"} <= _guidance_candidate_ids(action_error["guidance"])
    assert harness.invoker.calls == []


async def test_helper_error_payload_keeps_prior_success_and_failed_action() -> None:
    """Partial action history includes prior successes plus the failed call."""
    harness = _service_harness()

    await harness.services.async_call(
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )
    harness.invoker.errors.append(RuntimeError("boom"))
    payload = await _helper_error_for(harness, "light", "turn_on", target={"entity_id": "light.bedroom"})

    assert payload["execution"]["status"] == "helper_error"
    assert isinstance(payload["execution"]["message"], str)
    assert payload["execution"]["message"]
    assert payload["execution"]["message"] != "service_call_failed"
    assert _action_statuses(payload) == ["ok", "error"]
    assert _action_keys(payload) == [None, "service_call_failed"]


async def test_actions_disabled_gate_records_blocked_action_and_returns_none() -> None:
    """A disabled action master switch records an errored action and returns None."""
    harness = _service_harness(actions_enabled=False)

    result = await _ok_call(harness, "light", "turn_on")

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["actions_disabled"]
    assert harness.invoker.calls == []


async def test_action_domain_allowlist_blocks_unlisted_domain() -> None:
    """Configured action domains are enforced before invocation."""
    harness = _service_harness(action_domains=frozenset({"light"}))

    result = await _ok_call(harness, "switch", "turn_on")

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["action_domain_not_allowed"]
    assert harness.invoker.calls == []


async def test_async_services_for_target_reports_per_entity_services() -> None:
    """Discovery returns, per resolved entity, the services whose target accepts it."""
    snapshot = replace(
        _snapshot(),
        services_target={
            "light": {
                "get_state": {"entity": [{"domain": ["light"]}]},
                "turn_on": {"entity": [{"domain": ["light"]}]},
            },
            "switch": {"turn_on": {"entity": [{"domain": ["switch"]}]}},
        },
    )
    harness = _service_harness(snapshot=snapshot)

    result = harness.services.async_services_for_target({"entity_id": "light.bedroom"})

    assert result == {
        "light.bedroom": {
            "light": {
                "get_state": {"supports_response": SupportsResponse.OPTIONAL.value, "fields": []},
                "turn_on": {"supports_response": SupportsResponse.NONE.value, "fields": ["brightness_pct"]},
            }
        }
    }


async def test_cross_domain_target_is_blocked_with_service_supported_fix() -> None:
    """A service whose target excludes the resolved entity's domain blocks with matching fixes."""
    snapshot = replace(
        _snapshot(),
        services_target={"light": {"turn_on": {"entity": [{"domain": ["light"]}]}}},
    )
    harness = _service_harness(snapshot=snapshot)

    result = await _ok_call(harness, "light", "turn_on", target={"entity_id": "switch.outlet"})

    assert result is None
    assert _action_statuses_via_state(harness) == ["error"]
    assert _action_keys_via_state(harness) == ["service_target_not_supported"]
    error = cast(dict[str, object], harness.runtime.state.actions[0]["error"])
    # The fix names the visible entity the service does target (light.bedroom), not the switch.
    assert "light.bedroom" in _guidance_candidate_ids(error["guidance"])
    assert harness.invoker.calls == []


async def test_unresolved_target_fix_list_ranks_service_supported_entities_first() -> None:
    """Target-not-visible fix lists order entities the service supports ahead of others."""
    snapshot = replace(
        _snapshot(),
        states={
            "cover.blind_supported": _state(
                "cover.blind_supported", "open", "Supported Blind", attributes={"supported_features": 4}
            ),
            "cover.blind_plain": _state("cover.blind_plain", "open", "Plain Blind"),
        },
        services={"cover": ("stop_cover",)},
        services_supports_response={"cover": {"stop_cover": SupportsResponse.NONE.value}},
        services_target={"cover": {"stop_cover": {"entity": [{"domain": ["cover"], "supported_features": [4]}]}}},
        services_schema={"cover": {"stop_cover": SWITCH_TURN_ON_BRIEF}},
    )
    harness = _service_harness(snapshot=snapshot)

    result = await _ok_call(harness, "cover", "stop_cover", target={"entity_id": "cover.nope"})

    assert result is None
    assert _action_keys_via_state(harness) == ["service_target_not_visible"]
    error = cast(dict[str, object], harness.runtime.state.actions[0]["error"])
    assert {"cover.blind_supported", "cover.blind_plain"} <= _guidance_candidate_ids(error["guidance"])


def _service_harness(
    *,
    actions_enabled: bool = True,
    action_domains: frozenset[str] = frozenset(),
    invoker: RecordingInvoker | None = None,
    deadline: float = math.inf,
    snapshot: HomeSnapshot | None = None,
) -> ServiceHarness:
    """Build a snapshot-backed services facade with an active runtime."""
    snapshot = snapshot or _snapshot()
    active_invoker = invoker or RecordingInvoker()
    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=20),
        settings=SandboxSettings(
            execution_timeout_seconds=10,
            helper_call_budget=20,
            scope=DEFAULT_SCOPE,
            actions_enabled=actions_enabled,
            action_domains=action_domains,
            prompt_profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
        ),
        invoke=active_invoker,
        deadline=deadline,
    )
    clear_runtime()
    activate_runtime(runtime, snapshot)
    facades = build_facades(snapshot)
    hass = cast(SafeHass, facades["hass"])
    return ServiceHarness(
        snapshot=snapshot,
        runtime=runtime,
        services=hass.services,
        invoker=active_invoker,
    )


def _snapshot() -> HomeSnapshot:
    """Return a realistic service/action snapshot with visibility indexes."""
    return HomeSnapshot(
        created_at="2026-06-29T00:00:00+00:00",
        states={
            "light.bedroom": _state("light.bedroom", "on", "Bedroom Light"),
            "switch.outlet": _state("switch.outlet", "off", "Outlet"),
            "switch.kitchen": _state("switch.kitchen", "off", "Kitchen"),
        },
        entities={},
        devices={},
        areas={},
        floors={},
        config=_config(),
        services={
            "light": ("get_state", "turn_on"),
            "switch": ("turn_on",),
            "test_response": ("optional", "required"),
        },
        services_supports_response={
            "light": {
                "get_state": SupportsResponse.OPTIONAL.value,
                "turn_on": SupportsResponse.NONE.value,
            },
            "switch": {"turn_on": SupportsResponse.NONE.value},
            "test_response": {
                "optional": SupportsResponse.OPTIONAL.value,
                "required": SupportsResponse.ONLY.value,
            },
        },
        indexes=SnapshotIndexes(
            entity_ids_by_device_id={"device-bedroom": ("light.bedroom",)},
            entity_ids_by_area_id={"area-bedroom": ("light.bedroom",)},
            device_ids_by_area_id={"area-bedroom": ("device-bedroom",)},
            entity_ids_by_config_entry_id={},
            entity_ids_by_label={"label-night": ("light.bedroom",)},
            device_ids_by_label={},
            area_ids_by_floor_id={"floor-main": ("area-bedroom",)},
        ),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema={
            "light": {
                "get_state": LIGHT_GET_STATE_BRIEF,
                "turn_on": LIGHT_TURN_ON_BRIEF,
            },
            "switch": {"turn_on": SWITCH_TURN_ON_BRIEF},
            "test_response": {
                "optional": TEST_OPTIONAL_BRIEF,
                "required": TEST_REQUIRED_BRIEF,
            },
        },
    )


def _config() -> SafeConfig:
    """Build a minimal frozen config record for snapshot helpers."""
    return SafeConfig(
        location_name="Home",
        latitude=0.0,
        longitude=0.0,
        elevation=0,
        time_zone="UTC",
        language="en",
        country=None,
        currency="USD",
        internal_url=None,
        external_url=None,
        units=SafeUnitSystem(
            temperature_unit="°C",
            length_unit="m",
            mass_unit="kg",
            pressure_unit="Pa",
            volume_unit="L",
            area_unit="m²",
            wind_speed_unit="m/s",
            accumulated_precipitation_unit="mm",
        ),
    )


def _state(
    entity_id: str,
    state: str,
    name: str,
    *,
    attributes: Mapping[str, object] | None = None,
) -> SafeState:
    """Build a minimal visible state record for target validation."""
    domain, object_id = entity_id.split(".", 1)
    merged: dict[str, object] = {"friendly_name": name}
    if attributes:
        merged.update(attributes)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=state,
        attributes=merged,
        last_changed="2026-06-29T00:00:00+00:00",
        last_changed_timestamp=1782691200.0,
        last_reported="2026-06-29T00:00:00+00:00",
        last_reported_timestamp=1782691200.0,
        last_updated="2026-06-29T00:00:00+00:00",
        last_updated_timestamp=1782691200.0,
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
        area_id=None,
        device_id=None,
        platform=domain,
        unique_id=entity_id,
    )


async def _helper_error_for(
    harness: ServiceHarness,
    domain: str,
    service: str,
    service_data: Mapping[str, object] | None = None,
    *,
    blocking: bool = False,
    target: Mapping[str, object] | None = None,
    return_response: bool = False,
) -> HelperErrorPayload:
    """Execute a facade call and return the executor-shaped helper-error payload."""
    with pytest.raises(HelperExecutionError) as err:
        await harness.services.async_call(
            domain,
            service,
            service_data,
            blocking=blocking,
            target=target,
            return_response=return_response,
        )
    return helper_error_payload_for_state(err.value, harness.runtime.state)


async def _ok_call(
    harness: ServiceHarness,
    domain: str,
    service: str,
    service_data: Mapping[str, object] | None = None,
    *,
    blocking: bool = False,
    target: Mapping[str, object] | None = None,
    return_response: bool = False,
) -> object:
    """Run a facade call expected to succeed and return its result."""
    return await harness.services.async_call(
        domain,
        service,
        service_data,
        blocking=blocking,
        target=target,
        return_response=return_response,
    )


def _copy_action(action: ProposedAction) -> ProposedAction:
    """Copy an action payload so later record mutation cannot affect assertions."""
    return {
        "domain": action["domain"],
        "service": action["service"],
        "service_data": _copy_mapping(action["service_data"]),
        "target": _copy_mapping(action["target"]),
        "blocking": action["blocking"],
        "return_response": action["return_response"],
    }


def _copy_mapping(value: object) -> object:
    """Copy shallow mappings used in action payloads."""
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _actions(payload: HelperErrorPayload) -> list[ActionRecord]:
    """Return action records from a helper-error payload."""
    actions = payload.get("actions")
    assert actions is not None
    return actions


def _first_action(payload: HelperErrorPayload) -> ActionRecord:
    """Return the first recorded action from a helper-error payload."""
    return _actions(payload)[0]


def _action_statuses(payload: HelperErrorPayload) -> list[object]:
    """Return action statuses from a helper-error payload."""
    return [action["status"] for action in _actions(payload)]


def _action_keys(payload: HelperErrorPayload) -> list[object]:
    """Return action error keys from a helper-error payload."""
    return [_action_key(action) for action in _actions(payload)]


def _action_statuses_via_state(harness: ServiceHarness) -> list[str]:
    """Return action statuses recorded on the active runtime state."""
    return [cast(str, action["status"]) for action in harness.runtime.state.actions]


def _action_keys_via_state(harness: ServiceHarness) -> list[str]:
    """Return action error keys recorded on the active runtime state."""
    return [
        cast(str, cast(dict[str, object], action["error"])["key"])
        for action in harness.runtime.state.actions
        if action.get("error")
    ]


def _guidance_candidate_ids(guidance: object) -> set[str]:
    """Return candidate ids from a serialized action-error guidance payload."""
    assert isinstance(guidance, Mapping)
    candidates = guidance["candidates"]
    assert isinstance(candidates, list)
    return {str(candidate["id"]) for candidate in candidates if isinstance(candidate, Mapping)}


def _code_actions(payload: CodeErrorPayload) -> list[ActionRecord]:
    """Return action records from a code-error payload."""
    actions = payload.get("actions")
    assert actions is not None
    return actions


def _code_action_statuses(payload: CodeErrorPayload) -> list[object]:
    """Return action statuses from a code-error payload."""
    return [action["status"] for action in _code_actions(payload)]


def _code_action_keys(payload: CodeErrorPayload) -> list[object]:
    """Return action error keys from a code-error payload."""
    return [_action_key(action) for action in _code_actions(payload)]


def _single_adjustment(action: ActionRecord) -> dict[str, object]:
    """Return the only adjustment on an action record."""
    adjustments = action.get("adjustments")
    assert isinstance(adjustments, list)
    assert len(adjustments) == 1
    adjustment = adjustments[0]
    assert isinstance(adjustment, dict)
    return cast(dict[str, object], adjustment)


def _adjustment_keys(action: ActionRecord) -> list[str]:
    """Return adjustment keys from one action record."""
    adjustments = action.get("adjustments")
    assert isinstance(adjustments, list)
    return [str(adjustment["key"]) for adjustment in adjustments if isinstance(adjustment, dict)]


def _action_key(action: ActionRecord) -> object:
    """Return the stable helper key for one action error, if present."""
    error = action.get("error")
    if error is None:
        return None
    return cast(Mapping[str, object], error)["key"]
