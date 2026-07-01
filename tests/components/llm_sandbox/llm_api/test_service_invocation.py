"""Behavior tests for live service invocation through the safe facade."""

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
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
    code_error_payload_for_state,
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
            "domain": "light",
            "service": "turn_on",
            "service_data": {"brightness_pct": 80},
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": False,
            "return_response": False,
            "status": "ok",
            "response": None,
            "error": None,
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
            "domain": "light",
            "service": "get_state",
            "service_data": None,
            "target": {"entity_id": ["light.bedroom"]},
            "blocking": True,
            "return_response": True,
            "status": "ok",
            "response": service_response,
            "error": None,
        }
    ]


async def test_service_not_found_helper_error_includes_domain_service_hints() -> None:
    """Unknown services surface a stable error code and domain-local schema hints."""
    harness = _service_harness()

    payload = await _helper_error_for(harness, "light", "missing")

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_not_found"
    assert _service_hints(payload) == {
        "available_services": {
            "get_state": LIGHT_GET_STATE_BRIEF,
            "turn_on": LIGHT_TURN_ON_BRIEF,
        }
    }
    assert _action_keys(payload) == ["service_not_found"]
    assert _action_statuses(payload) == ["error"]


async def test_service_validation_error_uses_translation_key_and_single_service_hint() -> None:
    """Live validation failures keep HA's translation key and requested-service brief."""
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
    assert payload["execution"]["code"] == "invalid_light_target"
    assert _service_hints(payload) == {"available_services": {"turn_on": LIGHT_TURN_ON_BRIEF}}
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
    assert payload["execution"]["code"] == "service_call_failed"
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
    assert payload["execution"]["code"] == "service_call_timeout"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["service_call_timeout"]
    assert harness.invoker.calls == []


async def test_explicit_hidden_entity_target_is_rejected_before_invocation() -> None:
    """Explicit entity targets must exist in the visible snapshot state set."""
    harness = _service_harness()

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "light.hidden"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_target_not_visible"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["service_target_not_visible"]
    assert harness.invoker.calls == []


async def test_service_data_entity_selector_bypass_is_cleaned_and_rejected() -> None:
    """Entity selectors smuggled in service_data still pass through visibility checks."""
    harness = _service_harness()

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        service_data={"brightness_pct": 25, "entity_id": "light.hidden"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_target_not_visible"
    action = _first_action(payload)
    assert action["service_data"] == {"brightness_pct": 25}
    assert action["target"] == {"entity_id": "light.hidden"}
    assert action["status"] == "error"
    assert _action_keys(payload) == ["service_target_not_visible"]
    assert harness.invoker.calls == []


async def test_device_target_resolves_to_visible_entity_target_for_invocation() -> None:
    """Aggregate device targets are expanded to visible entity_id targets."""
    harness = _service_harness()

    result = await harness.services.async_call(
        "light",
        "turn_on",
        target={"device_id": "device-bedroom"},
    )

    assert result is None
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

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": []},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_target_not_visible"
    assert _action_statuses(payload) == ["error"]
    assert harness.invoker.calls == []


async def test_helper_error_payload_keeps_prior_success_and_failed_action() -> None:
    """Partial action history includes prior successes plus the failed call."""
    harness = _service_harness()

    await harness.services.async_call(
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )
    payload = await _helper_error_for(harness, "light", "missing")

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_not_found"
    assert _action_statuses(payload) == ["ok", "error"]
    assert _action_keys(payload) == [None, "service_not_found"]


async def test_code_error_payload_keeps_prior_successful_action() -> None:
    """A later code failure payload still exposes already-executed actions."""
    harness = _service_harness()

    await harness.services.async_call(
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )
    payload = code_error_payload_for_state(
        kind="ZeroDivisionError",
        message="division by zero",
        state=harness.runtime.state,
    )

    assert payload["execution"]["status"] == "code_error"
    assert _code_action_statuses(payload) == ["ok"]
    assert _code_action_keys(payload) == [None]


async def test_generic_service_exception_is_service_call_failed_helper_error() -> None:
    """Plain service exceptions are classified as failed service actions."""
    harness = _service_harness(invoker=RecordingInvoker(errors=[RuntimeError("boom")]))

    payload = await _helper_error_for(
        harness,
        "light",
        "turn_on",
        target={"entity_id": "light.bedroom"},
    )

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "service_call_failed"
    assert _service_hints(payload) == {"available_services": {"turn_on": LIGHT_TURN_ON_BRIEF}}
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["service_call_failed"]


async def test_actions_disabled_gate_records_helper_error_action() -> None:
    """The action master switch blocks live service calls before invocation."""
    harness = _service_harness(actions_enabled=False)

    payload = await _helper_error_for(harness, "light", "turn_on")

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "actions_disabled"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["actions_disabled"]
    assert harness.invoker.calls == []


async def test_action_domain_allowlist_blocks_unlisted_domain() -> None:
    """Configured action domains are enforced before invocation."""
    harness = _service_harness(action_domains=frozenset({"light"}))

    payload = await _helper_error_for(harness, "switch", "turn_on")

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == "action_domain_not_allowed"
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == ["action_domain_not_allowed"]
    assert harness.invoker.calls == []


@pytest.mark.parametrize(
    ("domain", "service", "blocking", "return_response", "expected_key"),
    [
        pytest.param(
            "light",
            "get_state",
            False,
            True,
            "service_response_requires_blocking",
            id="return-response-requires-blocking",
        ),
        pytest.param(
            "light",
            "turn_on",
            True,
            True,
            "service_response_not_supported",
            id="none-service-rejects-return-response",
        ),
        pytest.param(
            "test_response",
            "required",
            False,
            False,
            "service_lacks_response_request",
            id="only-service-requires-return-response",
        ),
    ],
)
async def test_response_flag_rules_record_helper_error_action(
    domain: str,
    service: str,
    blocking: bool,
    return_response: bool,
    expected_key: str,
) -> None:
    """Response-mode contracts are enforced before live invocation."""
    harness = _service_harness()

    payload = await _helper_error_for(
        harness,
        domain,
        service,
        blocking=blocking,
        return_response=return_response,
    )

    assert payload["execution"]["status"] == "helper_error"
    assert payload["execution"]["code"] == expected_key
    assert _action_statuses(payload) == ["error"]
    assert _action_keys(payload) == [expected_key]
    assert harness.invoker.calls == []


def _service_harness(
    *,
    actions_enabled: bool = True,
    action_domains: frozenset[str] = frozenset(),
    invoker: RecordingInvoker | None = None,
    deadline: float = math.inf,
) -> ServiceHarness:
    """Build a snapshot-backed services facade with an active runtime."""
    snapshot = _snapshot()
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


def _state(entity_id: str, state: str, name: str) -> SafeState:
    """Build a minimal visible state record for target validation."""
    domain, object_id = entity_id.split(".", 1)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=state,
        attributes={"friendly_name": name},
        last_changed="2026-06-29T00:00:00+00:00",
        last_reported="2026-06-29T00:00:00+00:00",
        last_updated="2026-06-29T00:00:00+00:00",
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
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


def _service_hints(payload: HelperErrorPayload) -> Mapping[str, object] | None:
    """Return helper service hints as a concrete mapping for assertions."""
    return cast(Mapping[str, object] | None, payload.get("service_hints"))


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


def _action_key(action: ActionRecord) -> object:
    """Return the stable helper key for one action error, if present."""
    error = action["error"]
    if error is None:
        return None
    return cast(Mapping[str, object], error)["key"]
