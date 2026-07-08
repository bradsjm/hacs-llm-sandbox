"""Tests for Monty await-forgiveness normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.facades import (
    SafeAreaRegistry,
    SafeCategoryRegistry,
    SafeDate,
    SafeDateFacade,
    SafeDateTime,
    SafeDateTimeFacade,
    SafeDeviceRegistry,
    SafeEntityRegistry,
    SafeFloorRegistry,
    SafeHass,
    SafeLabelRegistry,
    SafeLLMContext,
    SafeServiceRegistry,
    SafeStateMachine,
)
from custom_components.llm_sandbox.llm_api.normalization.await_normalization import (
    AWAITED_ASYNC_CALLS,
    REWROTE_SYNC_SUBSCRIPT,
    STRIPPED_AWAIT_FROM_SYNC,
    normalize_awaits,
)

VIEW_CLASSES = [
    SafeHass,
    SafeStateMachine,
    SafeServiceRegistry,
    SafeEntityRegistry,
    SafeDeviceRegistry,
    SafeAreaRegistry,
    SafeFloorRegistry,
    SafeLabelRegistry,
    SafeCategoryRegistry,
    SafeDate,
    SafeDateTime,
    SafeDateFacade,
    SafeDateTimeFacade,
    SafeLLMContext,
]


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = hass.states.get('light.bedroom')",
            "result = hass.states.get('light.bedroom')",
            set(),
            id="sync-state-get-unchanged",
        ),
        pytest.param(
            "result = await hass.states.get('light.bedroom')",
            "result = hass.states.get('light.bedroom')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-get",
        ),
        pytest.param(
            "result = hass.services.async_call('light', 'turn_on')",
            "result = await hass.services.async_call('light', 'turn_on')",
            {AWAITED_ASYNC_CALLS},
            id="wrap-missing-await-on-async-call",
        ),
        pytest.param(
            "result = await hass.services.async_call('light', 'turn_on')",
            "result = await hass.services.async_call('light', 'turn_on')",
            set(),
            id="already-correct-await",
        ),
        pytest.param(
            "result = await hass.services.async_services_for_domain('light')",
            "result = hass.services.async_services_for_domain('light')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-domain-services",
        ),
        pytest.param(
            "result = await hass.services.supports_response('light', 'turn_on')",
            "result = hass.services.supports_response('light', 'turn_on')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-supports-response",
        ),
        pytest.param(
            "result = await er.async_get(hass)",
            "result = er.async_get(hass)",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-module-get",
        ),
        pytest.param(
            "result = await area_registry.async_get_area_by_name('Bedroom')",
            "result = area_registry.async_get_area_by_name('Bedroom')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-area-lookup",
        ),
        pytest.param(
            "result = await date.today()",
            "result = date.today()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-date-today",
        ),
        pytest.param(
            "result = await datetime.now()",
            "result = datetime.now()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-datetime-now",
        ),
        pytest.param(
            "result = await datetime.utcnow().isoformat()",
            "result = datetime.utcnow().isoformat()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-datetime-utcnow-chain",
        ),
        pytest.param(
            "state = await hass.states.get('light.bedroom')\nresult = hass.services.async_call('light', 'turn_on')",
            "state = hass.states.get('light.bedroom')\nresult = await hass.services.async_call('light', 'turn_on')",
            {AWAITED_ASYNC_CALLS, STRIPPED_AWAIT_FROM_SYNC},
            id="wrap-and-strip",
        ),
        pytest.param(
            "result = states['light.bedroom']",
            "result = states.get('light.bedroom')",
            {REWROTE_SYNC_SUBSCRIPT},
            id="rewrite-states-subscript",
        ),
        pytest.param(
            "result = await hass.states['light.bedroom']",
            "result = hass.states.get('light.bedroom')",
            {REWROTE_SYNC_SUBSCRIPT, STRIPPED_AWAIT_FROM_SYNC},
            id="rewrite-and-strip-hass-states-subscript",
        ),
        pytest.param(
            "result = len(states)",
            "result = len(states.async_entity_ids())",
            {REWROTE_SYNC_SUBSCRIPT},
            id="rewrite-state-machine-len",
        ),
        pytest.param(
            "result = x.get('foo')",
            "result = x.get('foo')",
            set(),
            id="local-variable-not-rooted",
        ),
        pytest.param(
            "result = hass.services.async_call(",
            "result = hass.services.async_call(",
            set(),
            id="syntax-error-fail-open",
        ),
    ],
)
def test_normalize_awaits_rewrites_only_rooted_facade_operations(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = normalize_awaits(code, VIEW_CLASSES)

    assert normalized == expected_code
    assert set(labels) == expected_labels
