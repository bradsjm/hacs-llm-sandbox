"""Focused behavioral tests for the end-state overlay reducer."""

from collections.abc import Mapping

from llm_sandbox_evals.schema import DesiredState
from llm_sandbox_evals.scoring.end_state import (
    EndStateResult,
    OverlayStateSeed,
    assess_end_state,
)
import pytest


def _seed(entity_id: str, state: str, domain: str | None = None) -> OverlayStateSeed:
    dom = domain or entity_id.split(".", 1)[0]
    return OverlayStateSeed(entity_id, dom, state)


def _call(
    domain: str,
    service: str,
    entity_id: str | list[str],
    *,
    service_data: Mapping[str, object] | None = None,
    status: str = "ok",
) -> dict[str, object]:
    target_key = "entity_id"
    return {
        "domain": domain,
        "service": service,
        "target": {target_key: entity_id},
        "service_data": dict(service_data) if service_data else {},
        "status": status,
    }


# ---------------------------------------------------------------------------
# No predicates
# ---------------------------------------------------------------------------


def test_no_predicates_returns_not_authored() -> None:
    result = assess_end_state([], (), ())
    assert result == EndStateResult("not_authored", False, False)


# ---------------------------------------------------------------------------
# Initially satisfied / no invocation
# ---------------------------------------------------------------------------


def test_initially_satisfied_with_no_calls_passes() -> None:
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    result = assess_end_state(desired, seeds, ())
    assert result.status == "satisfied"
    assert result.passed is True
    assert result.comparisons[0].actual_state == "on"


# ---------------------------------------------------------------------------
# Light and switch turn_on / turn_off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "service", "initial", "desired_state"),
    [
        pytest.param("light", "turn_on", "off", "on", id="light-turn-on"),
        pytest.param("light", "turn_off", "on", "off", id="light-turn-off"),
        pytest.param("switch", "turn_on", "off", "on", id="switch-turn-on"),
        pytest.param("switch", "turn_off", "on", "off", id="switch-turn-off"),
    ],
)
def test_direct_transition_satisfies_predicate(domain: str, service: str, initial: str, desired_state: str) -> None:
    entity_id = f"{domain}.device"
    desired = (DesiredState(entity_id, desired_state),)
    seeds = (_seed(entity_id, initial),)
    calls = (_call(domain, service, entity_id),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == desired_state


# ---------------------------------------------------------------------------
# Toggle: single and ordered repeated
# ---------------------------------------------------------------------------


def test_single_toggle_flips_state() -> None:
    desired = (DesiredState("switch.outlet", "on"),)
    seeds = (_seed("switch.outlet", "off"),)
    calls = (_call("switch", "toggle", "switch.outlet"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


def test_ordered_repeated_toggle_final_call_wins() -> None:
    desired = (DesiredState("switch.outlet", "off"),)
    seeds = (_seed("switch.outlet", "off"),)
    calls = (
        _call("switch", "toggle", "switch.outlet"),
        _call("switch", "toggle", "switch.outlet"),
    )
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == "off"


def test_ordered_calls_where_final_call_wins() -> None:
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = (
        _call("light", "turn_on", "light.bedroom"),
        _call("light", "turn_off", "light.bedroom"),
    )
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


# ---------------------------------------------------------------------------
# Multi-target direct calls
# ---------------------------------------------------------------------------


def test_multi_target_call_satisfies_all_predicates() -> None:
    desired = (DesiredState("light.a", "on"), DesiredState("light.b", "on"))
    seeds = (_seed("light.a", "off"), _seed("light.b", "off"))
    calls = (_call("light", "turn_on", ["light.a", "light.b"]),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert all(c.matched for c in result.comparisons)


# ---------------------------------------------------------------------------
# Unsupported / indirect / rejected / duplicate-target calls: no overlay effect
# ---------------------------------------------------------------------------


def test_unsupported_service_has_no_overlay_effect() -> None:
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = (_call("light", "set_brightness", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


def test_wrong_domain_call_has_no_overlay_effect() -> None:
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    # A switch service targeting a light entity must not flip the light.
    calls = (_call("switch", "turn_on", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


def test_errored_call_has_no_overlay_effect() -> None:
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    # Service data is ignored by the reducer; only the transition matters.
    calls = (_call("light", "turn_on", "light.bedroom", service_data={"brightness": 128}),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "unsatisfied"


def test_indirect_area_selector_has_no_overlay_effect() -> None:
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = ({"domain": "light", "service": "turn_on", "target": {"area_id": "bedroom"}},)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "unsatisfied"


# ---------------------------------------------------------------------------
# Unevaluable: missing seed, non-binary seed
# ---------------------------------------------------------------------------


def test_missing_seed_is_unevaluable() -> None:
    desired = (DesiredState("light.missing", "on"),)
    result = assess_end_state(desired, (), ())
    assert result.status == "unevaluable"
    assert result.evaluable is False


def test_non_binary_seed_is_unevaluable() -> None:
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "playing"),)
    result = assess_end_state(desired, seeds, ())
    assert result.status == "unevaluable"


def test_one_missing_seed_makes_entire_set_unevaluable() -> None:
    desired = (DesiredState("light.bedroom", "on"), DesiredState("light.missing", "on"))
    seeds = (_seed("light.bedroom", "off"),)
    result = assess_end_state(desired, seeds, ())
    assert result.status == "unevaluable"


# ---------------------------------------------------------------------------
# Unsatisfied: action ran but final state does not match
# ---------------------------------------------------------------------------


def test_action_ran_but_final_state_mismatch_is_unsatisfied() -> None:
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = (_call("light", "turn_on", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "unsatisfied"
    assert result.passed is False
    assert result.comparisons[0].actual_state == "on"
