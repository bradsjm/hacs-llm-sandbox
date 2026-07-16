"""Focused behavioral tests for the end-state overlay reducer."""

from collections.abc import Mapping

from llm_sandbox_evals.schema import DesiredEntity
from llm_sandbox_evals.scoring.end_state import (
    EndStateResult,
    OverlayStateSeed,
    assess_end_state,
)
import pytest


def _seed(
    entity_id: str,
    state: str,
    domain: str | None = None,
    attributes: dict[str, object] | None = None,
) -> OverlayStateSeed:
    dom = domain or entity_id.split(".", 1)[0]
    return OverlayStateSeed(entity_id, dom, state, attributes or {})


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
    desired = (DesiredEntity("light.bedroom", "on"),)
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
    desired = (DesiredEntity(entity_id, desired_state),)
    seeds = (_seed(entity_id, initial),)
    calls = (_call(domain, service, entity_id),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == desired_state


@pytest.mark.parametrize(
    ("service_data", "desired_attributes"),
    [
        pytest.param({"brightness_pct": 50}, {"brightness": 128}, id="brightness-percent"),
        pytest.param({"brightness": 64}, {"brightness": 64}, id="brightness-value"),
        pytest.param(
            {"color_temp_kelvin": 2700},
            {"color_temp_kelvin": 2700},
            id="color-temperature",
        ),
    ],
)
def test_light_turn_on_applies_supported_attribute_effects(
    service_data: dict[str, object], desired_attributes: dict[str, object]
) -> None:
    desired = (DesiredEntity("light.bedroom", attributes=desired_attributes),)
    seeds = (_seed("light.bedroom", "off", attributes=dict.fromkeys(desired_attributes, 180)),)
    calls = (_call("light", "turn_on", "light.bedroom", service_data=service_data),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_attributes == desired_attributes


def test_mixed_state_and_attribute_predicate_compares_both_authored_fields() -> None:
    desired = (DesiredEntity("light.bedroom", "on", {"brightness": 128}),)
    seeds = (_seed("light.bedroom", "off", attributes={"brightness": 180}),)
    calls = (
        _call(
            "light",
            "turn_on",
            "light.bedroom",
            service_data={"brightness_pct": 50, "unrelated": "ignored"},
        ),
    )

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == "on"
    assert result.comparisons[0].actual_attributes == {"brightness": 128}


def test_turn_off_preserves_authored_attributes() -> None:
    desired = (DesiredEntity("light.bedroom", "off", {"brightness": 180}),)
    seeds = (_seed("light.bedroom", "on", attributes={"brightness": 180}),)
    calls = (_call("light", "turn_off", "light.bedroom", service_data={"brightness": 64}),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_attributes == {"brightness": 180}


def test_climate_set_temperature_changes_only_target_temperature() -> None:
    desired = (DesiredEntity("climate.workshop", attributes={"temperature": 22}),)
    seeds = (_seed("climate.workshop", "heat", attributes={"temperature": 20}),)
    calls = (_call("climate", "set_temperature", "climate.workshop", service_data={"temperature": 22}),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == "heat"
    assert result.comparisons[0].actual_attributes == {"temperature": 22}


@pytest.mark.parametrize(
    ("service", "initial_state", "desired_state", "initial_position", "desired_position"),
    [
        pytest.param("open_cover", "closed", "open", 0, 100, id="open"),
        pytest.param("close_cover", "open", "closed", 100, 0, id="close"),
    ],
)
def test_cover_open_close_updates_boundary_state_and_position(
    service: str,
    initial_state: str,
    desired_state: str,
    initial_position: int,
    desired_position: int,
) -> None:
    desired = (DesiredEntity("cover.blinds", desired_state, {"current_position": desired_position}),)
    seeds = (_seed("cover.blinds", initial_state, attributes={"current_position": initial_position}),)
    calls = (_call("cover", service, "cover.blinds"),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == desired_state
    assert result.comparisons[0].actual_attributes == {"current_position": desired_position}


@pytest.mark.parametrize(
    ("position", "desired_state"),
    [
        pytest.param(0, "closed", id="closed-boundary"),
        pytest.param(50, "open", id="partially-open"),
    ],
)
def test_cover_position_updates_state_and_position(position: int, desired_state: str) -> None:
    desired = (DesiredEntity("cover.blinds", desired_state, {"current_position": position}),)
    seeds = (_seed("cover.blinds", "open", attributes={"current_position": 100}),)
    calls = (_call("cover", "set_cover_position", "cover.blinds", service_data={"position": position}),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == desired_state
    assert result.comparisons[0].actual_attributes == {"current_position": position}


@pytest.mark.parametrize(
    ("entity_id", "state", "attributes", "domain", "service", "service_data"),
    [
        pytest.param(
            "climate.workshop",
            "heat",
            {"temperature": 20},
            "climate",
            "set_temperature",
            {"temperature": True},
            id="boolean-temperature",
        ),
        pytest.param(
            "cover.blinds",
            "closed",
            {"current_position": 0},
            "cover",
            "set_cover_position",
            {"position": 101},
            id="out-of-range-position",
        ),
        pytest.param(
            "cover.blinds",
            "closed",
            {"current_position": 0},
            "cover",
            "set_cover_position",
            {"position": 50.0},
            id="non-integer-position",
        ),
    ],
)
def test_invalid_required_numeric_data_has_no_effect(
    entity_id: str,
    state: str,
    attributes: dict[str, object],
    domain: str,
    service: str,
    service_data: dict[str, object],
) -> None:
    desired = (DesiredEntity(entity_id, attributes=attributes),)
    seeds = (_seed(entity_id, state, attributes=attributes),)
    calls = (_call(domain, service, entity_id, service_data=service_data),)

    result = assess_end_state(desired, seeds, calls)

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == state
    assert result.comparisons[0].actual_attributes == attributes


# ---------------------------------------------------------------------------
# Toggle: single and ordered repeated
# ---------------------------------------------------------------------------


def test_single_toggle_flips_state() -> None:
    desired = (DesiredEntity("switch.outlet", "on"),)
    seeds = (_seed("switch.outlet", "off"),)
    calls = (_call("switch", "toggle", "switch.outlet"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


def test_ordered_repeated_toggle_final_call_wins() -> None:
    desired = (DesiredEntity("switch.outlet", "off"),)
    seeds = (_seed("switch.outlet", "off"),)
    calls = (
        _call("switch", "toggle", "switch.outlet"),
        _call("switch", "toggle", "switch.outlet"),
    )
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == "off"


def test_ordered_calls_where_final_call_wins() -> None:
    desired = (DesiredEntity("light.bedroom", "off"),)
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
    desired = (DesiredEntity("light.a", "on"), DesiredEntity("light.b", "on"))
    seeds = (_seed("light.a", "off"), _seed("light.b", "off"))
    calls = (_call("light", "turn_on", ["light.a", "light.b"]),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"
    assert all(c.matched for c in result.comparisons)


# ---------------------------------------------------------------------------
# Unsupported / indirect / rejected / duplicate-target calls: no overlay effect
# ---------------------------------------------------------------------------


def test_unsupported_service_has_no_overlay_effect() -> None:
    desired = (DesiredEntity("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = (_call("light", "set_brightness", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


def test_wrong_domain_call_has_no_overlay_effect() -> None:
    desired = (DesiredEntity("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    # A switch service targeting a light entity must not flip the light.
    calls = (_call("switch", "turn_on", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "satisfied"


@pytest.mark.parametrize(
    ("desired", "seed", "call"),
    [
        pytest.param(
            DesiredEntity("light.bedroom", "off"),
            _seed("light.bedroom", "off"),
            _call("light", "turn_on", "light.bedroom", status="error"),
            id="state-transition",
        ),
        pytest.param(
            DesiredEntity("climate.workshop", attributes={"temperature": 20}),
            _seed("climate.workshop", "heat", attributes={"temperature": 20}),
            _call(
                "climate",
                "set_temperature",
                "climate.workshop",
                service_data={"temperature": 22},
                status="error",
            ),
            id="attribute-effect",
        ),
        pytest.param(
            DesiredEntity("cover.blinds", "closed", {"current_position": 0}),
            _seed("cover.blinds", "closed", attributes={"current_position": 0}),
            _call("cover", "open_cover", "cover.blinds", status="error"),
            id="state-and-attribute-effect",
        ),
    ],
)
def test_errored_call_has_no_overlay_effect(
    desired: DesiredEntity,
    seed: OverlayStateSeed,
    call: dict[str, object],
) -> None:
    result = assess_end_state((desired,), (seed,), (call,))

    assert result.status == "satisfied"
    assert result.comparisons[0].actual_state == seed.state
    assert result.comparisons[0].actual_attributes == seed.attributes


def test_indirect_area_selector_has_no_overlay_effect() -> None:
    desired = (DesiredEntity("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = ({"domain": "light", "service": "turn_on", "target": {"area_id": "bedroom"}},)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "unsatisfied"


# ---------------------------------------------------------------------------
# Unevaluable: missing seed, non-binary seed
# ---------------------------------------------------------------------------


def test_missing_seed_is_unevaluable() -> None:
    desired = (DesiredEntity("light.missing", "on"),)
    result = assess_end_state(desired, (), ())
    assert result.status == "unevaluable"
    assert result.evaluable is False


def test_non_binary_seed_is_unevaluable() -> None:
    desired = (DesiredEntity("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "playing"),)
    result = assess_end_state(desired, seeds, ())
    assert result.status == "unevaluable"


@pytest.mark.parametrize(
    "desired",
    [
        pytest.param(
            DesiredEntity("climate.workshop", "heat", {"temperature": 20}),
            id="climate-state",
        ),
        pytest.param(
            DesiredEntity("climate.workshop", attributes={"current_temperature": 21}),
            id="current-temperature",
        ),
        pytest.param(
            DesiredEntity("cover.blinds", attributes={"current_tilt_position": 0}),
            id="cover-tilt",
        ),
    ],
)
def test_unsupported_state_or_attribute_predicate_is_unevaluable(desired: DesiredEntity) -> None:
    seed = _seed(
        desired.entity_id,
        "heat" if desired.entity_id.startswith("climate.") else "closed",
        attributes=dict(desired.attributes),
    )

    result = assess_end_state((desired,), (seed,), ())

    assert result.status == "unevaluable"


def test_one_missing_seed_makes_entire_set_unevaluable() -> None:
    desired = (DesiredEntity("light.bedroom", "on"), DesiredEntity("light.missing", "on"))
    seeds = (_seed("light.bedroom", "off"),)
    result = assess_end_state(desired, seeds, ())
    assert result.status == "unevaluable"


# ---------------------------------------------------------------------------
# Unsatisfied: action ran but final state does not match
# ---------------------------------------------------------------------------


def test_action_ran_but_final_state_mismatch_is_unsatisfied() -> None:
    desired = (DesiredEntity("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    calls = (_call("light", "turn_on", "light.bedroom"),)
    result = assess_end_state(desired, seeds, calls)
    assert result.status == "unsatisfied"
    assert result.passed is False
    assert result.comparisons[0].actual_state == "on"
