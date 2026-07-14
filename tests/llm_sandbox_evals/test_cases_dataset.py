import json
from pathlib import Path

from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.schema import DesiredState, EvalCase
import pytest


def test_authoring_schema_contains_the_closed_case_contract() -> None:
    schema_path = Path("llm_sandbox_evals/data/cases_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    case = schema["$defs"]["case"]

    assert set(case["properties"]) == {"id", "home", "user_request", "required_actions", "desired_states"}
    assert case["required"] == ["id", "home", "user_request", "required_actions"]
    assert set(schema["$defs"]["action"]["properties"]) == {
        "domain",
        "service",
        "target_entity_ids",
        "service_data",
    }
    desired = schema["$defs"]["desired_state"]
    assert set(desired["properties"]) == {"entity_id", "state"}
    assert desired["properties"]["entity_id"]["pattern"] == "^(light|switch)\\..+"
    assert desired["properties"]["state"]["enum"] == ["on", "off"]


_STATE_CASE_IDS = {
    "direct_turn_on_utility_room_ceiling",
    "direct_turn_off_utility_room_accent",
    "direct_toggle_utility_room_outlet",
    "discover_utility_room_lights",
    "discover_basement_ceiling_lights",
    "no_action_light_already_on",
    "condition_turn_off_living_room_ceiling",
    "condition_history_change_turn_off",
    "no_action_history_no_recent_change",
    "ambiguous_logic_living_room_recent",
}

_ACTION_ONLY_CASE_IDS = {
    "brightness_utility_room_ceiling",
    "color_utility_room_accent",
    "ambiguous_bare_light",
    "ambiguous_ceiling_no_area",
}


def test_corpus_state_and_action_only_split_is_exact() -> None:
    case_ids = {case.id for case in CASES}
    state_cases = {case.id for case in CASES if case.desired_states}
    action_only_cases = {case.id for case in CASES if not case.desired_states}

    assert state_cases == _STATE_CASE_IDS
    assert action_only_cases == _ACTION_ONLY_CASE_IDS
    assert state_cases | action_only_cases == case_ids


def test_basement_discovery_predicate_covers_every_required_target() -> None:
    case = next(case for case in CASES if case.id == "discover_basement_ceiling_lights")
    required_targets = set(case.required_actions[0].target_entity_ids)
    predicate_targets = {predicate.entity_id for predicate in case.desired_states}
    assert predicate_targets == required_targets
    assert all(predicate.state == "on" for predicate in case.desired_states)


def test_no_action_already_on_case_desires_on_without_actions() -> None:
    case = next(case for case in CASES if case.id == "no_action_light_already_on")
    assert case.required_actions == ()
    assert case.desired_states == (DesiredState("light.living_room_ceiling", "on"),)


def test_duplicate_desired_state_entity_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate desired state"):
        EvalCase(
            "dup",
            "home_full",
            "request",
            (),
            (
                DesiredState("light.bedroom", "on"),
                DesiredState("light.bedroom", "off"),
            ),
        )


@pytest.mark.parametrize(
    ("entity_id", "state"),
    [
        pytest.param("media_player.tv", "on", id="unsupported-domain"),
        pytest.param("light.bedroom", "playing", id="unsupported-state"),
        pytest.param("naked", "on", id="missing-domain-separator"),
    ],
)
def test_invalid_desired_state_vocabulary_is_rejected(entity_id: str, state: str) -> None:
    with pytest.raises(ValueError, match="desired state"):
        DesiredState(entity_id, state)
