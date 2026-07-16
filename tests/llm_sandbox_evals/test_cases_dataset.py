from collections import Counter
from collections.abc import Callable
import json
from pathlib import Path

from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.schema import (
    AnswerPredicate,
    DesiredEntity,
    EvalCase,
    ExpectedToolCall,
    RequestVariant,
    RequiredAction,
)
import pytest


def test_authoring_schema_contains_the_closed_case_contract() -> None:
    schema_path = Path("llm_sandbox_evals/data/cases_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    case = schema["$defs"]["case"]

    assert set(case["properties"]) == {
        "id",
        "home",
        "category",
        "tags",
        "judge_code",
        "oracle",
        "requests",
        "required_actions",
        "desired_entities",
        "expected_tool_calls",
        "expected_answer",
    }
    assert case["required"] == ["id", "home", "category", "requests", "required_actions"]
    assert case["additionalProperties"] is False
    assert case["properties"]["judge_code"] == {"type": "boolean", "default": False}
    assert set(schema["$defs"]["action"]["properties"]) == {
        "domain",
        "service",
        "target_entity_ids",
        "service_data",
    }
    desired = schema["$defs"]["desired_entity"]
    assert set(desired["properties"]) == {"entity_id", "state", "attributes"}
    assert desired["required"] == ["entity_id"]
    assert desired["anyOf"] == [{"required": ["state"]}, {"required": ["attributes"]}]


_DESIRED_ENTITY_CASE_IDS = {
    "direct_turn_on_utility_room_ceiling",
    "direct_turn_off_utility_room_accent",
    "direct_toggle_utility_room_outlet",
    "discover_basement_ceiling_lights",
    "brightness_utility_room_ceiling",
    "color_utility_room_accent",
    "no_action_light_already_on",
    "condition_turn_off_living_room_ceiling",
    "condition_history_change_turn_off",
    "no_action_history_no_recent_change",
    "ambiguous_logic_living_room_recent",
    "selector_evening_lights_living_room_off",
    "multi_action_swap_utility_room_lights",
    "condition_climate_below_threshold_set_temperature",
    "no_action_climate_not_below_threshold",
    "cover_open_office_blinds",
    "cover_close_bedroom_shade",
    "cover_position_office_blinds_half",
    "cover_no_action_bedroom_shade_not_closed",
}

_ACTION_ONLY_CASE_IDS = {
    "discover_utility_room_lights",
    "ambiguous_bare_light",
    "ambiguous_ceiling_no_area",
}

_JUDGE_CODE_CASE_IDS = {
    "discover_utility_room_lights",
    "discover_basement_ceiling_lights",
    "condition_history_change_turn_off",
    "no_action_history_no_recent_change",
    "ambiguous_logic_living_room_recent",
}


def test_effect_corpus_desired_entity_and_action_only_split_is_exact() -> None:
    case_ids = {case.id for case in CASES}
    effect_cases = {case.id for case in CASES if case.oracle == "effect"}
    desired_entity_cases = {case.id for case in CASES if case.oracle == "effect" and case.desired_entities}
    action_only_cases = {case.id for case in CASES if case.oracle == "effect" and not case.desired_entities}

    assert desired_entity_cases == _DESIRED_ENTITY_CASE_IDS
    assert action_only_cases == _ACTION_ONLY_CASE_IDS
    assert len(desired_entity_cases) == 19
    assert len(action_only_cases) == 3
    assert desired_entity_cases | action_only_cases == effect_cases
    assert len(case_ids) == 36



def test_corpus_category_distribution_is_exact() -> None:
    assert Counter(case.category for case in CASES) == {
        "direct": 5,
        "discovery": 3,
        "service_data": 3,
        "conditional": 7,
        "ambiguity": 3,
        "tool_contract": 6,
        "read_answer": 8,
        "composition": 1,
    }

def test_dedicated_oracle_cases_author_narrow_contracts() -> None:
    cases_by_id = {case.id: case for case in CASES}

    assert cases_by_id["tool_call_get_history_utility_room"].oracle == "tool_calls"
    assert cases_by_id["tool_call_get_history_utility_room"].expected_tool_calls == (
        ExpectedToolCall("get_history", {"entity_ids": ["light.utility_room_ceiling"]}),
    )
    assert cases_by_id["answer_count_lights_on_utility_room"].expected_answer == AnswerPredicate("count", count=1)
    assert cases_by_id["answer_state_utility_room_accent"].expected_answer == AnswerPredicate(
        "state",
        entity_id="light.utility_room_accent",
        state="on",
    )
    assert cases_by_id["tool_call_get_statistics_balcony_power"].expected_tool_calls == (
        ExpectedToolCall("get_statistics", {"statistic_ids": ["sensor.balcony_power"]}),
    )
    assert cases_by_id["tool_call_get_logbook_living_room_accent"].expected_tool_calls == (
        ExpectedToolCall("get_logbook", {"entity_ids": ["light.living_room_accent"]}),
    )
    assert cases_by_id["answer_mean_balcony_power"].expected_answer == AnswerPredicate(
        "scalar",
        scalar_value=40,
        tolerance=0,
    )
    assert cases_by_id["answer_both_living_room_lights_turned_on"].expected_answer == AnswerPredicate(
        "boolean",
        value=True,
    )
    assert cases_by_id["answer_living_room_lights_turned_on_set"].expected_answer == AnswerPredicate(
        "entity_set",
        entity_ids=("light.living_room_ceiling", "light.living_room_accent"),
    )
    assert cases_by_id["answer_living_room_accent_turn_on_time"].expected_answer == AnswerPredicate(
        "time_range",
        start="2026-06-29T11:50:00+00:00",
        end="2026-06-29T11:50:00+00:00",
    )
    assert cases_by_id["answer_count_climate_label_second_floor"].expected_answer == AnswerPredicate("count", count=9)


def test_authored_code_judge_selection_is_exact() -> None:
    assert {case.id for case in CASES if case.judge_code} == _JUDGE_CODE_CASE_IDS
    assert all(case.judge_code is False for case in CASES if case.id not in _JUDGE_CODE_CASE_IDS)


def _judge_effect_case() -> EvalCase:
    return EvalCase(
        "judge-effect",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Turn on the bedroom light."),),
        (),
        judge_code=True,
    )


def _judge_tool_calls_case() -> EvalCase:
    return EvalCase(
        "judge-tool-calls",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Check the bedroom light history."),),
        (),
        oracle="tool_calls",
        expected_tool_calls=(ExpectedToolCall("get_history"),),
        judge_code=True,
    )


def _judge_answer_case() -> EvalCase:
    return EvalCase(
        "judge-answer",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Is the bedroom light on?"),),
        (),
        oracle="answer",
        expected_answer=AnswerPredicate("boolean", value=True),
        judge_code=True,
    )


@pytest.mark.parametrize(
    "case_factory",
    [
        pytest.param(_judge_effect_case, id="effect"),
        pytest.param(_judge_tool_calls_case, id="tool-calls"),
        pytest.param(_judge_answer_case, id="answer"),
    ],
)
def test_any_oracle_type_can_explicitly_opt_into_code_judging(case_factory: Callable[[], EvalCase]) -> None:
    case = case_factory()

    assert case.judge_code is True


def test_utility_discovery_uses_exact_action_fallback() -> None:
    case = next(case for case in CASES if case.id == "discover_utility_room_lights")

    assert case.desired_entities == ()
    assert len(case.required_actions) == 1
    assert case.required_actions[0].target_entity_ids == (
        "light.utility_room_accent",
        "light.utility_room_ceiling",
    )


def test_evening_living_room_selector_uses_state_primary_scoring() -> None:
    case = next(case for case in CASES if case.id == "selector_evening_lights_living_room_off")

    assert case.required_actions == (
        RequiredAction(
            "light",
            "turn_off",
            ("light.living_room_ceiling", "light.living_room_accent"),
        ),
    )
    assert case.desired_entities == (
        DesiredEntity("light.living_room_ceiling", state="off"),
        DesiredEntity("light.living_room_accent", state="off"),
    )


def test_basement_discovery_predicate_covers_every_required_target() -> None:
    case = next(case for case in CASES if case.id == "discover_basement_ceiling_lights")
    required_targets = set(case.required_actions[0].target_entity_ids)
    predicate_targets = {predicate.entity_id for predicate in case.desired_entities}
    assert len(case.desired_entities) == 12
    assert predicate_targets == required_targets
    assert all(predicate.state == "on" for predicate in case.desired_entities)


@pytest.mark.parametrize(
    ("case_id", "service_data"),
    [
        pytest.param("brightness_utility_room_ceiling", {"brightness_pct": 50}, id="brightness"),
        pytest.param("color_utility_room_accent", {"color_temp_kelvin": 2700}, id="color-temperature"),
    ],
)
def test_attribute_action_cases_author_final_values(case_id: str, service_data: dict[str, object]) -> None:
    case = next(case for case in CASES if case.id == case_id)

    assert case.required_actions[0].service_data == service_data
    expected_attributes = {
        "brightness_utility_room_ceiling": {"brightness": 128},
        "color_utility_room_accent": {"color_temp_kelvin": 2700},
    }
    assert case.desired_entities[0].attributes == expected_attributes[case_id]


def test_color_action_case_uses_the_explicit_temperature_request() -> None:
    case = next(case for case in CASES if case.id == "color_utility_room_accent")

    assert case.requests[0].text == "Set the Utility Room accent light to 2700 K warm white."


def test_no_action_already_on_case_desires_on_without_actions() -> None:
    case = next(case for case in CASES if case.id == "no_action_light_already_on")
    assert case.required_actions == ()
    assert case.desired_entities == (DesiredEntity("light.living_room_ceiling", "on"),)


def test_second_tranche_effect_case_contracts_are_exact() -> None:
    cases = {case.id: case for case in CASES}

    assert cases["multi_action_swap_utility_room_lights"].required_actions == (
        RequiredAction("light", "turn_off", ("light.utility_room_accent",)),
        RequiredAction("light", "turn_on", ("light.utility_room_ceiling",)),
    )
    assert cases["multi_action_swap_utility_room_lights"].desired_entities == (
        DesiredEntity("light.utility_room_accent", "off"),
        DesiredEntity("light.utility_room_ceiling", "on"),
    )
    assert cases["condition_climate_below_threshold_set_temperature"].required_actions == (
        RequiredAction("climate", "set_temperature", ("climate.workshop",), {"temperature": 22}),
    )
    assert cases["condition_climate_below_threshold_set_temperature"].desired_entities == (
        DesiredEntity("climate.workshop", attributes={"temperature": 22}),
    )
    assert cases["no_action_climate_not_below_threshold"].required_actions == ()
    assert cases["no_action_climate_not_below_threshold"].desired_entities == (
        DesiredEntity("climate.storage_room", attributes={"temperature": 20.0}),
    )
    assert cases["cover_open_office_blinds"].required_actions == (
        RequiredAction("cover", "open_cover", ("cover.office_blinds",)),
    )
    assert cases["cover_open_office_blinds"].desired_entities == (
        DesiredEntity("cover.office_blinds", "open", {"current_position": 100}),
    )
    assert cases["cover_close_bedroom_shade"].required_actions == (
        RequiredAction("cover", "close_cover", ("cover.bedroom_shade",)),
    )
    assert cases["cover_close_bedroom_shade"].desired_entities == (
        DesiredEntity("cover.bedroom_shade", "closed", {"current_position": 0}),
    )
    assert cases["cover_position_office_blinds_half"].required_actions == (
        RequiredAction("cover", "set_cover_position", ("cover.office_blinds",), {"position": 50}),
    )
    assert cases["cover_position_office_blinds_half"].desired_entities == (
        DesiredEntity("cover.office_blinds", "open", {"current_position": 50}),
    )
    assert cases["cover_no_action_bedroom_shade_not_closed"].required_actions == ()
    assert cases["cover_no_action_bedroom_shade_not_closed"].desired_entities == (
        DesiredEntity("cover.bedroom_shade", "open", {"current_position": 100}),
    )


def test_second_tranche_automation_case_contracts_are_exact() -> None:
    cases = {case.id: case for case in CASES}

    assert cases["tool_call_get_automation_motion_lights"].expected_tool_calls == (
        ExpectedToolCall("get_automation", {"query": "motion lights"}),
    )
    assert cases["tool_call_get_automation_motion_lights_content"].expected_tool_calls == (
        ExpectedToolCall(
            "get_automation",
            {
                "entity_ids": ["automation.living_room_motion_lights"],
                "include": ["content"],
            },
        ),
    )
    assert cases["tool_call_get_automation_motion_lights_runs"].expected_tool_calls == (
        ExpectedToolCall(
            "get_automation",
            {
                "entity_ids": ["automation.living_room_motion_lights"],
                "include": ["runs"],
                "hours": 1,
            },
        ),
    )
    assert cases["answer_living_room_motion_automation_ran_last_hour"].expected_answer == AnswerPredicate(
        "boolean",
        value=True,
    )


def test_duplicate_desired_entity_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate desired entity"):
        EvalCase(
            "dup",
            "home_full",
            "test",
            (RequestVariant("canonical", "request"),),
            (),
            (
                DesiredEntity("light.bedroom", "on"),
                DesiredEntity("light.bedroom", "off"),
            ),
        )


def test_desired_entity_requires_an_authored_final_value() -> None:
    with pytest.raises(ValueError, match="state or attributes"):
        DesiredEntity("light.bedroom")


def test_desired_entity_does_not_restrict_domain_or_state_vocabulary() -> None:
    assert DesiredEntity("media_player.tv", "playing").state == "playing"
