import json
from pathlib import Path

from llm_sandbox_evals.cases import CASES

_EXPECTED = [
    ("action_turn_on_bedroom_light", "Turn on bedroom light", "turn_on", "light.bedroom"),
    ("action_turn_off_bedroom_light", "Turn off bedroom light", "turn_off", "light.bedroom"),
    ("action_turn_on_living_light", "Turn on living room light", "turn_on", "light.living"),
    ("action_turn_off_living_light", "Turn off living room light", "turn_off", "light.living"),
]


def test_dataset_is_exactly_the_four_action_baseline_cases() -> None:
    assert [
        (case.id, case.user_request, case.expected_actions[0].service, case.expected_actions[0].target_entity_ids[0])
        for case in CASES
    ] == _EXPECTED
    assert all(case.home == "home_minimal" for case in CASES)
    assert all(len(case.expected_actions) == 1 for case in CASES)
    assert all(case.expected_actions[0].domain == "light" for case in CASES)


def test_authoring_schema_contains_only_the_minimal_case_contract() -> None:
    schema_path = Path("llm_sandbox_evals/data/cases_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    case = schema["$defs"]["case"]

    assert set(case["properties"]) == {"id", "home", "user_request", "expected_actions"}
    assert case["required"] == ["id", "home", "user_request", "expected_actions"]
    assert set(schema["$defs"]["action"]["properties"]) == {
        "domain",
        "service",
        "target_entity_ids",
        "service_data",
    }
