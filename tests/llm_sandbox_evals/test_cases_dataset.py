from collections.abc import Sequence
from pathlib import Path

from llm_sandbox_evals.cases import CASES, load_cases
from llm_sandbox_evals.harness import _select_cases
from llm_sandbox_evals.schema import EvalCase, ExpectedAction
from pydantic_evals import Dataset


def test_load_cases_returns_native_dataset_inputs_in_stable_order() -> None:
    dataset = _native_dataset()
    loaded_cases = load_cases()

    assert [case.id for case in loaded_cases] == [case.name for case in dataset.cases]
    assert loaded_cases == [case.inputs for case in dataset.cases]
    assert loaded_cases == CASES
    assert [case.id for case in loaded_cases[:3]] == [
        "state_living_temperature",
        "state_kitchen_light",
        "state_living_fan",
    ]


def test_checked_in_dataset_round_trips_through_native_file_format(tmp_path: Path) -> None:
    dataset = _native_dataset()
    roundtrip_path = tmp_path / "cases.yaml"

    dataset.to_file(roundtrip_path)
    reloaded = Dataset[EvalCase, object, object].from_file(roundtrip_path)

    assert [case.name for case in reloaded.cases] == [case.name for case in dataset.cases]
    assert [case.inputs.id for case in reloaded.cases] == [case.inputs.id for case in dataset.cases]
    assert [case.inputs for case in reloaded.cases] == [case.inputs for case in dataset.cases]


def test_loaded_dataset_preserves_structural_expectation_types() -> None:
    loaded_cases = load_cases()
    domain_limited_case = _case_by_id(loaded_cases, "action_domain_not_allowed")
    aggregate_case = _case_by_id(loaded_cases, "recorder_aggregate_last_seen_to_state")
    multi_action_case = _case_by_id(loaded_cases, "action_multi_sequence")

    assert isinstance(domain_limited_case.action_domains, frozenset)
    assert domain_limited_case.action_domains == frozenset({"light"})
    assert aggregate_case.expected.required_tool_arg_values == (("aggregate", "last_seen"), ("to_state", "on"))
    assert tuple(isinstance(pair, tuple) for pair in aggregate_case.expected.required_tool_arg_values) == (True, True)
    assert aggregate_case.expected.required_result_paths == ("summary", "mode")
    assert aggregate_case.expected.recorder_window == (
        "2026-06-28T12:00:00+00:00",
        "2026-06-29T12:00:00+00:00",
    )
    assert multi_action_case.expected.actions == (
        ExpectedAction(domain="light", service="turn_off", target_entity_ids=("light.living",)),
        ExpectedAction(domain="fan", service="set_percentage", target_entity_ids=("fan.living_fan",)),
    )


def test_harness_select_cases_consumes_loaded_suite_order() -> None:
    selected_cases = _select_cases(None, None)

    assert selected_cases == CASES
    assert [case.id for case in _select_cases(["recorder_read"], None)[:3]] == [
        "recorder_living_temp_history",
        "recorder_bedroom_humidity_statistics",
        "recorder_living_light_logbook",
    ]


def _native_dataset() -> Dataset[EvalCase, object, object]:
    dataset_path = Path(__file__).parents[2] / "llm_sandbox_evals" / "data" / "cases.yaml"
    return Dataset[EvalCase, object, object].from_file(dataset_path)


def _case_by_id(cases: Sequence[EvalCase], case_id: str) -> EvalCase:
    return next(case for case in cases if case.id == case_id)
