from collections.abc import Sequence
from pathlib import Path

from llm_sandbox_evals.cases import CASES, load_cases
from llm_sandbox_evals.harness import _select_cases
from llm_sandbox_evals.schema import EvalCase
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
