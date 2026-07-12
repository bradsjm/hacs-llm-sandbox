from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
from typing import cast

from llm_sandbox_evals.cases import CASES, load_cases
from llm_sandbox_evals.harness import _select_cases
from llm_sandbox_evals.schema import EvalCase
from pydantic_evals import Dataset

_DELETED_CASE_IDS = frozenset(
    {
        "multi_invalid_then_recover_living_temp_history",
        "multi_hidden_garage_safety",
        "action_ambiguous_target",
        "recorder_selector_floor_target",
    }
)
_CATEGORY_COUNTS = {
    "state": 18,
    "registry": 8,
    "history": 12,
    "statistics": 5,
    "logbook": 4,
    "automation": 3,
    "action": 26,
    "safety": 2,
    "system": 2,
}


def test_load_cases_returns_native_dataset_inputs_in_stable_order() -> None:
    dataset = _native_dataset()
    loaded_cases = load_cases()

    assert [case.id for case in loaded_cases] == [case.name for case in dataset.cases]
    assert loaded_cases == [case.inputs for case in dataset.cases]
    assert loaded_cases == CASES


def test_harness_select_cases_consumes_loaded_suite_order() -> None:
    assert _select_cases(None, None) == CASES
    assert [case.id for case in _select_cases(["history"], None)[:3]] == [
        "recorder_living_temp_history",
        "multi_discover_living_temp_history",
        "multi_parallel_temp_history_humidity_stats",
    ]


def test_dataset_cases_use_structured_oracles() -> None:
    provenance_only = [
        case.id
        for case in CASES
        if case.expected.provenance_values
        and not case.expected.tool_result_checks
        and not case.expected.actions
        and case.expected.blocked_outcome is None
    ]
    non_action_reads_without_results = [
        case.id
        for case in CASES
        if not case.expected.actions and case.expected.blocked_outcome is None and not case.expected.tool_result_checks
    ]

    assert provenance_only == []
    assert non_action_reads_without_results == []


def test_blocked_cases_require_expected_rejections_without_allowed_effects() -> None:
    blocked_cases = [case for case in CASES if case.expected.blocked_outcome is not None]

    assert len(blocked_cases) == 6
    assert all(case.expected.blocked_outcome.error_keys for case in blocked_cases)
    assert all(case.expected.blocked_outcome.actions for case in blocked_cases)
    assert all(not case.expected.actions for case in blocked_cases)


def test_cases_use_the_primary_objective_category_taxonomy() -> None:
    case_ids = {case.id for case in CASES}
    schema_category = _cases_schema_category()
    no_data_case = _case_by_id(CASES, "recovery_statistic_no_data")

    assert not (case_ids & _DELETED_CASE_IDS)
    assert Counter(case.category for case in CASES) == _CATEGORY_COUNTS
    assert set(schema_category["enum"]) == set(_CATEGORY_COUNTS)
    assert schema_category["maxLength"] == 13
    assert all(len(case.category) <= 13 for case in CASES)
    assert no_data_case.category == "statistics"
    assert no_data_case.expected.tool_result_checks[0].min_results == 0


def test_context_device_ids_and_action_domain_override_are_preserved() -> None:
    assert _case_by_id(CASES, "state_living_fan").llm_context.device_id == "device_assist_living"
    assert _case_by_id(CASES, "context_location_scope_override").llm_context.device_id == "device_router"
    assert _case_by_id(CASES, "action_domain_not_allowed").action_domains == frozenset({"light"})


def test_real_office_inventory_requires_all_fixture_backed_entities_and_devices() -> None:
    office = _case_by_id(CASES, "real_office_entities")
    check = office.expected.tool_result_checks[0]

    assert set(check.entity_ids) == {
        "cover.office_blinds",
        "light.office_lights_group",
        "sensor.office_air_quality_monitor_pm25",
    }
    assert set(check.entry_values) == {
        "Office Blinds",
        "Office Air Quality Monitor",
        "Office Light Group",
        "Office Air Quality Monitor Pm25",
    }


def _native_dataset() -> Dataset[EvalCase, object, object]:
    dataset_path = Path(__file__).parents[2] / "llm_sandbox_evals" / "data" / "cases.yaml"
    return Dataset[EvalCase, object, object].from_file(dataset_path)


def _cases_schema_category() -> dict[str, object]:
    schema_path = Path(__file__).parents[2] / "llm_sandbox_evals" / "data" / "cases_schema.json"
    schema = cast(dict[str, object], json.loads(schema_path.read_text(encoding="utf-8")))
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    case = definitions["case"]
    assert isinstance(case, dict)
    properties = case["properties"]
    assert isinstance(properties, dict)
    category = properties["category"]
    assert isinstance(category, dict)
    enum = category["enum"]
    assert isinstance(enum, list)
    assert all(isinstance(value, str) for value in enum)
    return category


def _case_by_id(cases: Sequence[EvalCase], case_id: str) -> EvalCase:
    return next(case for case in cases if case.id == case_id)
