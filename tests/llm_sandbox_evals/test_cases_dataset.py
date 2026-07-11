from collections.abc import Sequence
from pathlib import Path
import re

from llm_sandbox_evals.cases import CASES, load_cases
from llm_sandbox_evals.harness import _select_cases
from llm_sandbox_evals.schema import EvalCase
from pydantic_evals import Dataset

_REMOVED_TOOL_CONTRACT_CASE_IDS = frozenset(
    {
        "sql_query_visible_entities_by_domain",
        "recovery_recorder_entity_not_visible",
        "recovery_missing_state_note",
        "recovery_service_target_not_visible",
        "recovery_service_not_found",
        "recovery_no_retry_resolved_from",
        "recovery_selector_no_match",
        "recovery_bad_iso_window",
        "recovery_code_error_one_retry",
    }
)
_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_DEPENDENT_RECORDER_CASE_IDS = frozenset(
    {
        "multi_discover_living_temp_history",
        "multi_history_then_living_fan",
        "multi_logbook_then_living_light_off",
        "multi_history_then_living_light_off",
        "multi_state_and_logbook_living_light",
        "multi_real_history_then_close_blinds",
    }
)


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


def test_dataset_excludes_removed_tool_contract_cases() -> None:
    case_ids = {case.id for case in CASES}

    assert not (case_ids & _REMOVED_TOOL_CONTRACT_CASE_IDS)


def test_dataset_cases_have_meaningful_oracles() -> None:
    weak_case_ids = [case.id for case in CASES if not _has_meaningful_oracle(case)]

    assert weak_case_ids == []


def test_dataset_does_not_use_legacy_expected_values() -> None:
    legacy_case_ids = [case.id for case in CASES if case.expected.expected_values]

    assert legacy_case_ids == []


def test_dataset_does_not_use_answer_only_oracles() -> None:
    answer_only_case_ids = [
        case.id for case in CASES if case.expected.answer_values and not _has_structured_oracle(case)
    ]

    assert answer_only_case_ids == []


def test_dataset_answer_values_do_not_require_raw_entity_ids_unless_requested() -> None:
    violating_case_ids = [
        case.id
        for case in CASES
        if _answer_values_include_entity_id(case) and not _request_asks_for_technical_identifiers(case.user_request)
    ]

    assert violating_case_ids == []


def test_blocked_cases_expect_blocked_outcome_without_successful_actions() -> None:
    invalid_blocked_case_ids = [
        case.id
        for case in CASES
        if case.category == "action_blocked" and (case.expected.blocked_outcome is None or case.expected.actions)
    ]

    assert invalid_blocked_case_ids == []


def test_dependent_recorder_cases_require_one_composed_execute_call() -> None:
    dependent_cases = [case for case in CASES if case.id in _DEPENDENT_RECORDER_CASE_IDS]

    assert {case.id for case in dependent_cases} == _DEPENDENT_RECORDER_CASE_IDS
    assert all(case.expected.tool_call_par == 1 for case in dependent_cases)
    assert all(
        [check.tool_name for check in case.expected.tool_result_checks] == ["execute_home_code"]
        for case in dependent_cases
    )


def test_direct_recorder_cases_retain_standalone_tools_and_parallel_policy() -> None:
    direct_cases = [
        case for case in CASES if case.category == "recorder_read" and case.id not in _DEPENDENT_RECORDER_CASE_IDS
    ]
    direct_tool_names = {check.tool_name for case in direct_cases for check in case.expected.tool_result_checks}
    parallel_case = _case_by_id(CASES, "multi_parallel_temp_history_humidity_stats")

    assert direct_tool_names == {"get_history", "get_statistics", "get_logbook"}
    assert [check.tool_name for check in parallel_case.expected.tool_result_checks] == [
        "get_history",
        "get_statistics",
    ]
    assert parallel_case.expected.tool_call_par == 2


def test_automation_cases_are_exactly_the_three_direct_read_cases() -> None:
    automation_cases = [case for case in CASES if case.category == "automation_read"]

    assert {case.id for case in automation_cases} == {
        "automation_discover_evening_living_lights",
        "automation_explain_evening_living_lights",
        "automation_recent_evening_living_lights_run",
    }
    assert all(
        [check.tool_name for check in case.expected.tool_result_checks] == ["get_automation"]
        for case in automation_cases
    )
    assert all(
        len(case.expected.tool_result_checks) == 1 and case.expected.tool_call_par is None for case in automation_cases
    )
    assert all(case.expected.blocked_outcome is None for case in automation_cases)


def _native_dataset() -> Dataset[EvalCase, object, object]:
    dataset_path = Path(__file__).parents[2] / "llm_sandbox_evals" / "data" / "cases.yaml"
    return Dataset[EvalCase, object, object].from_file(dataset_path)


def _case_by_id(cases: Sequence[EvalCase], case_id: str) -> EvalCase:
    return next(case for case in cases if case.id == case_id)


def _has_meaningful_oracle(case: EvalCase) -> bool:
    return _has_structured_oracle(case)


def _has_structured_oracle(case: EvalCase) -> bool:
    expected = case.expected
    return bool(
        expected.provenance_values
        or expected.tool_result_checks
        or expected.actions
        or expected.blocked_outcome is not None
    )


def _answer_values_include_entity_id(case: EvalCase) -> bool:
    return any(_ENTITY_ID_RE.search(value) is not None for value in case.expected.answer_values)


def _request_asks_for_technical_identifiers(user_request: str) -> bool:
    lowered = user_request.lower()
    return "entity id" in lowered or "entity ids" in lowered or "technical identifier" in lowered
