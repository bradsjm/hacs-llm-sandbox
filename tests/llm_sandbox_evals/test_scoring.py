from typing import cast

import pytest
from custom_components.llm_sandbox.const import TOOL_GET_HISTORY
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected, StepTrace, ToolCall
from llm_sandbox_evals.scoring import check_case


@pytest.mark.parametrize(
    ("tool_args", "expected_passed"),
    [
        pytest.param({"aggregate": "last_seen", "to_state": "on"}, True, id="matching-args"),
        pytest.param({"aggregate": "last_seen", "to_state": "off"}, False, id="wrong-to-state"),
        pytest.param({"aggregate": "last_seen"}, False, id="missing-to-state"),
    ],
)
def test_required_tool_arg_values_check(tool_args: dict[str, object], expected_passed: bool) -> None:
    checks = check_case(
        _case(),
        "",
        (),
        set(),
        _snapshot(),
        (StepTrace(tool_calls=(ToolCall("call-1", TOOL_GET_HISTORY, tool_args),), tool_results=({},)),),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["required_tool_arg_values"] is expected_passed


def test_required_tool_arg_values_check_fails_when_values_are_split_across_calls() -> None:
    checks = check_case(
        _case(),
        "",
        (),
        set(),
        _snapshot(),
        (
            StepTrace(
                tool_calls=(
                    ToolCall("call-1", TOOL_GET_HISTORY, {"aggregate": "last_seen"}),
                    ToolCall("call-2", TOOL_GET_HISTORY, {"to_state": "on"}),
                ),
                tool_results=({}, {}),
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["required_tool_arg_values"] is False


def test_max_tool_calls_check_counts_all_observed_tool_calls() -> None:
    checks = check_case(
        _case(max_tool_calls=1),
        "",
        (),
        set(),
        _snapshot(),
        (
            StepTrace(
                tool_calls=(ToolCall("call-1", TOOL_GET_HISTORY, {"aggregate": "last_seen", "to_state": "on"}),),
                tool_results=({},),
            ),
            StepTrace(
                tool_calls=(ToolCall("call-2", TOOL_GET_HISTORY, {"entity_ids": ["light.living"]}),),
                tool_results=({},),
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["required_tool_arg_values"] is True
    assert passed_by_name["tool_calls_within_max"] is False


def _case(max_tool_calls: int | None = None) -> EvalCase:
    return EvalCase(
        id="scoring-unit",
        category="unit",
        home="home_default",
        user_request="When did light.living last turn on?",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(
            tool_name=TOOL_GET_HISTORY,
            execution_status="na",
            required_tool_arg_values=(("aggregate", "last_seen"), ("to_state", "on")),
            max_tool_calls=max_tool_calls,
        ),
        par_turns=1,
    )


def _snapshot() -> HomeSnapshot:
    return cast(HomeSnapshot, get_home("home_default").snapshot())
