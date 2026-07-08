import pytest
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected, ExpectedAction
from llm_sandbox_evals.scoring import check_case, score_case


def test_outcome_scoring_passes_facts_actions_and_efficiency() -> None:
    checks = check_case(
        _case(
            Expected(
                answer_facts=("sensor.living_temp",),
                actions=(ExpectedAction("light", "turn_off", ("light.living",)),),
                max_tool_calls=4,
                reference_tool_calls=1,
            )
        ),
        "The value came from sensor.living_temp.",
        ({"domain": "light", "service": "turn_off", "target": {"entity_id": "light.living"}},),
        2,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name == {
        "answer_facts_present": True,
        "actions_match": True,
        "tool_calls_within_max": True,
        "tool_call_efficiency": False,
    }
    assert score_case(checks) == pytest.approx(0.875)


def test_required_gate_failure_forces_zero_score() -> None:
    checks = check_case(
        _case(Expected(answer_facts=("sensor.living_temp",), max_tool_calls=1, reference_tool_calls=1)),
        "No matching fact here.",
        (),
        2,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["answer_facts_present"] is False
    assert passed_by_name["tool_calls_within_max"] is False
    assert score_case(checks) == 0.0


def test_empty_expected_actions_rejects_unexpected_recorded_action() -> None:
    checks = check_case(
        _case(Expected()),
        "",
        ({"domain": "light", "service": "turn_on", "target": {"entity_id": "light.living"}},),
        1,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_expected_action_rejects_extra_recorded_action() -> None:
    checks = check_case(
        _case(
            Expected(
                answer_facts=("done",),
                actions=(ExpectedAction("light", "turn_on", ("light.living",)),),
            )
        ),
        "done",
        (
            {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.living"}},
            {"domain": "lock", "service": "unlock", "target": {"entity_id": "lock.front_door"}},
        ),
        1,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def _case(expected: Expected) -> EvalCase:
    return EvalCase(
        id="scoring-unit",
        category="unit",
        home="home_default",
        user_request="score this outcome",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=expected,
    )
