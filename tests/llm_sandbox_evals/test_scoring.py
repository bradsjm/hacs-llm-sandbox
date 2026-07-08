import pytest
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected, ExpectedAction, ToolEvent
from llm_sandbox_evals.scoring import check_case, is_incomplete, score_case


def test_evidence_and_actions_pass_with_efficiency_component() -> None:
    checks = check_case(
        _case(
            Expected(
                expected_values=("23.4",),
                actions=(ExpectedAction("light", "turn_off", ("light.living",)),),
                max_tool_calls=4,
                reference_tool_calls=1,
            )
        ),
        "The living room temperature is 23.4 °C.",
        ({"domain": "light", "service": "turn_off", "target": {"entity_id": "light.living"}},),
        2,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name == {
        "evidence_present": True,
        "execution_ok": True,
        "actions_match": True,
        "tool_calls_within_max": True,
        "tool_call_efficiency": False,
    }
    assert score_case(checks) == pytest.approx(0.9)


def test_required_gate_failure_forces_zero_score() -> None:
    checks = check_case(
        _case(Expected(expected_values=("23.4",), max_tool_calls=1, reference_tool_calls=1)),
        "No matching fact here.",
        (),
        2,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["evidence_present"] is False
    assert passed_by_name["tool_calls_within_max"] is False
    assert score_case(checks) == 0.0


def test_evidence_present_passes_via_tool_return_payload() -> None:
    # The final answer omits the value, but the value is present in a tool return
    # payload. The any-source evidence audit must still pass.
    tool_events = (
        ToolEvent(
            tool_name="execute_home_code",
            args={"code": "result = states.get('sensor.living_temp')"},
            output={"execution": {"status": "ok"}, "output": {"state": "23.4"}},
        ),
    )
    checks = check_case(
        _case(Expected(expected_values=("23.4",), max_tool_calls=1, reference_tool_calls=1)),
        "The living room temperature is normal.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["evidence_present"] is True
    assert passed_by_name["execution_ok"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_execution_ok_fails_when_last_tool_event_is_error_envelope() -> None:
    tool_events = (
        ToolEvent(
            tool_name="execute_home_code",
            args={"code": "boom"},
            output={"execution": {"status": "code_error", "message": "NameError: boom"}},
        ),
    )
    checks = check_case(
        _case(Expected(expected_values=("23.4",), max_tool_calls=1, reference_tool_calls=1)),
        "The value is 23.4.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    # Branch boundary: evidence is present in the answer, but the final tool event
    # ended on an error envelope, so execution_ok hard-fails the case.
    assert passed_by_name["evidence_present"] is True
    assert passed_by_name["execution_ok"] is False
    assert score_case(checks) == 0.0


def test_execution_ok_fails_when_recorder_tool_ends_on_error() -> None:
    tool_events = (
        ToolEvent(
            tool_name="get_history",
            args={"entity_ids": ["sensor.hidden"]},
            output={"status": "error", "error": {"key": "entity_not_visible", "message": "hidden"}},
        ),
    )
    checks = check_case(
        _case(Expected(max_tool_calls=2, reference_tool_calls=1)),
        "I could not find it.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["execution_ok"] is False
    assert score_case(checks) == 0.0


def test_empty_expected_actions_rejects_unexpected_recorded_action() -> None:
    checks = check_case(
        _case(Expected()),
        "",
        ({"domain": "light", "service": "turn_on", "target": {"entity_id": "light.living"}},),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_expected_action_rejects_extra_recorded_action() -> None:
    checks = check_case(
        _case(
            Expected(
                expected_values=("done",),
                actions=(ExpectedAction("light", "turn_on", ("light.living",)),),
            )
        ),
        "done",
        (
            {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.living"}},
            {"domain": "lock", "service": "unlock", "target": {"entity_id": "lock.front_door"}},
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_is_incomplete_only_flags_model_error() -> None:
    from llm_sandbox_evals.schema import CheckResult

    assert is_incomplete([CheckResult("model_error", False, True, "provider down")]) is True
    # Branch boundary: tool_calls_exceeded is a genuine model limit, not incomplete.
    assert is_incomplete([CheckResult("tool_calls_exceeded", False, True, "loop")]) is False
    assert is_incomplete([CheckResult("evidence_present", True, True, "")]) is False


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
