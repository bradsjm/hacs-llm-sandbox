from llm_sandbox_evals.schema import EvalCase, ExpectedToolCall, RequestVariant, ToolEvent
from llm_sandbox_evals.scoring.evaluate import evaluate_case
from llm_sandbox_evals.scoring.tool_calls import score_tool_calls


def _event(
    tool_name: str,
    args: dict[str, object],
    *,
    output: dict[str, object] | None = None,
) -> ToolEvent:
    default_output = {
        "get_history": {"rows": []},
        "get_statistics": {"statistics": {}},
    }.get(tool_name, {})
    return ToolEvent(tool_name, args, default_output if output is None else output)


def test_tool_calls_match_one_to_one_with_canonical_arguments() -> None:
    expected = (
        ExpectedToolCall("get_history", {"options": {"limit": 1, "values": [1, 2.0]}}),
        ExpectedToolCall("get_history", {"entity_ids": ["light.utility_room_ceiling"]}),
    )
    events = (
        _event("get_history", {"entity_ids": ["light.utility_room_ceiling"]}),
        _event("get_history", {"options": {"values": [1.0, 2], "limit": 1.0}}),
    )

    result = score_tool_calls(expected, events)

    assert result.passed is True
    assert result.reason == "tool_calls_matched"
    assert tuple(comparison.matched_event for comparison in result.comparisons) == (events[1], events[0])
    assert result.unmatched_events == ()


def test_tool_calls_do_not_reuse_one_event_for_duplicate_expectations() -> None:
    expected = (ExpectedToolCall("get_history"), ExpectedToolCall("get_history"))
    event = _event("get_history", {})

    result = score_tool_calls(expected, (event,))

    assert result.passed is False
    assert result.reason == "tool_calls_missing"
    assert tuple(comparison.matched_event for comparison in result.comparisons) == (event, None)


def test_tool_calls_distinguish_missing_from_mismatched_arguments() -> None:
    expected = (ExpectedToolCall("get_history", {"entity_ids": ["light.a"]}),)

    missing = score_tool_calls(expected, (_event("get_statistics", {}),))
    mismatched = score_tool_calls(expected, (_event("get_history", {"entity_ids": ["light.b"]}),))

    assert missing.reason == "tool_calls_missing"
    assert mismatched.reason == "tool_calls_mismatched"


def test_extra_successful_events_are_diagnostic_only() -> None:
    expected = (ExpectedToolCall("get_history"),)
    matched = _event("get_history", {})
    extra = _event("get_statistics", {})

    result = score_tool_calls(expected, (matched, extra))

    assert result.passed is True
    assert result.unmatched_events == (extra,)


def test_subset_matching_ignores_extra_actual_args() -> None:
    """Authored expected keys are compared; extra actual keys (optional params) do not fail."""
    expected = (ExpectedToolCall("get_history", {"entity_ids": ["light.utility_room_ceiling"]}),)
    actual = _event(
        "get_history",
        {"entity_ids": ["light.utility_room_ceiling"], "hours": 6, "limit": 100},
    )

    result = score_tool_calls(expected, (actual,))

    assert result.passed is True
    assert result.reason == "tool_calls_matched"


def test_failed_events_are_excluded_from_matching_and_extras() -> None:
    failed = _event("get_history", {}, output={"status": "error"})

    result = score_tool_calls((ExpectedToolCall("get_history"),), (failed,))

    assert result.reason == "tool_calls_missing"
    assert result.unmatched_events == ()


def test_no_authored_expected_calls_is_not_a_passing_contract() -> None:
    result = score_tool_calls((), (_event("get_history", {}),))

    assert result.passed is False
    assert result.reason == "tool_calls_no_events"


def test_tool_call_oracle_is_primary_and_retains_effect_diagnostics() -> None:
    expected = ExpectedToolCall("get_history", {"entity_ids": ["light.a"]})
    case = EvalCase(
        "tool-case",
        "home_minimal",
        "tool_contract",
        (RequestVariant("canonical", "Read history"),),
        (),
        oracle="tool_calls",
        expected_tool_calls=(expected,),
    )

    evaluation = evaluate_case(
        case,
        (),
        overlay_seeds=(),
        invoker_calls=(),
        tool_events=(_event("get_history", {"entity_ids": ["light.a"]}),),
        answer="irrelevant",
    )

    assert evaluation.outcome.scoring_mode == "tool_calls"
    assert evaluation.outcome.state == "correct"
    assert evaluation.tool_call_result is not None
    assert evaluation.answer_result is None
    assert evaluation.action_result.passed is True
    assert evaluation.end_state_result.status == "not_authored"
