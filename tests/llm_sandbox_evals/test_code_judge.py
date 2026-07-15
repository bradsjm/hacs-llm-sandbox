import asyncio
from collections.abc import Callable

from custom_components.llm_sandbox.llm_api.executor import MAX_MONTY_CODE_CHARS
from llm_sandbox_evals.code_judge import CodeQualityJudge
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalDiagnostics,
    ToolEvent,
)
from pydantic_evals.evaluators import EvaluationReason
import pytest

_MODEL = "openai-chat:gpt-5.4"


async def test_code_judge_submits_complete_ordered_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_calls: list[tuple[object, object, str, dict[str, object]]] = []

    async def judge_input_output(
        request: object, output: object, rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model_settings is None
        helper_calls.append((request, output, rubric, {"model": model}))
        return _GradingOutput(reason="clear code", pass_=True, score=0.9)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)
    events = (
        _execute("second()", "ok", [], 3),
        ToolEvent(
            "get_history",
            {"entity_ids": ["light.kitchen"], "start_time": "2026-07-15T12:00:00Z"},
            {"status": "ok", "rows": [{"state": "on"}]},
            call_index=1,
        ),
        ToolEvent(
            "execute_home_code",
            {"code": "first()"},
            {
                "execution": {"status": "ok"},
                "output": {"selected": ["light.kitchen"]},
                "actions": [
                    {
                        "status": "success",
                        "domain": "light",
                        "service": "turn_on",
                        "target_entity_ids": ["light.kitchen"],
                    }
                ],
                "resolutions": [{"requested": "kitchen light", "entity_id": "light.kitchen"}],
                "notes": ["snapshot remains frozen"],
            },
            call_index=2,
        ),
    )

    await CodeQualityJudge(_MODEL, 0.1).evaluate(_context(_trace(events)))

    assert len(helper_calls) == 1
    judge_input, judge_output, rubric, model = helper_calls[0]
    assert judge_input == {
        "request_text": "Turn on the kitchen light.",
        "deterministic_outcome": {
            "state": "correct",
            "scoring_mode": "actions",
            "score_reason": "ok",
        },
    }
    assert isinstance(judge_output, dict)
    submissions = judge_output["execute_home_code"]
    assert isinstance(submissions, list)
    assert [submission["call_index"] for submission in submissions] == [2, 3]
    assert [submission["source"] for submission in submissions] == ["first()", "second()"]
    assert submissions[0]["output"] == {
        "value": {"selected": ["light.kitchen"]},
        "serialized_chars": 30,
        "item_count": 1,
        "truncated": False,
    }
    assert submissions[0]["resolutions"]["value"] == [
        {"requested": "kitchen light", "entity_id": "light.kitchen"}
    ]
    assert submissions[0]["notes"]["value"] == ["snapshot remains frozen"]
    assert submissions[0]["actions"][0]["target_entity_ids"] == ["light.kitchen"]
    other_tools = judge_output["other_tool_calls"]
    assert isinstance(other_tools, list)
    assert other_tools == [
        {
            "call_index": 1,
            "tool_name": "get_history",
            "args": {
                "value": {
                    "entity_ids": ["light.kitchen"],
                    "start_time": "2026-07-15T12:00:00Z",
                },
                "serialized_chars": 68,
                "item_count": 2,
                "truncated": False,
            },
            "status": "ok",
        }
    ]
    assert "ephemeral glue code" in rubric
    assert model == {"model": _MODEL}


async def test_code_judge_returns_native_score_and_pass_results_with_the_provider_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        return _GradingOutput(reason="the code is concise", pass_=True, score=0.75)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)

    results = await CodeQualityJudge(_MODEL, 0.1).evaluate(_context(_trace(())))
    score = results["code_quality_score"]
    passed = results["code_quality_pass"]

    assert type(score).__name__ == "EvaluationReason"
    assert type(passed).__name__ == "EvaluationReason"
    assert (score.value, score.reason) == (0.75, "the code is concise")
    assert (passed.value, passed.reason) == (True, "the code is concise")


async def test_code_judge_bounds_one_helper_attempt_with_the_configured_model_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def pending_judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        nonlocal calls
        assert model == _MODEL
        assert model_settings is None
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", pending_judge_input_output)

    with pytest.raises(TimeoutError):
        await CodeQualityJudge(_MODEL, 0.01).evaluate(_context(_trace(())))

    assert calls == 1


async def test_code_judge_observes_start_before_the_helper_and_terminal_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []

    async def judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        timeline.append("helper")
        return _GradingOutput(reason="clear", pass_=True, score=1.0)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)

    await CodeQualityJudge(
        _MODEL, 0.1, on_judging=lambda: timeline.append("started"), on_terminal=lambda: timeline.append("finished")
    ).evaluate(_context(_trace(())))

    assert timeline == ["started", "helper", "finished"]


async def test_code_judge_observes_terminal_after_an_ordinary_helper_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await CodeQualityJudge(
            _MODEL, 0.1, on_judging=lambda: events.append("started"), on_terminal=lambda: events.append("finished")
        ).evaluate(_context(_trace(())))

    assert events == ["started", "finished"]


async def test_code_judge_isolates_observer_exceptions_without_suppressing_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        return _GradingOutput(reason="clear", pass_=True, score=1.0)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)

    results = await CodeQualityJudge(
        _MODEL, 0.1, on_judging=_raise_observer_error, on_terminal=_raise_observer_error
    ).evaluate(_context(_trace(())))

    passed = results["code_quality_pass"]
    assert type(passed).__name__ == "EvaluationReason"
    assert (passed.value, passed.reason) == (True, "clear")


async def test_code_judge_propagates_cancellation_without_a_terminal_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_started = asyncio.Event()
    events: list[str] = []

    async def pending_judge_input_output(
        _request: object, _submissions: object, _rubric: str, *, model: str, model_settings: object | None
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        helper_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", pending_judge_input_output)
    evaluation = asyncio.create_task(
        CodeQualityJudge(
            _MODEL, 1.0, on_judging=lambda: events.append("started"), on_terminal=lambda: events.append("finished")
        ).evaluate(_context(_trace(())))
    )
    await helper_started.wait()
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await evaluation

    assert events == ["started"]


@pytest.mark.parametrize(
    "trace_factory",
    [
        pytest.param(
            lambda: _trace((), request_text="r" * (MAX_MONTY_CODE_CHARS + 1)),
            id="oversized-request",
        ),
        pytest.param(
            lambda: _trace((_execute(42, "ok", [], 0),)),
            id="malformed-source",
        ),
        pytest.param(
            lambda: _trace(
                tuple(
                    _execute(str(index) * MAX_MONTY_CODE_CHARS, "ok", [], index)
                    for index in range(3)
                )
            ),
            id="oversized-complete-source-trajectory",
        ),
    ],
)
async def test_code_judge_does_not_call_provider_when_complete_source_context_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    trace_factory: Callable[[], CaseTrace],
) -> None:
    results = await _unavailable_results(monkeypatch, trace_factory())

    assert results == {}


async def test_code_judge_bounds_an_oversized_execution_status_before_the_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = "ok-" + "s" * MAX_MONTY_CODE_CHARS

    _request, judge_output = await _judge_arguments(
        monkeypatch, _trace((_execute("code()", status, [], 0),))
    )
    submissions = judge_output["execute_home_code"]

    execution_status = submissions[0]["execution_status"]
    assert isinstance(execution_status, str)
    assert execution_status != status
    assert execution_status.endswith("[status truncated]")


@pytest.mark.parametrize(
    "actions",
    [
        pytest.param(
            [{"ordinal": index} for index in range(33)],
            id="too-many-actions",
        ),
        pytest.param(
            [{"payload": {"nested": {"tail": "x" * MAX_MONTY_CODE_CHARS}}}],
            id="oversized-action-record",
        ),
    ],
)
async def test_code_judge_does_not_call_provider_when_complete_action_context_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    actions: list[dict[str, object]],
) -> None:
    results = await _unavailable_results(
        monkeypatch,
        _trace((_execute("code()", "ok", actions, 0),)),
    )

    assert results == {}


async def test_code_judge_summarizes_large_output_with_original_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [{"ordinal": index, "state": "on"} for index in range(500)]
    event = ToolEvent(
        "execute_home_code",
        {"code": "result = values"},
        {
            "execution": {"status": "ok"},
            "output": values,
            "actions": [],
            "resolutions": [],
            "notes": [],
        },
    )

    _request, judge_output = await _judge_arguments(monkeypatch, _trace((event,)))
    submissions = judge_output["execute_home_code"]
    output = submissions[0]["output"]

    assert output["truncated"] is True
    assert output["item_count"] == 500
    assert output["serialized_chars"] > MAX_MONTY_CODE_CHARS
    assert output["value"]["kind"] == "sequence"
    assert len(output["value"]["sample"]) == 8


async def _judge_arguments(
    monkeypatch: pytest.MonkeyPatch, trace: CaseTrace
) -> tuple[object, dict[str, list[dict[str, object]]]]:
    calls: list[tuple[object, dict[str, list[dict[str, object]]]]] = []

    async def judge_input_output(
        request: object,
        output: dict[str, list[dict[str, object]]],
        _rubric: str,
        *,
        model: str,
        model_settings: object | None,
    ) -> _GradingOutput:
        assert model == _MODEL
        assert model_settings is None
        calls.append((request, output))
        return _GradingOutput(reason="clear", pass_=True, score=1.0)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)

    await CodeQualityJudge(_MODEL, 0.1).evaluate(_context(trace))

    assert len(calls) == 1
    return calls[0]


async def _unavailable_results(
    monkeypatch: pytest.MonkeyPatch,
    trace: CaseTrace,
) -> dict[str, EvaluationReason]:
    async def judge_input_output(
        _request: object,
        _output: object,
        _rubric: str,
        *,
        model: str,
        model_settings: object | None,
    ) -> _GradingOutput:
        raise AssertionError((model, model_settings))

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge_input_output)
    return await CodeQualityJudge(_MODEL, 0.1).evaluate(_context(trace))


def _execute(code: object, status: object, actions: list[dict[str, object]], call_index: int) -> ToolEvent:
    return ToolEvent(
        "execute_home_code", {"code": code}, {"execution": {"status": status}, "actions": actions}, call_index
    )


def _context(trace: CaseTrace) -> _Context:
    return _Context(trace)


def _trace(tool_events: tuple[ToolEvent, ...], *, request_text: str = "Turn on the kitchen light.") -> CaseTrace:
    return CaseTrace(
        case_id="code-judge",
        candidate_id="baseline",
        model_id="candidate-model",
        request_variant_id="canonical",
        request_text=request_text,
        answer="Done.",
        required_actions=(),
        desired_entities=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome("correct", "actions", "ok"),
        action_result=ActionResult(True, "ok"),
        action_ledger=ActionLedger(),
        tool_events=tool_events,
        diagnostics=EvalDiagnostics(),
        oracle="answer",
    )


def _raise_observer_error() -> None:
    raise RuntimeError("observer unavailable")


class _Context:
    def __init__(self, trace: CaseTrace) -> None:
        self.output = trace


class _GradingOutput:
    def __init__(self, *, reason: str, pass_: bool, score: float) -> None:
        self.reason = reason
        self.pass_ = pass_
        self.score = score
