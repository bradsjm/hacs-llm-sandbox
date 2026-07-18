import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, fields
import json
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _diagnostics, _usage, run_case
from llm_sandbox_evals.phases import PhaseObservation
from llm_sandbox_evals.presentation import effective_cause
from llm_sandbox_evals.prompts import load_candidates
from llm_sandbox_evals.schema import CaseTrace, ToolEvent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
import pytest

# Keep the boundary explicit because this test package shares the runtime package name.
# isort: split


_CEREBRAS_RATE_LIMIT_BODY = {
    "message": "Tokens per minute limit exceeded - too many tokens processed.",
    "type": "too_many_tokens_error",
    "param": "quota",
    "code": "token_quota_exceeded",
}

_NESTED_QUOTA_BODY = {
    "message": "request rejected",
    "type": "api_error",
    "code": "generic_provider_error",
    "error": {
        "message": "token throughput exceeded",
        "type": "api_error",
        "code": "token_quota_exceeded",
    },
}


@dataclass
class _UsageDouble:
    requests: int | None
    input_tokens: int | None
    output_tokens: int | None
    details: dict[str, object]


@dataclass
class _ResultDouble:
    usage_data: _UsageDouble | None

    @property
    def usage(self) -> _UsageDouble | None:
        # Mirror the real AgentRunResult.usage property shape (pydantic-ai 2.5).
        return self.usage_data


@pytest.mark.parametrize(
    ("usage_data", "expected"),
    [
        pytest.param(
            _UsageDouble(1, 10, 5, {}),
            {
                "requests": 1,
                "request_tokens": 10,
                "response_tokens": 5,
                "total_tokens": 15,
            },
            id="full-usage",
        ),
        pytest.param(
            _UsageDouble(1, 10, None, {}),
            {
                "requests": 1,
                "request_tokens": 10,
                "response_tokens": None,
                "total_tokens": None,
            },
            id="partial-token-components",
        ),
        pytest.param(None, None, id="missing-usage-property"),
    ],
)
def test_usage_reads_property_and_preserves_token_components(
    usage_data: _UsageDouble | None, expected: dict[str, object] | None
) -> None:
    result = _ResultDouble(usage_data)
    assert _usage(result) == expected




def test_parallel_batches_count_unique_model_responses() -> None:
    events = (
        ToolEvent("get_history", {}, {}, turn_index=2, batch_index=0, batch_size=2),
        ToolEvent("get_statistics", {}, {}, turn_index=2, batch_index=1, batch_size=2),
    )

    diagnostics = _diagnostics(events, (), elapsed=0.0, usage=None, failure=None)

    assert diagnostics.parallel_batches == 1
    assert diagnostics.max_batch_size == 2


def test_diagnostics_reject_unfinished_empty_return_but_accept_valid_empty_recorder_result() -> None:
    events = (
        ToolEvent("execute_home_code", {}, {}),
        ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a"]}, "entries": []}),
    )

    diagnostics = _diagnostics(events, (), elapsed=0.0, usage=None, failure=None)

    assert diagnostics.successful_tool_calls == 1
    assert diagnostics.failed_tool_calls == 1


async def test_run_case_records_provider_failure_as_incomplete_with_no_action_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    config = EvalConfig(
        models=["bad-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=_failing_stream, model_name="bad-model")

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    trace = await run_case(candidate, "bad-model", CASES[0], CASES[0].requests[0], config, profile=profile)

    assert trace.outcome.state == "incomplete"
    assert trace.outcome.scoring_mode is None
    assert trace.outcome.score_reason is None
    assert trace.outcome.score == 0.0
    assert trace.provider_error is not None
    assert trace.diagnostics.failure == "provider_error"
    # Operational failures resolve to their real cause, never a scored action reason.
    assert effective_cause(trace) == "provider_error"
    assert trace.reasoning_effort == config.reasoning_effort
    assert trace.temperature == config.temperature


async def test_run_case_captures_top_level_cerebras_rate_limit_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_name = "cerebras/llama-3.3-70b"
    trace = await _run_case_with_model(
        monkeypatch,
        tmp_path,
        model_name=model_name,
        model=FunctionModel(stream_function=_cerebras_rate_limit_stream, model_name=model_name),
    )

    _assert_cerebras_rate_limit_trace(trace)


async def test_run_case_finds_cerebras_rate_limit_through_exception_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_name = "cerebras/llama-3.3-70b"
    trace = await _run_case_with_model(
        monkeypatch,
        tmp_path,
        model_name=model_name,
        model=FunctionModel(stream_function=_wrapped_cerebras_rate_limit_stream, model_name=model_name),
    )

    _assert_cerebras_rate_limit_trace(trace)
    assert trace.provider_error is not None
    assert "gateway request failed" in trace.provider_error


async def test_run_case_uses_nested_quota_code_over_generic_top_level_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_name = "provider/model-v1"
    trace = await _run_case_with_model(
        monkeypatch,
        tmp_path,
        model_name=model_name,
        model=FunctionModel(stream_function=_nested_quota_stream, model_name=model_name),
    )

    assert trace.outcome.state == "incomplete"
    assert trace.outcome.scoring_mode is None
    assert trace.outcome.score_reason is None
    assert trace.diagnostics.failure == "rate_limit"
    assert trace.execution_error is not None
    assert trace.execution_error.exception_type == "ModelHTTPError"
    assert trace.execution_error.status_code == 400
    assert trace.execution_error.provider_code == "token_quota_exceeded"
    assert trace.execution_error.provider_model == model_name
    assert trace.execution_error.provider_detail is not None
    assert json.loads(trace.execution_error.provider_detail) == _NESTED_QUOTA_BODY


async def test_run_case_keeps_root_model_protocol_error_ahead_of_wrapped_http_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_name = "provider/model-v1"
    trace = await _run_case_with_model(
        monkeypatch,
        tmp_path,
        model_name=model_name,
        model=FunctionModel(stream_function=_model_protocol_error_stream, model_name=model_name),
    )

    assert trace.outcome.state == "incomplete"
    assert trace.outcome.scoring_mode is None
    assert trace.outcome.score_reason is None
    assert trace.diagnostics.failure == "model_protocol_error"
    assert trace.execution_error is not None
    assert trace.execution_error.exception_type == "UnexpectedModelBehavior"
    assert trace.execution_error.status_code is None
    assert trace.execution_error.provider_code is None
    assert trace.provider_error is not None
    assert "ModelHTTPError" in trace.provider_error
    assert "UnexpectedModelBehavior" in trace.provider_error


async def test_run_case_keeps_non_rate_limit_http_error_as_provider_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_name = "provider/model-v1"
    trace = await _run_case_with_model(
        monkeypatch,
        tmp_path,
        model_name=model_name,
        model=FunctionModel(stream_function=_provider_http_error_stream, model_name=model_name),
    )

    assert trace.outcome.state == "incomplete"
    assert trace.outcome.scoring_mode is None
    assert trace.outcome.score_reason is None
    assert trace.diagnostics.failure == "provider_error"
    assert trace.execution_error is not None
    assert trace.execution_error.exception_type == "ModelHTTPError"
    assert trace.execution_error.status_code == 503
    assert trace.execution_error.provider_code == "upstream_unavailable"
    assert trace.execution_error.provider_model == "provider/model-v1"
    assert trace.execution_error.provider_detail is not None
    assert json.loads(trace.execution_error.provider_detail) == {
        "error": {
            "code": "upstream_unavailable",
            "message": "provider unavailable",
            "type": "api_error",
        }
    }
    assert trace.provider_error is not None
    assert "upstream_unavailable" in trace.provider_error


async def test_run_case_keeps_normal_completion_outside_failure_classification(tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    config = EvalConfig(
        models=["stub"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )
    trace = await run_case(
        candidate,
        "stub",
        CASES[0],
        CASES[0].requests[0],
        config,
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )

    assert trace.answer is not None
    assert trace.outcome.state in {"correct", "incorrect"}
    assert trace.outcome.score_reason is not None
    assert trace.diagnostics.failure is None
    assert trace.provider_error is None
    assert trace.execution_error is None
    assert effective_cause(trace) == trace.outcome.score_reason


async def test_run_case_cap_exhausted_is_scored_with_real_action_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    config = EvalConfig(
        models=["looping-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
        max_tool_calls=2,
    )

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=_looping_stream, model_name="looping-model")

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    case = next(case for case in CASES if case.id == "direct_turn_off_utility_room_accent")
    trace = await run_case(candidate, "looping-model", case, case.requests[0], config, profile=profile)

    # Cap exhaustion is scored incorrect with the real action reason, not forced to action_mismatch.
    assert trace.answer is None
    assert trace.outcome.state == "incorrect"
    assert trace.outcome.scoring_mode == "cap_exhausted"
    assert trace.outcome.score_reason == "cap_exhausted"
    assert trace.outcome.score == 0.0
    assert trace.provider_error is None
    assert trace.execution_error is None
    assert trace.diagnostics.failure == "cap_exhausted"
    assert trace.diagnostics.cap_exhausted is True
    assert effective_cause(trace) == "cap_exhausted"
    assert trace.diagnostics.tool_calls == 3
    assert trace.diagnostics.failed_tool_calls == 1
    assert len(trace.tool_events) == 3
    assert trace.tool_events[-1].output == {}
    assert len(trace.action_ledger.successful) == 2
    assert all(
        action["domain"] == "light" and action["service"] == "turn_off" for action in trace.action_ledger.successful
    )


async def test_run_case_failure_trace_retains_partial_model_response_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Fix #3: a non-stub failure trace retains response-level usage captured before the failure.
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    config = EvalConfig(
        models=["usage-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=_usage_then_fail_stream, model_name="usage-model")

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    trace = await run_case(candidate, "usage-model", CASES[0], CASES[0].requests[0], config, profile=profile)

    assert trace.outcome.state == "incomplete"
    assert trace.diagnostics.failure == "provider_error"
    # The partial usage from the captured streamed ModelResponse is retained on the failure trace.
    assert trace.diagnostics.usage is not None
    assert trace.diagnostics.usage["request_tokens"] is not None
    assert trace.diagnostics.usage["response_tokens"] is not None
    assert trace.diagnostics.usage["total_tokens"] == (
        trace.diagnostics.usage["request_tokens"] + trace.diagnostics.usage["response_tokens"]
    )
    assert trace.diagnostics.usage["partial"] is True


async def test_run_case_observes_native_stream_phases_without_leaking_payloads_or_regressing_tool_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    case = next(case for case in CASES if case.id == "direct_turn_on_utility_room_ceiling")
    observations: list[PhaseObservation] = []

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=_thinking_tool_then_response_stream, model_name="native-stream")

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    trace = await run_case(
        candidate,
        "native-stream",
        case,
        case.requests[0],
        EvalConfig(
            models=["native-stream"],
            candidates=[candidate.id],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
        on_phase=observations.append,
    )

    assert trace.provider_error is None
    assert trace.outcome.state == "correct"
    assert [(observation.phase, observation.tool_name) for observation in observations] == [
        ("queued", None),
        ("awaiting_model", None),
        ("thinking", None),
        ("thinking", None),
        ("running_tool", "execute_home_code"),
        ("processing_tool_result", "execute_home_code"),
        ("responding", None),
        ("responding", None),
        ("scoring", None),
        ("finished", None),
    ]
    assert tuple(field.name for field in fields(observations[0])) == ("phase", "tool_name")


async def test_run_case_isolates_phase_observer_exceptions(tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    case = next(case for case in CASES if case.id == "direct_turn_on_utility_room_ceiling")

    trace = await run_case(
        candidate,
        "stub",
        case,
        case.requests[0],
        EvalConfig(
            models=["stub"],
            candidates=[candidate.id],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
        on_phase=_raise_phase_observer_error,
    )

    assert trace.outcome.state == "correct"
    assert trace.answer == "Done."


async def test_run_case_timeout_closes_pending_native_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    stream_started = asyncio.Event()
    stream_closed = asyncio.Event()

    async def pending_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        stream_started.set()
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            stream_closed.set()

    def make_model(_model_id: str) -> FunctionModel:
        return FunctionModel(stream_function=pending_stream, model_name="timeout-model")

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    trace = await run_case(
        candidate,
        "timeout-model",
        CASES[0],
        CASES[0].requests[0],
        EvalConfig(
            models=["timeout-model"],
            candidates=[candidate.id],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
            model_timeout=0.01,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )

    assert trace.outcome.state == "incomplete"
    assert trace.diagnostics.failure == "timeout"
    assert trace.execution_error is not None
    assert "after=0.01s" in trace.execution_error.message
    assert trace.provider_error is not None
    assert "after=0.01s" in trace.provider_error
    assert stream_started.is_set()
    await asyncio.wait_for(stream_closed.wait(), timeout=0.1)


async def _failing_stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    if messages:
        raise RuntimeError("provider rejected model")
    yield "unreachable"


async def _cerebras_rate_limit_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    raise ModelHTTPError(429, "cerebras/llama-3.3-70b", _CEREBRAS_RATE_LIMIT_BODY)
    yield "unreachable"


async def _wrapped_cerebras_rate_limit_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    try:
        raise ModelHTTPError(429, "cerebras/llama-3.3-70b", _CEREBRAS_RATE_LIMIT_BODY)
    except ModelHTTPError as err:
        raise RuntimeError("gateway request failed") from err
    yield "unreachable"


async def _nested_quota_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    raise ModelHTTPError(400, "provider/model-v1", _NESTED_QUOTA_BODY)
    yield "unreachable"


async def _model_protocol_error_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    raise UnexpectedModelBehavior("invalid provider response") from ModelHTTPError(
        429, "provider/model-v1", _CEREBRAS_RATE_LIMIT_BODY
    )
    yield "unreachable"


async def _provider_http_error_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    raise ModelHTTPError(
        503,
        "provider/model-v1",
        {
            "error": {
                "message": "provider unavailable",
                "type": "api_error",
                "code": "upstream_unavailable",
            }
        },
    )
    yield "unreachable"


async def _usage_then_fail_stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
    # First turn: emit a tool call carrying response-level usage. Second turn: fail.
    if len(messages) <= 1:
        yield _execute_home_code_delta(
            "usage-1",
            'await hass.services.async_call("light", "turn_on", target={"entity_id": "light.utility_room_ceiling"})\nresult = "done"',
        )
        return
    raise RuntimeError("provider failed mid-run")


async def _looping_stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
    yield _execute_home_code_delta(
        f"loop-{len(messages)}",
        'await hass.services.async_call("light", "turn_off", target={"entity_id": "light.utility_room_accent"})\nresult = "done"',
    )


async def _thinking_tool_then_response_stream(
    messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls | dict[int, DeltaThinkingPart]]:
    if any(
        isinstance(part, ToolCallPart)
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    ):
        yield "Done."
        return
    yield {0: DeltaThinkingPart(content="private analysis")}
    yield _execute_home_code_delta(
        "native-stream-1",
        'await hass.services.async_call("light", "turn_on", target={"entity_id": "light.utility_room_ceiling"})\nresult = "done"',
        part_index=1,
    )


def _execute_home_code_delta(tool_call_id: str, code: str, *, part_index: int = 0) -> DeltaToolCalls:
    return {
        part_index: DeltaToolCall(
            name="execute_home_code",
            json_args=ToolCallPart(
                tool_name="execute_home_code", args={"code": code}, tool_call_id=tool_call_id
            ).args_as_json_str(),
            tool_call_id=tool_call_id,
        )
    }


def _raise_phase_observer_error(_observation: PhaseObservation) -> None:
    raise RuntimeError("observer unavailable")


async def _run_case_with_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    model_name: str,
    model: FunctionModel,
) -> CaseTrace:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]

    def make_model(_model_id: str) -> FunctionModel:
        return model

    monkeypatch.setattr("llm_sandbox_evals.agent_runner.make_model", make_model)
    return await run_case(
        candidate,
        model_name,
        CASES[0],
        CASES[0].requests[0],
        EvalConfig(
            models=[model_name],
            candidates=[candidate.id],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )


def _assert_cerebras_rate_limit_trace(trace: CaseTrace) -> None:
    assert trace.outcome.state == "incomplete"
    assert trace.outcome.scoring_mode is None
    assert trace.outcome.score_reason is None
    assert trace.diagnostics.failure == "rate_limit"
    assert trace.execution_error is not None
    assert trace.execution_error.exception_type == "ModelHTTPError"
    assert trace.execution_error.status_code == 429
    assert trace.execution_error.provider_code == "token_quota_exceeded"
    assert trace.execution_error.provider_model == "cerebras/llama-3.3-70b"
    assert trace.execution_error.provider_detail is not None
    assert json.loads(trace.execution_error.provider_detail) == _CEREBRAS_RATE_LIMIT_BODY
    assert trace.provider_error is not None
    assert "Traceback (most recent call last)" in trace.provider_error
    assert all(
        key in trace.provider_error and value in trace.provider_error
        for key, value in _CEREBRAS_RATE_LIMIT_BODY.items()
    )
