from dataclasses import dataclass
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _diagnostics, _partial_usage, _usage, run_case
from llm_sandbox_evals.presentation import effective_cause
from llm_sandbox_evals.prompts import load_candidates
from llm_sandbox_evals.schema import ToolEvent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage
import pytest

from llm_sandbox_evals import agent_runner


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
        pytest.param(_UsageDouble(1, 10, 5, {}), {
            "requests": 1,
            "request_tokens": 10,
            "response_tokens": 5,
            "total_tokens": 15,
        }, id="full-usage"),
        pytest.param(_UsageDouble(1, 10, None, {}), {
            "requests": 1,
            "request_tokens": 10,
            "response_tokens": None,
            "total_tokens": None,
        }, id="partial-token-components"),
        pytest.param(None, None, id="missing-usage-property"),
    ],
)
def test_usage_reads_property_and_preserves_token_components(
    usage_data: _UsageDouble | None, expected: dict[str, object] | None
) -> None:
    result = _ResultDouble(usage_data)
    assert _usage(result) == expected


def test_partial_usage_sums_model_response_usage_when_final_is_unavailable() -> None:
    response_a = ModelResponse(
        parts=[ToolCallPart(tool_name="execute_home_code", args={}, tool_call_id="a")],
        usage=RunUsage(input_tokens=12, output_tokens=8),
    )
    response_b = ModelResponse(
        parts=[ToolCallPart(tool_name="execute_home_code", args={}, tool_call_id="b")],
        usage=RunUsage(input_tokens=3, output_tokens=2),
    )

    usage = _partial_usage([response_a, response_b])

    assert usage is not None
    assert usage["request_tokens"] == 15
    assert usage["response_tokens"] == 10
    assert usage["total_tokens"] == 25
    assert usage["partial"] is True


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
    monkeypatch: object, tmp_path: Path
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
        return FunctionModel(_failing_model, model_name="bad-model")

    monkeypatch.setattr(agent_runner, "make_model", make_model)  # type: ignore[attr-defined]
    trace = await run_case(candidate, "bad-model", CASES[0], config, profile=profile)

    assert trace.outcome.state == "incomplete"
    assert trace.outcome.action_reason is None
    assert trace.outcome.score == 0.0
    assert trace.provider_error is not None
    assert trace.diagnostics.failure == "provider_error"
    # Operational failures resolve to their real cause, never a scored action reason.
    assert effective_cause(trace) == "provider_error"
    assert trace.reasoning_effort == config.reasoning_effort
    assert trace.temperature == config.temperature


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
    trace = await run_case(candidate, "stub", CASES[0], config, profile=resolve_profile(DEFAULT_PROMPT_PROFILE))

    assert trace.answer is not None
    assert trace.outcome.state in {"correct", "incorrect"}
    assert trace.outcome.action_reason is not None
    assert trace.diagnostics.failure is None
    assert trace.provider_error is None
    assert effective_cause(trace) == trace.outcome.action_reason


async def test_run_case_cap_exhausted_is_scored_with_real_action_reason(monkeypatch: object, tmp_path: Path) -> None:
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
        return FunctionModel(_looping_model, model_name="looping-model")

    monkeypatch.setattr(agent_runner, "make_model", make_model)  # type: ignore[attr-defined]
    case = next(case for case in CASES if case.id == "direct_turn_off_utility_room_accent")
    trace = await run_case(candidate, "looping-model", case, config, profile=profile)

    # Cap exhaustion is scored incorrect with the real action reason, not forced to action_mismatch.
    assert trace.answer is None
    assert trace.outcome.state == "incorrect"
    assert trace.outcome.action_reason is not None
    assert trace.outcome.action_reason != "action_mismatch"
    assert trace.outcome.score == 0.0
    assert trace.provider_error is None
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
    monkeypatch: object, tmp_path: Path
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
        return FunctionModel(_usage_then_fail_model, model_name="usage-model")

    monkeypatch.setattr(agent_runner, "make_model", make_model)  # type: ignore[attr-defined]
    trace = await run_case(candidate, "usage-model", CASES[0], config, profile=profile)

    assert trace.outcome.state == "incomplete"
    assert trace.diagnostics.failure == "provider_error"
    # The partial usage from the captured ModelResponse is retained on the failure trace.
    assert trace.diagnostics.usage is not None
    assert trace.diagnostics.usage["request_tokens"] == 10
    assert trace.diagnostics.usage["response_tokens"] == 5
    assert trace.diagnostics.usage["total_tokens"] == 15
    assert trace.diagnostics.usage["partial"] is True


async def _failing_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    raise RuntimeError("provider rejected model")


async def _usage_then_fail_model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    # First turn: emit a tool call carrying response-level usage. Second turn: fail.
    if len(messages) <= 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="execute_home_code",
                    args={
                        "code": 'await hass.services.async_call("light", "turn_on", target={"entity_id": "light.utility_room_ceiling"})\nresult = "done"'
                    },
                    tool_call_id="usage-1",
                )
            ],
            usage=RunUsage(input_tokens=10, output_tokens=5),
        )
    raise RuntimeError("provider failed mid-run")


async def _looping_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="execute_home_code",
                args={
                    "code": 'await hass.services.async_call("light", "turn_off", target={"entity_id": "light.utility_room_accent"})\nresult = "done"'
                },
                tool_call_id=f"loop-{len(_messages)}",
            )
        ]
    )
