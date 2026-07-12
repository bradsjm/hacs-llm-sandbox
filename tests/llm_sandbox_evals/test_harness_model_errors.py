from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import run_case
from llm_sandbox_evals.prompts import load_candidates
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llm_sandbox_evals import agent_runner


async def test_run_case_records_model_error(monkeypatch: object, tmp_path: Path) -> None:
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
    assert trace.outcome.score == 0.0
    assert trace.provider_error is not None
    assert trace.diagnostics.failure == "provider_error"


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
    assert trace.outcome.reason
    assert trace.diagnostics.failure is None
    assert trace.provider_error is None


async def test_run_case_does_not_force_final_answer_after_max_tool_calls(monkeypatch: object, tmp_path: Path) -> None:
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
    case = next(case for case in CASES if case.id == "action_turn_off_living_light")
    trace = await run_case(candidate, "looping-model", case, config, profile=profile)

    assert trace.answer is None
    assert trace.outcome.state == "incorrect"
    assert trace.outcome.score == 0.0
    assert trace.provider_error is None
    assert trace.diagnostics.failure == "cap_exhausted"
    assert trace.diagnostics.cap_exhausted is True
    assert trace.diagnostics.tool_calls == 3
    assert len(trace.tool_events) == 3
    assert trace.tool_events[-1].output == {}
    assert len(trace.action_ledger.successful) == 2
    assert all(
        action["domain"] == "light" and action["service"] == "turn_off"
        for action in trace.action_ledger.successful
    )


async def _failing_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    raise RuntimeError("provider rejected model")


async def _looping_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="execute_home_code",
                args={
                    "code": 'await hass.services.async_call("light", "turn_off", target={"entity_id": "light.living"})\nresult = "done"'
                },
                tool_call_id=f"loop-{len(_messages)}",
            )
        ]
    )
