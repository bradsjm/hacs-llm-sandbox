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

    assert trace.score == 0.0
    assert trace.error is not None
    assert [(check.name, check.passed, check.required) for check in trace.checks] == [("model_error", False, True)]


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
    trace = await run_case(candidate, "looping-model", CASES[0], config, profile=profile)

    assert trace.output == ""
    assert trace.score == 0.0
    assert trace.error is not None
    assert [(check.name, check.passed, check.required) for check in trace.checks] == [
        ("tool_calls_exceeded", False, True)
    ]


async def _failing_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    raise RuntimeError("provider rejected model")


async def _looping_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name="execute_home_code", args={"code": "result = 'ok'"}, tool_call_id="loop")]
    )
