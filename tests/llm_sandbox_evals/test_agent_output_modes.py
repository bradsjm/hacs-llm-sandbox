import json

from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.agent_runner import build_agent
from llm_sandbox_evals.config import EvalOutputMode
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import EvalRuntime, build_eval_runtime
from llm_sandbox_evals.schema import BlockedOutcome, CaseContext, EvalAnswer, EvalCase, Expected, PromptCandidate
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from llm_sandbox_evals import agent_runner


def _runtime() -> EvalRuntime:
    """Build the smallest fixture-backed runtime needed to construct an eval agent."""
    case = EvalCase(
        id="output-mode",
        category="state",
        home="home_default",
        user_request="Return a result.",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(blocked_outcome=BlockedOutcome()),
    )
    candidate = PromptCandidate("test", "test prompt", "execute", "history", "statistics", "logbook", "automation")
    fixture = get_home(case.home)
    return build_eval_runtime(
        case,
        candidate,
        resolve_profile("balanced"),
        apply_scope(fixture.snapshot(), EVAL_SCOPE),
        fixture,
    )


async def _request_output_mode(
    runtime: EvalRuntime, monkeypatch: pytest.MonkeyPatch, *, output_mode: EvalOutputMode = "tool"
) -> str:
    """Run one final result through an agent and return the protocol sent to the model."""
    requested_modes: list[str] = []

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requested_modes.append(info.model_request_parameters.output_mode)
        payload = EvalAnswer(answer="done").model_dump(mode="json")
        if info.model_request_parameters.output_mode == "native":
            return ModelResponse(parts=[TextPart(content=json.dumps(payload))])
        output_tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=output_tool.name, args=payload, tool_call_id="output")])

    monkeypatch.setattr(agent_runner, "make_model", lambda _model_id: FunctionModel(respond, model_name="test"))
    agent = build_agent(runtime, "test", output_mode)
    result = await agent.run(runtime.case.user_request, deps=runtime)

    assert result.output == EvalAnswer(answer="done")
    return requested_modes[0]


async def test_build_agent_defaults_to_tool_output_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default agent request uses the structured output tool protocol."""
    assert await _request_output_mode(_runtime(), monkeypatch) == "tool"


async def test_build_agent_uses_native_output_mode_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The JSON-schema CLI mode uses provider-native structured output requests."""
    assert await _request_output_mode(_runtime(), monkeypatch, output_mode="json-schema") == "native"
