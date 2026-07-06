from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import run_case
from llm_sandbox_evals.models import ModelResponseError
from llm_sandbox_evals.prompts import load_candidates
from llm_sandbox_evals.schema import AgentStep, ToolCall


class FailingAdapter:
    def __init__(self, *, detail: str = "provider rejected model") -> None:
        self.calls = 0
        self.detail = detail

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> AgentStep:
        _ = (model_id, messages, tools)
        self.calls += 1
        raise ModelResponseError("provider rejected model", detail=self.detail)


class LoopingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> AgentStep:
        _ = (model_id, messages, tools)
        self.calls += 1
        tool_call = ToolCall(
            id=f"loop-{self.calls}",
            tool_name="execute_home_code",
            tool_args={"code": "result = 'ok'"},
        )
        return AgentStep(
            tool_calls=(tool_call,),
            text="",
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {"name": tool_call.tool_name, "arguments": '{"code": "result = \'ok\'"}'},
                    }
                ],
            },
            raw=f"loop raw {self.calls}",
        )


async def test_run_case_records_model_error(tmp_path: Path) -> None:
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

    trace = await run_case(candidate, "bad-model", CASES[0], FailingAdapter(), profile, config)

    assert trace.score == 0.0
    assert trace.raw_output == "provider rejected model"
    assert [(check.name, check.passed, check.required) for check in trace.checks] == [("model_error", False, True)]


async def test_run_case_does_not_force_final_answer_after_max_turns(tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    adapter = LoopingAdapter()
    config = EvalConfig(
        models=["looping-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
        max_turns=2,
    )

    trace = await run_case(candidate, "looping-model", CASES[0], adapter, profile, config)

    assert adapter.calls == 2
    assert trace.turns == 2
    assert trace.final_answer == ""
    assert trace.raw_output == "loop raw 2"
    assert trace.score == 0.0
    assert ("max_turns_exceeded", False, True) in [
        (check.name, check.passed, check.required) for check in trace.checks
    ]
