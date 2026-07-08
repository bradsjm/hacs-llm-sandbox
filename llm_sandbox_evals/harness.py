"""Eval task body backed by a real Pydantic AI Agent."""

import asyncio
from collections.abc import Sequence

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from llm_sandbox_evals import cases
from llm_sandbox_evals.agent_runner import build_agent, reasoning_model_settings
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import build_eval_runtime
from llm_sandbox_evals.schema import CaseTrace, CheckResult, EvalCase, PromptCandidate
from llm_sandbox_evals.scoring import check_case, score_case
from llm_sandbox_evals.tools import EVAL_SCOPE, _for_scoring, apply_scope


async def run_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    config: EvalConfig,
    *,
    profile: PromptProfile,
) -> CaseTrace:
    """Run one matrix cell through the production-core Pydantic AI agent."""
    try:
        fixture = get_home(case.home)
        snapshot = apply_scope(fixture.snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
        runtime = build_eval_runtime(case, candidate, profile, snapshot, fixture)
        agent = build_agent(runtime, model_id)
        tool_calls_limit = case.expected.max_tool_calls or config.max_tool_calls
        result = await asyncio.wait_for(
            agent.run(
                case.user_request,
                deps=runtime,
                model_settings=reasoning_model_settings(model_id, config.reasoning_effort),
                usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit),
            ),
            timeout=config.model_timeout,
        )
        output = result.output
        tool_call_count = _tool_call_count(result.all_messages())
        recorded_actions = tuple(_for_scoring(action) for action in runtime.invoker.calls)
        checks = check_case(case, output, recorded_actions, tool_call_count)
        return CaseTrace(
            case_id=case.id,
            category=case.category,
            candidate_id=candidate.id,
            model_id=model_id,
            score=score_case(checks),
            output=output,
            tool_call_count=tool_call_count,
            recorded_actions=recorded_actions,
            checks=tuple(checks),
            error=None,
        )
    except UsageLimitExceeded as err:
        return _error_trace(candidate, model_id, case, "tool_calls_exceeded", f"{type(err).__name__}: {err}")
    except (TimeoutError, UnexpectedModelBehavior) as err:
        return _error_trace(candidate, model_id, case, "model_error", f"{type(err).__name__}: {err}")
    except Exception as err:  # noqa: BLE001 - provider/harness failures are isolated to the current matrix cell.
        return _error_trace(candidate, model_id, case, "model_error", f"{type(err).__name__}: {err}")


def _error_trace(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    check_name: str,
    feedback: str,
) -> CaseTrace:
    """Return a zero-score trace for an infrastructure, provider, or limit failure."""
    return CaseTrace(
        case_id=case.id,
        category=case.category,
        candidate_id=candidate.id,
        model_id=model_id,
        score=0.0,
        output="",
        tool_call_count=case.expected.max_tool_calls,
        recorded_actions=(),
        checks=(
            CheckResult(
                name=check_name,
                passed=False,
                required=True,
                feedback=feedback,
            ),
        ),
        error=f"{check_name}: {feedback}",
    )


def _select_cases(case_filters: list[str] | None, home_filters: list[str] | None) -> list[EvalCase]:
    """Select cases by id/category and optional home name, preserving CASES order."""
    selected = cases.CASES
    if home_filters is not None:
        home_names = set(home_filters)
        selected = [case for case in selected if case.home in home_names]

    # Branch boundary: no case/category filter means all remaining cases are selected.
    if case_filters is None:
        return list(selected)

    requested = set(case_filters)
    return [case for case in selected if case.id in requested or case.category in requested]


def _tool_call_count(messages: Sequence[object]) -> int:
    """Count native Pydantic AI tool-call parts in the full conversation."""
    return sum(
        1
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )
