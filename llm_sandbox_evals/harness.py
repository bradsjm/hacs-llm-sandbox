"""Eval task body backed by a real Pydantic AI Agent."""

import asyncio
import json
from collections.abc import Sequence

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits

from llm_sandbox_evals import cases
from llm_sandbox_evals.agent_runner import build_agent, build_model_settings
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import build_eval_runtime
from llm_sandbox_evals.schema import CaseTrace, CheckResult, EvalCase, PromptCandidate, ToolEvent
from llm_sandbox_evals.scoring import check_case, score_case
from llm_sandbox_evals.tools import EVAL_SCOPE, _for_scoring, apply_scope

_TOOL_EXECUTE_HOME_CODE = "execute_home_code"


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
                model_settings=build_model_settings(
                    model_id,
                    temperature=config.temperature,
                    reasoning_effort=config.reasoning_effort,
                ),
                usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit),
            ),
            timeout=config.model_timeout,
        )
        output = result.output
        messages = result.all_messages()
        tool_call_count = _tool_call_count(messages)
        tool_events = _tool_events(messages)
        recorded_actions = _recorded_actions_from_tool_events(tool_events)
        checks = check_case(case, output, recorded_actions, tool_call_count, tool_events)
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
            tool_events=tool_events,
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
        tool_events=(),
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


def _tool_events(messages: Sequence[object]) -> tuple[ToolEvent, ...]:
    """Pair each tool call with its return payload, preserving call order.

    Tool-call arguments are captured for traceability but are NOT used as
    evidence by scoring. Tool returns (``ToolReturnPart.content``) are the
    production result envelopes and feed the any-source evidence audit and the
    ``execution_ok`` gate.
    """
    returns_by_id: dict[str, object] = {}
    calls: list[ToolCallPart] = []
    for message in messages:
        # Returns arrive in ModelRequest parts (tool-result messages).
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    returns_by_id[part.tool_call_id] = part.content
        elif isinstance(message, ModelResponse):
            calls.extend(part for part in message.parts if isinstance(part, ToolCallPart))

    events: list[ToolEvent] = []
    for call in calls:
        output = returns_by_id.get(call.tool_call_id)
        events.append(
            ToolEvent(
                tool_name=call.tool_name,
                args=_coerce_args(call.args),
                output=_coerce_return(output),
            )
        )
    return tuple(events)


def _recorded_actions_from_tool_events(tool_events: tuple[ToolEvent, ...]) -> tuple[dict[str, object], ...]:
    """Return all execute_home_code action records, including blocked/error actions."""
    actions: list[dict[str, object]] = []
    for event in tool_events:
        # Branch boundary: only execute_home_code result envelopes carry action records.
        if event.tool_name != _TOOL_EXECUTE_HOME_CODE:
            continue
        raw_actions = event.output.get("actions")
        if not isinstance(raw_actions, list):
            continue
        for action in raw_actions:
            # Safety constraint: scorer input is copied from JSON-safe tool output,
            # never from the live/recording invoker seam.
            if isinstance(action, dict):
                actions.append(_for_scoring(action))
    return tuple(actions)


def _coerce_args(args: object) -> dict[str, object]:
    """Normalize a ToolCallPart args value into a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
        return decoded if isinstance(decoded, dict) else {"_raw": decoded}
    return {}


def _coerce_return(content: object) -> dict[str, object]:
    """Normalize a ToolReturnPart content value into a dict envelope."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return {"_raw": content}
        return decoded if isinstance(decoded, dict) else {"_raw": decoded}
    # Branch boundary: production tools always return dict envelopes, but guard
    # any non-dict scalar so the trace stays JSON-serializable.
    return {} if content is None else {"_raw": content}
