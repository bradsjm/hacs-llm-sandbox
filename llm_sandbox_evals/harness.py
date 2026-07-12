"""Eval task body backed by a real Pydantic AI Agent."""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import suppress
import json
from time import perf_counter

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits

from llm_sandbox_evals import cases
from llm_sandbox_evals.agent_runner import build_agent, build_model_settings
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import ToolBoundaryCallback, build_eval_runtime
from llm_sandbox_evals.schema import (
    CaseOutcome,
    CaseTrace,
    EvalCase,
    EvalDiagnostics,
    PromptCandidate,
    ToolEvent,
)
from llm_sandbox_evals.scoring import evaluate_case, successful_events
from llm_sandbox_evals.scoring.actions import build_action_ledger
from llm_sandbox_evals.tools import EVAL_SCOPE, _for_scoring, apply_scope

_TOOL_EXECUTE_HOME_CODE = "execute_home_code"
_OUTPUT_TOOL = "final_result"


async def run_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    config: EvalConfig,
    *,
    profile: PromptProfile,
    on_tool_boundary: ToolBoundaryCallback | None = None,
    on_response: Callable[[str], None] | None = None,
) -> CaseTrace:
    """Run one matrix cell through the production-core Pydantic AI agent."""
    started = perf_counter()
    captured: Sequence[object] = ()
    proposed_actions: Sequence[dict[str, object]] = ()
    try:
        fixture = get_home(case.home)
        snapshot = apply_scope(fixture.snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
        runtime = build_eval_runtime(case, candidate, profile, snapshot, fixture, on_tool_boundary=on_tool_boundary)
        proposed_actions = runtime.invoker.calls
        agent = build_agent(runtime, model_id)
        with capture_run_messages() as captured:
            result = await asyncio.wait_for(
                agent.run(
                    case.user_request,
                    deps=runtime,
                    model_settings=build_model_settings(
                        model_id,
                        temperature=config.temperature,
                        reasoning_effort=config.reasoning_effort,
                    ),
                    usage_limits=UsageLimits(tool_calls_limit=config.max_tool_calls),
                ),
                timeout=config.model_timeout,
            )
        output = result.output
        if on_response is not None:
            # Safety constraint: an observer cannot alter a completed model result or scoring.
            with suppress(Exception):
                on_response(output.answer)
        messages = result.all_messages()
        tool_events = _tool_events(messages)
        recorded_actions = _recorded_actions_from_tool_events(tool_events, runtime.invoker.calls)
        conversation_id = _conversation_id(messages)
        outcome, conclusions, action_results = evaluate_case(case, output, tool_events, recorded_actions)
        return CaseTrace(
            case_id=case.id,
            category=case.category,
            candidate_id=candidate.id,
            model_id=model_id,
            answer=output,
            expected=case.expected,
            outcome=outcome,
            conclusions=conclusions,
            actions=action_results,
            action_ledger=build_action_ledger(recorded_actions),
            tool_events=tool_events,
            diagnostics=_diagnostics(
                tool_events,
                messages,
                elapsed=perf_counter() - started,
                usage=_usage(result),
                failure=None,
            ),
            provider_error=None,
            conversation_id=conversation_id,
            user_request=case.user_request,
        )
    except UsageLimitExceeded as err:
        # Branch boundary: the model exhausted its allowed tool calls, which is scored behavior rather than a harness error.
        tool_events = _tool_events(captured)
        return _failure_trace(
            candidate,
            model_id,
            case,
            "cap_exhausted",
            _format_exception(err),
            diagnostic=False,
            elapsed=perf_counter() - started,
            tool_events=tool_events,
            messages=captured,
            recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
            conversation_id=_conversation_id(captured),
        )
    except TimeoutError as err:
        tool_events = _tool_events(captured)
        return _failure_trace(
            candidate,
            model_id,
            case,
            "timeout",
            _format_exception(err, timeout=config.model_timeout),
            diagnostic=True,
            elapsed=perf_counter() - started,
            tool_events=tool_events,
            messages=captured,
            recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
            conversation_id=_conversation_id(captured),
        )
    except UnexpectedModelBehavior as err:
        tool_events = _tool_events(captured)
        return _failure_trace(
            candidate,
            model_id,
            case,
            "model_protocol_error",
            _format_exception(err),
            diagnostic=True,
            elapsed=perf_counter() - started,
            tool_events=tool_events,
            messages=captured,
            recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
            conversation_id=_conversation_id(captured),
        )
    except Exception as err:  # noqa: BLE001 - provider/harness failures are isolated to the current matrix cell.
        tool_events = _tool_events(captured)
        return _failure_trace(
            candidate,
            model_id,
            case,
            "provider_error",
            _format_exception(err),
            diagnostic=True,
            elapsed=perf_counter() - started,
            tool_events=tool_events,
            messages=captured,
            recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
            conversation_id=_conversation_id(captured),
        )


def _failure_trace(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    failure: str,
    feedback: str,
    *,
    diagnostic: bool,
    elapsed: float | None = None,
    tool_events: tuple[ToolEvent, ...] = (),
    messages: Sequence[object] = (),
    recorded_actions: tuple[dict[str, object], ...] = (),
    conversation_id: str | None = None,
) -> CaseTrace:
    """Return an incomplete or cap-exhausted trace with captured diagnostics."""
    return CaseTrace(
        case_id=case.id,
        category=case.category,
        candidate_id=candidate.id,
        model_id=model_id,
        answer=None,
        expected=case.expected,
        outcome=CaseOutcome("incorrect" if failure == "cap_exhausted" else "incomplete", failure),
        conclusions=(),
        actions=(),
        action_ledger=build_action_ledger(recorded_actions),
        tool_events=tool_events,
        diagnostics=_diagnostics(
            tool_events,
            messages,
            elapsed=elapsed,
            usage=None,
            failure=failure,
            cap_exhausted=failure == "cap_exhausted",
        ),
        provider_error=feedback if diagnostic else None,
        conversation_id=conversation_id,
        user_request=case.user_request,
    )


def _format_exception(err: BaseException, *, timeout: float | None = None) -> str:
    """Format provider, timeout, and limit failures as compact one-line feedback."""
    message = _exception_message(err)
    if timeout is not None:
        message = f"{message} after={timeout:g}s"
    formatted = f"{type(err).__name__}: {message}"
    cause_chain = _format_cause_chain(err)
    if cause_chain:
        formatted = f"{formatted} caused_by={cause_chain}"
    return formatted


def _format_cause_chain(err: BaseException) -> str:
    """Return direct exception cause/context entries without tracebacks."""
    seen = {id(err)}
    chain: list[str] = []
    current = err.__cause__ or err.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__name__}: {_exception_message(current)}")
        # Branch boundary: prefer explicit causes, then implicit contexts, matching
        # Python exception chaining while keeping a single readable feedback line.
        current = current.__cause__ or current.__context__
    return " <- ".join(chain)


def _exception_message(err: BaseException) -> str:
    """Return a non-empty one-line exception message."""
    message = " ".join(str(err).split())
    return message or "timed out" if isinstance(err, TimeoutError) else message or "no detail"


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
     successful production evidence.
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
    call_index = 0
    response_index = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            response_calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
            batch_size = len(response_calls)
            for batch_index, call in enumerate(response_calls):
                # The structured final-result tool is an agent protocol message,
                # not a production tool event or evidence source.
                if call.tool_name == _OUTPUT_TOOL:
                    continue
                output = returns_by_id.get(call.tool_call_id)
                events.append(
                    ToolEvent(
                        tool_name=call.tool_name,
                        args=_coerce_args(call.args),
                        output=_coerce_return(output),
                        call_index=call_index,
                        turn_index=response_index,
                        batch_index=batch_index,
                        batch_size=batch_size,
                    )
                )
                call_index += 1
            response_index += 1
    return tuple(events)


def _diagnostics(
    events: Sequence[ToolEvent],
    messages: Sequence[object],
    *,
    elapsed: float | None,
    usage: dict[str, int | float | None] | None,
    failure: str | None,
    cap_exhausted: bool = False,
) -> EvalDiagnostics:
    """Build diagnostics without turning attempts, timing, or usage into score."""
    successful = successful_events(events)
    repairs = 0
    had_execute_error = False
    for event in events:
        if event.tool_name == _TOOL_EXECUTE_HOME_CODE:
            execution = event.output.get("execution")
            status = execution.get("status") if isinstance(execution, dict) else None
            if status in {"code_error", "helper_error"}:
                had_execute_error = True
            elif had_execute_error:
                repairs += 1
    response_turns = [message for message in messages if isinstance(message, ModelResponse)]
    turn_count = len(response_turns) or (max((event.turn_index for event in events), default=-1) + 1)
    batch_sizes = [event.batch_size for event in events]
    return EvalDiagnostics(
        tool_calls=len(events),
        successful_tool_calls=len(successful),
        failed_tool_calls=len(events) - len(successful),
        execute_repairs=repairs,
        model_turns=turn_count,
        parallel_batches=len({event.turn_index for event in events if event.batch_size > 1}),
        max_batch_size=max(batch_sizes, default=1),
        elapsed_seconds=elapsed,
        cap_exhausted=cap_exhausted,
        usage=usage,
        failure=failure,
    )


def _usage(result: object) -> dict[str, int | float | None] | None:
    """Copy provider usage fields when the completed result exposes them."""
    usage_method = getattr(result, "usage", None)
    if not callable(usage_method):
        return None
    usage = usage_method()
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    values: dict[str, int | float | None] = {
        "requests": getattr(usage, "requests", None),
        "request_tokens": input_tokens,
        "response_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float))
        else None,
    }
    details = getattr(usage, "details", {})
    if isinstance(details, dict) and isinstance(details.get("cost"), int | float):
        values["cost"] = details["cost"]
    return values


def _conversation_id(messages: Sequence[object]) -> str | None:
    """Return the first conversation id recorded on captured Pydantic AI messages."""
    for message in messages:
        conversation_id = getattr(message, "conversation_id", None)
        if conversation_id is not None:
            return str(conversation_id)
    return None


def _recorded_actions_from_tool_events(
    tool_events: tuple[ToolEvent, ...], proposed_actions: Sequence[dict[str, object]] = ()
) -> tuple[dict[str, object], ...]:
    """Return all execute_home_code action records, including proposed service data.

    Production model-facing action records are intentionally compact and omit the
    original request ``service_data``. Eval scoring needs that proposed data for
    exact side-effect identity, so successful records are enriched from the
    eval-only ``RecordingInvoker`` calls that were dispatched through the private
    runtime seam. Blocked policy records are still sourced only from tool output.
    """
    actions: list[dict[str, object]] = []
    proposed_index = 0
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
                normalized = _for_scoring(action)
                if normalized.get("status") != "error":
                    normalized, proposed_index = _enrich_action_from_proposed(
                        normalized, proposed_actions, proposed_index
                    )
                actions.append(normalized)
    return tuple(actions)


def _enrich_action_from_proposed(
    action: dict[str, object], proposed_actions: Sequence[dict[str, object]], start_index: int
) -> tuple[dict[str, object], int]:
    """Attach proposed service_data to one successful compact action record."""
    for index in range(start_index, len(proposed_actions)):
        proposed = proposed_actions[index]
        if not _same_action_identity(action, proposed):
            continue
        enriched = dict(action)
        if "service_data" not in enriched:
            enriched["service_data"] = proposed.get("service_data")
        return enriched, index + 1
    return action, start_index


def _same_action_identity(action: dict[str, object], proposed: dict[str, object]) -> bool:
    """Return whether a compact output record matches a proposed invoker action."""
    return action.get("domain") == proposed.get("domain") and action.get("service") == proposed.get("service")


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
