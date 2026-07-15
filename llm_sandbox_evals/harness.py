"""Eval task body backed by a real Pydantic AI Agent."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import json
from time import perf_counter
import traceback

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from pydantic_ai import AgentRunResult, AgentRunResultEvent, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import UsageLimits

from llm_sandbox_evals import cases
from llm_sandbox_evals.agent_runner import build_agent, build_model_settings
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.phases import LanePhase, PhaseEmitter, PhaseObservation, PhaseObserver
from llm_sandbox_evals.runtime import ToolBoundaryCallback, build_eval_runtime
from llm_sandbox_evals.schema import (
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EvalCase,
    EvalDiagnostics,
    ExecutionError,
    FailureClassification,
    PromptCandidate,
    RequestVariant,
    ToolEvent,
)
from llm_sandbox_evals.scoring import assess_end_state, evaluate_case, extract_overlay_seeds, score_actions
from llm_sandbox_evals.scoring.actions import build_action_ledger
from llm_sandbox_evals.tool_events import tool_succeeded
from llm_sandbox_evals.tools import EVAL_SCOPE, _for_scoring, apply_scope

_TOOL_EXECUTE_HOME_CODE = "execute_home_code"
_TOKEN_QUOTA_EXCEEDED = "token_quota_exceeded"


@dataclass(frozen=True, slots=True)
class _CapturedFailure:
    """Normalized diagnostic failure data retained outside action scoring."""

    classification: FailureClassification
    execution_error: ExecutionError
    detail: str


async def run_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    request_variant: RequestVariant,
    config: EvalConfig,
    *,
    profile: PromptProfile,
    on_tool_boundary: ToolBoundaryCallback | None = None,
    on_response: Callable[[str], None] | None = None,
    on_phase: PhaseObserver | None = None,
) -> CaseTrace:
    """Run one matrix cell through the production-core Pydantic AI agent."""
    started = perf_counter()
    captured: Sequence[object] = ()
    proposed_actions: Sequence[dict[str, object]] = ()
    current_phase: LanePhase | None = None

    def emit_phase(phase: LanePhase, tool_name: str | None = None) -> None:
        """Forward one payload-free phase without allowing observers to alter execution."""
        nonlocal current_phase
        # State mutation boundary: a delayed model tool-call event must not replace a runtime tool phase.
        if phase == "preparing_tool_call" and current_phase in {"running_tool", "processing_tool_result"}:
            return
        current_phase = phase
        if on_phase is not None:
            try:
                on_phase(PhaseObservation(phase, tool_name))
            except Exception:  # noqa: BLE001 - phase observers are isolated from execution and scoring.
                return

    def finish_trace(trace: CaseTrace) -> CaseTrace:
        """Mark a trace terminal only after its complete result exists."""
        emit_phase("finished")
        return trace

    def on_runtime_tool_boundary(tool_name: str, tool_started: bool) -> None:
        """Observe actual tool runtime boundaries before retaining existing lifecycle callbacks."""
        emit_phase("running_tool" if tool_started else "processing_tool_result", tool_name)
        if on_tool_boundary is not None:
            try:
                on_tool_boundary(tool_name, tool_started)
            except Exception:  # noqa: BLE001 - tool observers are isolated from the production tool result.
                return

    async def consume_stream() -> AgentRunResult[str]:
        """Consume the complete native event stream and return its terminal result."""
        emit_phase("awaiting_model")
        async with agent.run_stream_events(
            request_variant.text,
            deps=runtime,
            model_settings=build_model_settings(
                model_id,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
            ),
            usage_limits=UsageLimits(tool_calls_limit=config.max_tool_calls),
        ) as events:
            async for event in events:
                _observe_stream_phase(event, emit_phase)
                if isinstance(event, AgentRunResultEvent):
                    return event.result
        raise UnexpectedModelBehavior("agent event stream ended without a terminal result")

    try:
        emit_phase("queued")
        fixture = get_home(case.home)
        snapshot = apply_scope(fixture.snapshot(), EVAL_SCOPE)
        runtime = build_eval_runtime(
            case, candidate, profile, snapshot, fixture, on_tool_boundary=on_runtime_tool_boundary
        )
        proposed_actions = runtime.invoker.calls
        agent = build_agent(runtime, model_id)
        with capture_run_messages() as captured:
            result = await asyncio.wait_for(consume_stream(), timeout=config.model_timeout)
        output = result.output
        if on_response is not None:
            # Safety constraint: an observer cannot alter a completed model result or scoring.
            with suppress(Exception):
                on_response(output)
        messages = result.all_messages()
        tool_events = _tool_events(messages)
        recorded_actions = _recorded_actions_from_tool_events(tool_events, runtime.invoker.calls)
        conversation_id = _conversation_id(messages)
        emit_phase("scoring")
        overlay_seeds = extract_overlay_seeds(snapshot, case.desired_entities)
        recorded_invocations = tuple(dict(call) for call in runtime.invoker.calls)
        evaluation = evaluate_case(
            case,
            recorded_actions,
            overlay_seeds=overlay_seeds,
            invoker_calls=recorded_invocations,
            tool_events=tool_events,
            answer=output,
        )
        return finish_trace(
            CaseTrace(
                case_id=case.id,
                candidate_id=candidate.id,
                model_id=model_id,
                request_variant_id=request_variant.id,
                request_text=request_variant.text,
                answer=output,
                required_actions=case.required_actions,
                desired_entities=case.desired_entities,
                overlay_state_seeds=overlay_seeds,
                recorded_invocations=recorded_invocations,
                end_state_result=evaluation.end_state_result,
                outcome=evaluation.outcome,
                action_result=evaluation.action_result,
                action_ledger=evaluation.action_ledger,
                tool_events=tool_events,
                diagnostics=_diagnostics(
                    tool_events,
                    messages,
                    elapsed=perf_counter() - started,
                    usage=None if model_id == "stub" else _usage(result, messages),
                    failure=None,
                ),
                provider_error=None,
                conversation_id=conversation_id,
                category=case.category,
                tags=case.tags,
                oracle=case.oracle,
                expected_tool_calls=case.expected_tool_calls,
                expected_answer=case.expected_answer,
                tool_call_result=evaluation.tool_call_result,
                answer_result=evaluation.answer_result,
                reasoning_effort=config.reasoning_effort,
                temperature=config.temperature,
            ),
        )
    except UsageLimitExceeded as err:
        # Branch boundary: the model exhausted its allowed tool calls, which is scored behavior rather than a harness error.
        tool_events = _tool_events(captured)
        return finish_trace(
            _failure_trace(
                candidate,
                model_id,
                case,
                request_variant,
                "cap_exhausted",
                _format_exception(err),
                diagnostic=False,
                elapsed=perf_counter() - started,
                tool_events=tool_events,
                messages=captured,
                recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
                snapshot=snapshot,
                invoker_calls=proposed_actions,
                conversation_id=_conversation_id(captured),
                reasoning_effort=config.reasoning_effort,
                temperature=config.temperature,
                on_scoring=emit_phase,
            )
        )
    except Exception as err:  # noqa: BLE001 - provider/harness failures are isolated to the current matrix cell.
        captured_failure = _capture_failure(
            err, timeout=config.model_timeout if isinstance(err, TimeoutError) else None
        )
        tool_events = _tool_events(captured)
        return finish_trace(
            _failure_trace(
                candidate,
                model_id,
                case,
                request_variant,
                captured_failure.classification,
                captured_failure.detail,
                diagnostic=True,
                execution_error=captured_failure.execution_error,
                elapsed=perf_counter() - started,
                tool_events=tool_events,
                messages=captured,
                recorded_actions=_recorded_actions_from_tool_events(tool_events, proposed_actions),
                snapshot=snapshot,
                invoker_calls=proposed_actions,
                conversation_id=_conversation_id(captured),
                reasoning_effort=config.reasoning_effort,
                temperature=config.temperature,
                on_scoring=emit_phase,
            )
        )


def _failure_trace(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    request_variant: RequestVariant,
    failure: FailureClassification,
    feedback: str,
    *,
    diagnostic: bool,
    execution_error: ExecutionError | None = None,
    elapsed: float | None = None,
    tool_events: tuple[ToolEvent, ...] = (),
    messages: Sequence[object] = (),
    recorded_actions: tuple[dict[str, object], ...] = (),
    snapshot: HomeSnapshot | None = None,
    invoker_calls: Sequence[Mapping[str, object]] = (),
    conversation_id: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    on_scoring: Callable[[LanePhase], None] | None = None,
) -> CaseTrace:
    """Return an incomplete or cap-exhausted trace with captured diagnostics."""
    if on_scoring is not None:
        on_scoring("scoring")
    action_ledger = build_action_ledger(recorded_actions)
    scored_actions = score_actions(case.required_actions, action_ledger)
    action_result = ActionResult(
        False, scored_actions.reason, scored_actions.comparisons, scored_actions.unexpected_actions
    )
    overlay_seeds = extract_overlay_seeds(snapshot, case.desired_entities) if snapshot is not None else ()
    recorded_invocations = tuple(dict(call) for call in invoker_calls)
    end_state_result = assess_end_state(case.desired_entities, overlay_seeds, recorded_invocations)
    is_cap_exhausted = failure == "cap_exhausted"
    if is_cap_exhausted:
        # Branch boundary: cap exhaustion overrides both state and action scoring as an operational scored outcome.
        outcome = CaseOutcome("incorrect", "cap_exhausted", "cap_exhausted")
    else:
        outcome = CaseOutcome("incomplete", None, None)
    return CaseTrace(
        case_id=case.id,
        candidate_id=candidate.id,
        model_id=model_id,
        request_variant_id=request_variant.id,
        request_text=request_variant.text,
        answer=None,
        required_actions=case.required_actions,
        desired_entities=case.desired_entities,
        overlay_state_seeds=overlay_seeds,
        recorded_invocations=recorded_invocations,
        end_state_result=end_state_result,
        outcome=outcome,
        action_result=action_result,
        action_ledger=action_ledger,
        tool_events=tool_events,
        diagnostics=_diagnostics(
            tool_events,
            messages,
            elapsed=elapsed,
            # Branch boundary: preserve response-level provider usage captured before a failed terminal result.
            usage=None if model_id == "stub" else _partial_usage(messages),
            failure=failure,
            cap_exhausted=is_cap_exhausted,
        ),
        provider_error=feedback if diagnostic else None,
        execution_error=execution_error if diagnostic else None,
        conversation_id=conversation_id,
        category=case.category,
        tags=case.tags,
        oracle=case.oracle,
        expected_tool_calls=case.expected_tool_calls,
        expected_answer=case.expected_answer,
        tool_call_result=None,
        answer_result=None,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )


def _observe_stream_phase(event: object, emit_phase: PhaseEmitter) -> None:
    """Map only safe native stream facts into an execution phase."""
    if isinstance(event, FunctionToolCallEvent):
        emit_phase("preparing_tool_call", event.part.tool_name)
    elif isinstance(event, PartStartEvent | PartEndEvent):
        if isinstance(event.part, ThinkingPart):
            emit_phase("thinking")
        elif isinstance(event.part, TextPart):
            emit_phase("responding")
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta):
            emit_phase("thinking")
        elif isinstance(event.delta, TextPartDelta):
            emit_phase("responding")


def _capture_failure(err: BaseException, *, timeout: float | None = None) -> _CapturedFailure:
    """Return the typed classification, structured metadata, and raw traceback for one failure."""
    source: BaseException
    # Branch boundary: an actual timeout remains a timeout even if Python retained unrelated context.
    if isinstance(err, TimeoutError):
        classification: FailureClassification = "timeout"
        source = err
    elif isinstance(err, ModelHTTPError):
        classification = _classify_http_error(err)
        source = err
    elif isinstance(err, UnexpectedModelBehavior):
        classification = "model_protocol_error"
        source = err
    else:
        http_error = _find_model_http_error(err)
        if http_error is not None:
            classification = _classify_http_error(http_error)
            source = http_error
        elif _contains_exception(err, TimeoutError):
            classification = "timeout"
            source = _find_exception(err, TimeoutError) or err
        elif _contains_exception(err, UnexpectedModelBehavior):
            classification = "model_protocol_error"
            source = _find_exception(err, UnexpectedModelBehavior) or err
        else:
            classification = "provider_error"
            source = err

    return _CapturedFailure(
        classification=classification,
        execution_error=_execution_error(source, timeout=timeout),
        detail=_traceback_detail(err, timeout=timeout),
    )


def _classify_http_error(err: ModelHTTPError) -> FailureClassification:
    """Classify provider HTTP errors from structured status and body fields."""
    if err.status_code == 429 or _http_body_has_token_quota(err.body):
        return "rate_limit"
    return "provider_error"


def _execution_error(err: BaseException, *, timeout: float | None = None) -> ExecutionError:
    """Build JSON-compatible structured error metadata from the selected source exception."""
    message = _exception_message(err, timeout=timeout)
    if isinstance(err, ModelHTTPError):
        code, error_type = _http_error_code_and_type(err.body)
        return ExecutionError(
            exception_type=type(err).__name__,
            message=message,
            status_code=err.status_code,
            provider_code=_http_error_provider_code(err.body) or code or error_type,
            provider_model=err.model_name,
            provider_detail=_provider_detail(err.body),
        )
    return ExecutionError(exception_type=type(err).__name__, message=message)


def _http_error_code_and_type(body: object | None) -> tuple[str | None, str | None]:
    """Return nonempty provider error code/type values from mapping-shaped HTTP bodies."""
    candidates = _http_error_code_type_candidates(body)
    return candidates[0] if candidates else (None, None)


def _http_error_provider_code(body: object | None) -> str | None:
    """Return the diagnostic provider code, surfacing quota when it drives classification."""
    for code, error_type in _http_error_code_type_candidates(body):
        if code == _TOKEN_QUOTA_EXCEEDED or error_type == _TOKEN_QUOTA_EXCEEDED:
            return _TOKEN_QUOTA_EXCEEDED
    code, error_type = _http_error_code_and_type(body)
    return code or error_type


def _http_body_has_token_quota(body: object | None) -> bool:
    """Return whether any provider code/type slot carries the token quota indicator."""
    return any(
        code == _TOKEN_QUOTA_EXCEEDED or error_type == _TOKEN_QUOTA_EXCEEDED
        for code, error_type in _http_error_code_type_candidates(body)
    )


def _http_error_code_type_candidates(body: object | None) -> tuple[tuple[str | None, str | None], ...]:
    """Return direct provider code/type first, then nested error code/type."""
    if not isinstance(body, Mapping):
        return ()

    candidates: list[tuple[str | None, str | None]] = []
    direct_code = _nonempty_string(body.get("code"))
    direct_type = _nonempty_string(body.get("type"))
    if direct_code is not None or direct_type is not None:
        candidates.append((direct_code, direct_type))

    raw_error = body.get("error")
    if isinstance(raw_error, Mapping):
        nested_code = _nonempty_string(raw_error.get("code"))
        nested_type = _nonempty_string(raw_error.get("type"))
        if nested_code is not None or nested_type is not None:
            candidates.append((nested_code, nested_type))
    return tuple(candidates)


def _nonempty_string(value: object) -> str | None:
    """Return a provider-supplied string only when it carries a value."""
    return value if isinstance(value, str) and value else None


def _provider_detail(body: object | None) -> str | None:
    """Preserve provider HTTP bodies in a JSON-compatible diagnostic string."""
    if body is None:
        return None
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, indent=2, sort_keys=True, default=repr)
    except TypeError, ValueError:
        return repr(body)


def _traceback_detail(err: BaseException, *, timeout: float | None = None) -> str:
    """Return a full chained traceback, retaining timeout metadata when present."""
    detail = "".join(traceback.format_exception(err)).rstrip()
    if timeout is not None:
        detail = f"{detail}\nafter={timeout:g}s"
    return detail


def _find_model_http_error(err: BaseException) -> ModelHTTPError | None:
    """Return the first Pydantic AI HTTP error in the chained exception graph."""
    return _find_exception(err, ModelHTTPError)


def _find_exception[ExceptionT: BaseException](
    err: BaseException, exception_type: type[ExceptionT]
) -> ExceptionT | None:
    """Return the first matching exception while avoiding chained-exception cycles."""
    for current in _iter_exception_chain(err):
        if isinstance(current, exception_type):
            return current
    return None


def _contains_exception(err: BaseException, exception_type: type[BaseException]) -> bool:
    """Return whether any chained exception has the requested type."""
    return _find_exception(err, exception_type) is not None


def _iter_exception_chain(err: BaseException) -> Sequence[BaseException]:
    """Return the root/cause/context chain without following cycles indefinitely."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        # Branch boundary: explicit causes supersede implicit contexts in Python's displayed chain.
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _format_exception(err: BaseException, *, timeout: float | None = None) -> str:
    """Format provider, timeout, and limit failures as compact one-line feedback."""
    message = _exception_message(err, timeout=timeout)
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


def _exception_message(err: BaseException, *, timeout: float | None = None) -> str:
    """Return a non-empty one-line exception message."""
    message = " ".join(str(err).split())
    if isinstance(err, TimeoutError):
        message = message or "timed out"
    else:
        message = message or "no detail"
    if timeout is not None:
        return f"{message} after={timeout:g}s"
    return message


def _select_cases(case_filters: list[str] | None, home_filters: list[str] | None) -> list[EvalCase]:
    """Select cases by id and optional home name, preserving CASES order."""
    selected = cases.CASES
    if home_filters is not None:
        home_names = set(home_filters)
        selected = [case for case in selected if case.home in home_names]

    # Branch boundary: no case filter means all remaining cases are selected.
    if case_filters is None:
        return list(selected)

    requested = set(case_filters)
    return [case for case in selected if case.id in requested]


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

    Tool-call arguments and production result envelopes are captured for
    dedicated tool-contract scoring and diagnostics.
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
    failure: FailureClassification | None,
    cap_exhausted: bool = False,
) -> EvalDiagnostics:
    """Build diagnostics without turning attempts, timing, or usage into score."""
    successful = tuple(event for event in events if tool_succeeded(event))
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


def _usage(result: object, messages: Sequence[object] = ()) -> dict[str, int | float | bool | None] | None:
    """Copy final usage or sum ModelResponse usage when final usage is unavailable."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return _partial_usage(messages)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    # Branch boundary: some providers expose a final result usage shell without token components.
    if input_tokens is None and output_tokens is None:
        return _partial_usage(messages)
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


def _partial_usage(messages: Sequence[object]) -> dict[str, int | float | bool | None] | None:
    """Aggregate response-level usage retained in captured messages after an incomplete run."""
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    if not responses:
        return None
    fields = {
        "requests": "requests",
        "request_tokens": "input_tokens",
        "response_tokens": "output_tokens",
    }
    values: dict[str, int | float | bool | None] = dict.fromkeys(fields)
    found = False
    for name, field_name in fields.items():
        present = [getattr(response.usage, field_name, None) for response in responses]
        numeric = [value for value in present if isinstance(value, int | float) and not isinstance(value, bool)]
        if numeric:
            values[name] = sum(numeric)
            found = True
    request_tokens = values["request_tokens"]
    response_tokens = values["response_tokens"]
    values["total_tokens"] = (
        request_tokens + response_tokens
        if isinstance(request_tokens, int | float) and isinstance(response_tokens, int | float)
        else None
    )
    if values["total_tokens"] is not None:
        found = True
    values["partial"] = True
    return values if found else None


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
