"""Multi-turn eval task body reused by native experiments and DSPy."""

import json

from custom_components.llm_sandbox.const import TOOL_EXECUTE_HOME_CODE
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile

from llm_sandbox_evals import cases, prompts
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.models import ModelAdapter, ModelResponseError
from llm_sandbox_evals.schema import (
    CaseTrace,
    CheckResult,
    EvalCase,
    PromptCandidate,
    StepTrace,
)
from llm_sandbox_evals.scoring import check_case, score_case
from llm_sandbox_evals.tools import EVAL_SCOPE, RecordingInvoker, apply_scope, run_tool, tool_result_message


async def run_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    adapter: ModelAdapter,
    profile: PromptProfile,
    config: EvalConfig,
) -> CaseTrace:
    """Run one matrix cell through the bounded native tool-calling agent loop."""
    prompt = ""
    try:
        fixture = get_home(case.home)
        snapshot = apply_scope(fixture.snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
        messages = prompts.render_messages(candidate, case, snapshot)
        prompt = json.dumps(messages, indent=2)
        tools = prompts.function_schemas(candidate)
        invoker = RecordingInvoker()
        turns = 0
        recorded_actions: list[dict[str, object]] = []
        execute_statuses: set[str] = set()
        steps: list[StepTrace] = []
        final_answer = ""
        raw_output = ""
        tool_call: dict[str, object] | None = None
        tool_result: dict[str, object] | None = None
        cap = case.max_turns or config.max_turns

        while turns < cap:
            step = await adapter.respond(model_id, messages, tools)
            raw_output = step.raw
            messages.append(step.assistant_message)
            # Branch boundary: a message with no tool calls is the terminal natural-language answer.
            if not step.tool_calls:
                final_answer = step.text
                break

            results: list[dict[str, object] | None] = []
            for call in step.tool_calls:
                outcome = await run_tool(call, case, snapshot, profile, invoker=invoker)
                messages.append(tool_result_message(call.id, outcome.result))
                recorded_actions.extend(outcome.recorded_actions)
                status = _execution_status(outcome.result)
                if call.tool_name == TOOL_EXECUTE_HOME_CODE and status is not None:
                    execute_statuses.add(status)
                results.append(outcome.result)
                tool_call = {"id": call.id, "tool_name": call.tool_name, "tool_args": call.tool_args}
                tool_result = outcome.result
            steps.append(StepTrace(tool_calls=step.tool_calls, tool_results=tuple(results)))
            turns += 1
        checks = check_case(
            case,
            final_answer,
            tuple(recorded_actions),
            execute_statuses,
            snapshot,
            tuple(steps),
        )
        # Branch boundary: the eval loop must not synthesize a final answer after the cap.
        if turns >= cap and not final_answer:
            checks.append(
                CheckResult(
                    name="max_turns_exceeded",
                    passed=False,
                    required=True,
                    feedback=f"turns={turns} max_turns={cap}",
                )
            )
        score = score_case(checks, turns, case.par_turns, config.efficiency_k, config.efficiency_floor)
        return CaseTrace(
            case_id=case.id,
            category=case.category,
            candidate_id=candidate.id,
            model_id=model_id,
            score=score,
            prompt=prompt,
            raw_output=raw_output,
            tool_call=tool_call,
            tool_result=tool_result,
            recorded_actions=tuple(recorded_actions),
            checks=tuple(checks),
            turns=turns,
            par_turns=case.par_turns,
            final_answer=final_answer,
            steps=tuple(steps),
        )
    except ModelResponseError as err:
        # Branch boundary: provider failures are isolated to the current native evaluation case.
        return _error_trace(candidate, model_id, case, prompt, "model_error", err.detail)
    except Exception as err:  # noqa: BLE001 - harness isolates failures to the current matrix cell.
        return _error_trace(candidate, model_id, case, prompt, "harness_error", f"{type(err).__name__}: {err}")


def _error_trace(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    prompt: str,
    check_name: str,
    feedback: str,
) -> CaseTrace:
    """Return a zero-score trace for an infrastructure or provider failure."""
    return CaseTrace(
        case_id=case.id,
        category=case.category,
        candidate_id=candidate.id,
        model_id=model_id,
        score=0.0,
        prompt=prompt,
        raw_output=feedback,
        tool_call=None,
        tool_result=None,
        recorded_actions=(),
        checks=(
            CheckResult(
                name=check_name,
                passed=False,
                required=True,
                feedback=feedback,
            ),
        ),
        turns=0,
        par_turns=case.par_turns,
        final_answer="",
        steps=(),
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


def _execution_status(result: dict[str, object] | None) -> str | None:
    """Return nested execute_home_code execution.status from a tool result."""
    if result is None:
        return None
    execution = result.get("execution")
    if not isinstance(execution, dict):
        return None
    status = execution.get("status")
    return status if isinstance(status, str) else None
