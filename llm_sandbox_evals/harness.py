"""Matrix orchestration for dev-only multi-turn eval runs."""

import asyncio
import json
import sys
from datetime import UTC, datetime

from custom_components.llm_sandbox.const import TOOL_EXECUTE_HOME_CODE
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile, resolve_profile

from llm_sandbox_evals import cases, prompts
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.models import ModelAdapter, ModelResponseError, get_adapter
from llm_sandbox_evals.prompts import load_candidates
from llm_sandbox_evals.schema import (
    CandidateModelScore,
    CaseTrace,
    CheckResult,
    EvalCase,
    PromptCandidate,
    RunResult,
    StepTrace,
    ToolCall,
)
from llm_sandbox_evals.scoring import check_case, entity_ids_from_action, mean_score, score_case, strings_from_value
from llm_sandbox_evals.tools import EVAL_SCOPE, RecordingInvoker, apply_scope, run_tool, tool_result_message


async def run_matrix(config: EvalConfig) -> RunResult:
    """Run the candidate x model x case matrix and return all traces/scores."""
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    created_at = datetime.now(UTC).isoformat()
    profile = resolve_profile(config.prompt_profile)
    candidates = load_candidates(config.candidates, config.prompt_profile)
    selected_cases = _select_cases(config.cases, config.homes)
    model_ids = list(config.models)
    traces: list[CaseTrace] = []

    for candidate in candidates:
        for model_id in model_ids:
            adapter = get_adapter(model_id, config.reasoning_effort, model_timeout=config.model_timeout)
            _progress(f"[{candidate.id}/{model_id}] {len(selected_cases)} cases (concurrency={config.concurrency})")
            pair_traces = await _run_cases_for_pair(candidate, model_id, selected_cases, adapter, config, profile)
            traces.extend(pair_traces)

    return RunResult(
        run_id=run_id,
        created_at=created_at,
        candidate_ids=[candidate.id for candidate in candidates],
        model_ids=model_ids,
        case_ids=[case.id for case in selected_cases],
        traces=traces,
        scores=_score_matrix(traces, candidates, model_ids, selected_cases),
    )


async def run_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    adapter: ModelAdapter,
    profile: PromptProfile,
    config: EvalConfig,
    *,
    raise_model_errors: bool = False,
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
        referenced_entity_ids: set[str] = set()
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
                referenced_entity_ids.update(_referenced_ids_from_call(call))
                outcome = await run_tool(call, case, snapshot, profile, invoker=invoker)
                messages.append(tool_result_message(call.id, outcome.result))
                recorded_actions.extend(outcome.recorded_actions)
                referenced_entity_ids.update(_referenced_ids_from_result(outcome.result))
                for action in outcome.recorded_actions:
                    referenced_entity_ids.update(entity_ids_from_action(action, snapshot))
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
            referenced_entity_ids,
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
        # Branch boundary: pair orchestration may choose to fail fast across remaining cases for the same model.
        if raise_model_errors:
            raise
        return _error_trace(candidate, model_id, case, prompt, "model_error", err.detail)
    except Exception as err:  # noqa: BLE001 - harness isolates failures to the current matrix cell.
        return _error_trace(candidate, model_id, case, prompt, "harness_error", f"{type(err).__name__}: {err}")


async def _run_cases_for_pair(
    candidate: PromptCandidate,
    model_id: str,
    selected_cases: list[EvalCase],
    adapter: ModelAdapter,
    config: EvalConfig,
    prompt_profile: PromptProfile,
) -> list[CaseTrace]:
    """Run all cases for one candidate/model pair with bounded concurrency."""
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    model_error_lock = asyncio.Lock()
    model_error: ModelResponseError | None = None
    total = len(selected_cases)

    async def _one(index: int, case: EvalCase) -> CaseTrace:
        nonlocal model_error
        async with semaphore:
            async with model_error_lock:
                current_model_error = model_error
            # Branch boundary: once a provider/model setup error is known, remaining cases should not call it again.
            if current_model_error is not None:
                trace = _error_trace(candidate, model_id, case, "", "model_error", current_model_error.detail)
            else:
                try:
                    trace = await run_case(
                        candidate,
                        model_id,
                        case,
                        adapter,
                        prompt_profile,
                        config,
                        raise_model_errors=True,
                    )
                except ModelResponseError as err:
                    async with model_error_lock:
                        if model_error is None:
                            model_error = err
                            _log_model_error(candidate, model_id, case, err)
                    trace = _error_trace(candidate, model_id, case, "", "model_error", err.detail)
        _progress(f"  [{index + 1}/{total}] {case.id} score={trace.score:.2f} turns={trace.turns}")
        return trace

    return await asyncio.gather(*[_one(i, case) for i, case in enumerate(selected_cases)])


def _progress(message: str) -> None:
    """Write a progress line to stderr (ruff T201 forbids the print builtin)."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _log_model_error(candidate: PromptCandidate, model_id: str, case: EvalCase, err: ModelResponseError) -> None:
    """Write detailed provider diagnostics once per failing candidate/model pair."""
    _progress(
        f"  model error for candidate={candidate.id} model={model_id} case={case.id}; "
        "skipping remaining cases for this pair"
    )
    for line in err.detail.splitlines():
        _progress(f"    {line}")


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


def _score_matrix(
    traces: list[CaseTrace],
    candidates: list[PromptCandidate],
    model_ids: list[str],
    selected_cases: list[EvalCase],
) -> list[CandidateModelScore]:
    """Aggregate case traces into deterministic candidate/model summaries."""
    scores: list[CandidateModelScore] = []
    categories = list(dict.fromkeys(case.category for case in selected_cases))
    traces_by_pair = {(trace.candidate_id, trace.model_id, trace.case_id): trace for trace in traces}

    for candidate in candidates:
        for model_id in model_ids:
            case_scores: dict[str, float] = {}
            per_category: dict[str, float] = {}
            pair_traces: list[CaseTrace] = []
            for case in selected_cases:
                trace = traces_by_pair.get((candidate.id, model_id, case.id))
                case_scores[case.id] = 0.0 if trace is None else trace.score
                if trace is not None:
                    pair_traces.append(trace)
            for category in categories:
                category_scores = [case_scores[case.id] for case in selected_cases if case.category == category]
                per_category[category] = mean_score(category_scores)
            scores.append(
                CandidateModelScore(
                    candidate_id=candidate.id,
                    model_id=model_id,
                    mean=mean_score(list(case_scores.values())),
                    mean_turns=mean_score([float(trace.turns) for trace in pair_traces]),
                    per_category=per_category,
                    case_scores=case_scores,
                )
            )
    return scores


def _execution_status(result: dict[str, object] | None) -> str | None:
    """Return nested execute_home_code execution.status from a tool result."""
    if result is None:
        return None
    execution = result.get("execution")
    if not isinstance(execution, dict):
        return None
    status = execution.get("status")
    return status if isinstance(status, str) else None


def _referenced_ids_from_call(call: ToolCall) -> set[str]:
    """Return explicit entity/statistic ids present in native tool arguments."""
    ids = set(strings_from_value(call.tool_args.get("entity_ids")))
    ids.update(strings_from_value(call.tool_args.get("statistic_ids")))
    return ids


def _referenced_ids_from_result(result: dict[str, object] | None) -> set[str]:
    """Return recorder entity/statistic ids surfaced by fixture result envelopes."""
    if result is None:
        return set()
    ids: set[str] = set()
    entities = result.get("entities")
    if isinstance(entities, dict):
        ids.update(str(entity_id) for entity_id in entities)
    statistics = result.get("statistics")
    if isinstance(statistics, dict):
        ids.update(str(statistic_id) for statistic_id in statistics)
    return ids
