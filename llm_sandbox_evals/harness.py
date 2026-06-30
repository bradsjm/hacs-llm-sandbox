"""Matrix orchestration for dev-only eval runs."""

import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_sandbox_evals import cases
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.models import ModelAdapter, get_adapter
from llm_sandbox_evals.prompts import load_candidates, render_prompt
from llm_sandbox_evals.schema import CheckResult, EvalCase, PromptCandidate
from llm_sandbox_evals.scoring import check_case, mean_score, score_case
from llm_sandbox_evals.tools import run_tool


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """Full trace for one candidate/model/case execution."""

    case_id: str
    category: str
    candidate_id: str
    model_id: str
    score: float
    prompt: str
    raw_output: str
    tool_call: dict[str, object] | None
    tool_result: dict[str, object] | None
    recorded_actions: tuple[dict[str, object], ...]
    checks: tuple[CheckResult, ...]


@dataclass(frozen=True, slots=True)
class CandidateModelScore:
    """Aggregated scores for one candidate/model pair."""

    candidate_id: str
    model_id: str
    mean: float
    per_category: dict[str, float]
    case_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Complete result for one eval matrix run."""

    run_id: str
    created_at: str
    candidate_ids: list[str]
    model_ids: list[str]
    case_ids: list[str]
    traces: list[CaseTrace]
    scores: list[CandidateModelScore]


async def run_matrix(config: EvalConfig) -> RunResult:
    """Run the candidate x model x case matrix and return all traces/scores."""
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    created_at = datetime.now(UTC).isoformat()
    candidates = load_candidates(config.candidates)
    selected_cases = _select_cases(config.cases, config.homes)
    model_ids = list(config.models)
    traces: list[CaseTrace] = []

    for candidate in candidates:
        for model_id in model_ids:
            adapter = get_adapter(model_id)
            _progress(
                f"[{candidate.id}/{model_id}] {len(selected_cases)} cases "
                f"(concurrency={config.concurrency})"
            )
            pair_traces = await _run_cases_for_pair(
                candidate, model_id, selected_cases, adapter, config.concurrency
            )
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


async def _run_one_case(
    candidate: PromptCandidate,
    model_id: str,
    case: EvalCase,
    adapter: ModelAdapter,
) -> CaseTrace:
    """Run one matrix cell, converting all failures into a zero-score trace."""
    prompt = ""
    try:
        fixture = get_home(case.home)
        snapshot = fixture.snapshot()
        prompt = render_prompt(candidate, case, snapshot)
        model_result = await adapter.complete(model_id, prompt)
        outcome = await run_tool(model_result.tool_call, case, snapshot)
        checks = check_case(case, model_result.tool_call, outcome, snapshot)
        return CaseTrace(
            case_id=case.id,
            category=case.category,
            candidate_id=candidate.id,
            model_id=model_id,
            score=score_case(checks),
            prompt=prompt,
            raw_output=model_result.raw_text,
            tool_call=model_result.tool_call,
            tool_result=outcome.result,
            recorded_actions=outcome.recorded_actions,
            checks=tuple(checks),
        )
    except Exception as err:  # noqa: BLE001 - harness isolates failures to the current matrix cell.
        return CaseTrace(
            case_id=case.id,
            category=case.category,
            candidate_id=candidate.id,
            model_id=model_id,
            score=0.0,
            prompt=prompt,
            raw_output="",
            tool_call=None,
            tool_result=None,
            recorded_actions=(),
            checks=(
                CheckResult(
                    name="harness_error",
                    passed=False,
                    required=True,
                    feedback=f"{type(err).__name__}: {err}",
                ),
            ),
        )


async def _run_cases_for_pair(
    candidate: PromptCandidate,
    model_id: str,
    selected_cases: list[EvalCase],
    adapter: ModelAdapter,
    concurrency: int,
) -> list[CaseTrace]:
    """Run all cases for one candidate/model pair with bounded concurrency.

    Each case runs in its own asyncio task, so the executor's runtime
    contextvars stay isolated per task. A semaphore caps concurrent model calls
    (the slow I/O). Progress is written to stderr as cases complete; results are
    returned in input case order.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(selected_cases)

    async def _one(index: int, case: EvalCase) -> CaseTrace:
        async with semaphore:
            trace = await _run_one_case(candidate, model_id, case, adapter)
        _progress(f"  [{index + 1}/{total}] {case.id} score={trace.score:.2f}")
        return trace

    return await asyncio.gather(*[_one(i, case) for i, case in enumerate(selected_cases)])


def _progress(message: str) -> None:
    """Write a progress line to stderr (ruff T201 forbids the print builtin)."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


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
            for case in selected_cases:
                trace = traces_by_pair.get((candidate.id, model_id, case.id))
                case_scores[case.id] = 0.0 if trace is None else trace.score
            for category in categories:
                category_scores = [case_scores[case.id] for case in selected_cases if case.category == category]
                per_category[category] = mean_score(category_scores)
            scores.append(
                CandidateModelScore(
                    candidate_id=candidate.id,
                    model_id=model_id,
                    mean=mean_score(list(case_scores.values())),
                    per_category=per_category,
                    case_scores=case_scores,
                )
            )
    return scores
