"""Native pydantic-evals experiment for the candidate x model x case matrix."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from time import perf_counter
from typing import Literal

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile, resolve_profile
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting import EvaluationReport, ReportCase
from pydantic_evals.reporting.analyses import ReportAnalysis, ScalarResult, TableResult

from llm_sandbox_evals import prompts
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _select_cases, run_case
from llm_sandbox_evals.schema import CaseTrace, EvalCase, PromptCandidate

type MatrixCellMeta = dict[str, str | int]
type MatrixEventCallback = Callable[[MatrixProgressEvent], None]
type MatrixProgressState = Literal[
    "matrix_started", "cell_started", "tool_started", "tool_finished", "response_received", "cell_finished"
]


@dataclass(frozen=True, slots=True)
class MatrixCellRef:
    """JSON-native reference stored in pydantic-evals case inputs."""

    case_id: str
    candidate_id: str
    model_id: str
    home: str
    category: str


@dataclass(frozen=True, slots=True)
class MatrixProgressEvent:
    """Observer-only lifecycle fact for one matrix evaluation."""

    state: MatrixProgressState
    cell: MatrixCellRef | None = None
    request: str | None = None
    tool_name: str | None = None
    response: str | None = None
    trace: CaseTrace | None = None
    elapsed: float | None = None
    completion_index: int | None = None
    total: int | None = None


@dataclass(slots=True)
class SandboxOutcome(Evaluator[MatrixCellRef, CaseTrace, MatrixCellMeta]):
    """Expose the v2 binary outcome as native pydantic-evals results."""

    def evaluate(
        self, ctx: EvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]
    ) -> dict[str, bool | str | EvaluationReason]:
        """Publish binary quality and provider classification labels."""
        trace = ctx.output
        return {
            "score": EvaluationReason(value=trace.outcome.score, reason=trace.outcome.reason),
            "outcome": trace.outcome.state,
            "incomplete": trace.outcome.state == "incomplete",
            "failure_classification": trace.diagnostics.failure or "none",
        }


@dataclass(slots=True)
class CandidateMatrixReport(ReportEvaluator[MatrixCellRef, CaseTrace, MatrixCellMeta]):
    """Aggregate outcomes while keeping operational diagnostics out of ranking."""

    candidate_ids: list[str]
    model_ids: list[str]
    prompt_sizes: dict[str, tuple[int, int]]

    def evaluate(self, ctx: ReportEvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[ReportAnalysis]:
        """Return outcome rates, coverage, and non-ranking diagnostics."""
        cases = list(ctx.report.cases)
        categories = _categories(cases)
        pairs = _pair_rows(cases, self.candidate_ids, self.model_ids, categories, self.prompt_sizes)
        ranking = _ranking_rows(pairs, self.candidate_ids, self.model_ids, categories)
        counts = _counts([case.output for case in cases])
        return [
            TableResult(
                title="Candidate ranking",
                columns=[
                    "Candidate",
                    "Correct",
                    "Incorrect",
                    "Incomplete",
                    "Completed",
                    "Correct rate",
                    "Coverage",
                    "PromptChars",
                    "SizeRatio",
                    *categories,
                ],
                rows=ranking,
            ),
            TableResult(
                title="Candidate x model outcomes",
                columns=[
                    "Candidate",
                    "Model",
                    "Correct",
                    "Incorrect",
                    "Incomplete",
                    "Completed",
                    "Correct rate",
                    "Coverage",
                    "Calls",
                    "FailedCalls",
                    "Turns",
                    "Elapsed",
                    "Tokens",
                    "Cost",
                    *categories,
                ],
                rows=[
                    [
                        row.candidate_id,
                        row.model_id,
                        row.correct,
                        row.incorrect,
                        row.incomplete,
                        row.completed,
                        _optional_round(row.correct_rate),
                        _round(row.coverage),
                        _round(row.mean_calls),
                        _round(row.mean_failed_calls),
                        _round(row.mean_turns),
                        _round(row.mean_elapsed),
                        _optional_round(row.mean_tokens),
                        _optional_round(row.mean_cost),
                        *[_optional_round(row.category_rates[category]) for category in categories],
                    ]
                    for row in pairs
                ],
            ),
            ScalarResult(title="Overall correct rate", value=_round(_rate(counts["correct"], counts["completed"]))),
            ScalarResult(title="Correct cells", value=counts["correct"]),
            ScalarResult(title="Incorrect cells", value=counts["incorrect"]),
            ScalarResult(title="Incomplete cells", value=counts["incomplete"]),
            ScalarResult(title="Completed cells", value=counts["completed"]),
            ScalarResult(title="Coverage", value=_round(_rate(counts["completed"], len(cases)))),
        ]


@dataclass(frozen=True, slots=True)
class _PairRow:
    candidate_id: str
    model_id: str
    correct: int
    incorrect: int
    incomplete: int
    completed: int
    correct_rate: float | None
    coverage: float
    mean_calls: float
    mean_failed_calls: float
    mean_turns: float
    mean_elapsed: float
    mean_tokens: float | None
    mean_cost: float | None
    category_rates: dict[str, float | None]
    api_prompt_chars: int
    prompt_chars: int


def build_dataset(
    config: EvalConfig, candidates: Sequence[PromptCandidate], selected_cases: Sequence[EvalCase], run_id: str
) -> Dataset[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Build one native dataset containing every candidate/model/case matrix cell."""
    cases: list[Case[MatrixCellRef, CaseTrace, MatrixCellMeta]] = []
    for candidate in candidates:
        for model_id in config.models:
            for case in selected_cases:
                ref = MatrixCellRef(case.id, candidate.id, model_id, case.home, case.category)
                metadata: MatrixCellMeta = {
                    "run_id": run_id,
                    "case_id": ref.case_id,
                    "candidate_id": ref.candidate_id,
                    "model_id": ref.model_id,
                    "home": ref.home,
                    "category": ref.category,
                }
                cases.append(
                    Case(
                        name=f"{candidate.id}/{model_id}/{case.id}",
                        inputs=ref,
                        metadata=metadata,
                        evaluators=(SandboxOutcome(),),
                    )
                )
    return Dataset(
        name="llm_sandbox_matrix",
        cases=cases,
        evaluators=(),
        report_evaluators=(
            CandidateMatrixReport(
                [c.id for c in candidates],
                list(config.models),
                {c.id: prompts.candidate_prompt_sizes(c) for c in candidates},
            ),
        ),
    )


def make_matrix_task(
    config: EvalConfig,
    profile: PromptProfile,
    candidate_by_id: dict[str, PromptCandidate],
    case_by_id: dict[str, EvalCase],
    *,
    total: int,
    on_event: MatrixEventCallback | None = None,
) -> Callable[[MatrixCellRef], Awaitable[CaseTrace]]:
    """Build the pydantic-evals task that preserves run_case snapshot/tool semantics."""
    completed = 0

    async def task(cell: MatrixCellRef) -> CaseTrace:
        nonlocal completed
        case = case_by_id[cell.case_id]
        request = _presentation_request(case.user_request)
        start = perf_counter()
        _emit_event(on_event, MatrixProgressEvent("cell_started", cell=cell, request=request, total=total))

        def on_tool_boundary(tool_name: str, started: bool) -> None:
            _emit_event(
                on_event,
                MatrixProgressEvent(
                    "tool_started" if started else "tool_finished", cell=cell, request=request, tool_name=tool_name
                ),
            )

        def on_response(response: str) -> None:
            _emit_event(
                on_event, MatrixProgressEvent("response_received", cell=cell, request=request, response=response)
            )

        trace = await run_case(
            candidate_by_id[cell.candidate_id],
            cell.model_id,
            case,
            config,
            profile=profile,
            on_tool_boundary=on_tool_boundary,
            on_response=on_response,
        )
        completed += 1
        _emit_event(
            on_event,
            MatrixProgressEvent(
                "cell_finished",
                cell=cell,
                request=request,
                trace=trace,
                elapsed=perf_counter() - start,
                completion_index=completed,
                total=total,
            ),
        )
        return trace

    return task


async def run_matrix(
    config: EvalConfig, *, run_id: str | None = None, on_event: MatrixEventCallback | None = None
) -> EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Run the full matrix through one native pydantic-evals experiment."""
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    if os.environ.get("LOGFIRE_TOKEN"):
        from llm_sandbox_evals.logfire_config import configure_logfire

        configure_logfire()
    profile = resolve_profile(config.prompt_profile)
    candidates = prompts.load_candidates(config.candidates, config.prompt_profile)
    selected_cases = _select_cases(config.cases, config.homes)
    dataset = build_dataset(config, candidates, selected_cases, run_id)
    _emit_event(on_event, MatrixProgressEvent("matrix_started", total=len(dataset.cases)))
    return await dataset.evaluate(
        make_matrix_task(
            config,
            profile,
            {c.id: c for c in candidates},
            {c.id: c for c in selected_cases},
            total=len(dataset.cases),
            on_event=on_event,
        ),
        name="matrix",
        max_concurrency=max(1, config.concurrency),
        progress=False,
        retry_task=None,
    )


def _emit_event(callback: MatrixEventCallback | None, event: MatrixProgressEvent) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - presentation observers must not fail evaluation.
        return


def _presentation_request(request: str) -> str:
    return " ".join(request.split())[:240]


def overall_correct_rate(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> float:
    """Return the completed-cell correct rate."""
    for analysis in report.analyses:
        if isinstance(analysis, ScalarResult) and analysis.title == "Overall correct rate":
            return float(analysis.value)
    raise ValueError("report is missing the Overall correct rate analysis")


def matrix_summary_lines(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[str]:
    """Return concise machine-readable outcome summaries."""
    lines = [f"correct_rate: {overall_correct_rate(report):.3f}"]
    for analysis in report.analyses:
        if isinstance(analysis, TableResult) and analysis.title == "Candidate x model outcomes":
            lines.extend(
                f"{row[0]}/{row[1]}: correct_rate={row[6]} completed={row[5]} coverage={row[7]}"
                for row in analysis.rows
            )
    return lines


def _pair_rows(
    report_cases: Sequence[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]],
    candidate_ids: Sequence[str],
    model_ids: Sequence[str],
    categories: Sequence[str],
    prompt_sizes: dict[str, tuple[int, int]],
) -> list[_PairRow]:
    rows: list[_PairRow] = []
    for candidate_id in candidate_ids:
        for model_id in model_ids:
            selected = [
                c.output
                for c in report_cases
                if _metadata_str(c, "candidate_id") == candidate_id and _metadata_str(c, "model_id") == model_id
            ]
            counts = _counts(selected)
            diagnostics = [trace.diagnostics for trace in selected]
            category_rates = {
                category: _nullable_rate(
                    sum(t.outcome.state == "correct" for t in selected if t.category == category),
                    sum(t.outcome.state != "incomplete" for t in selected if t.category == category),
                )
                if any(t.category == category for t in selected)
                else None
                for category in categories
            }
            api_chars, authored_chars = prompt_sizes.get(candidate_id, (0, 0))
            rows.append(
                _PairRow(
                    candidate_id,
                    model_id,
                    counts["correct"],
                    counts["incorrect"],
                    counts["incomplete"],
                    counts["completed"],
                    _nullable_rate(counts["correct"], counts["completed"]),
                    _rate(counts["completed"], len(selected)),
                    _mean([d.tool_calls for d in diagnostics]),
                    _mean([d.failed_tool_calls for d in diagnostics]),
                    _mean([d.model_turns for d in diagnostics]),
                    _mean([d.elapsed_seconds for d in diagnostics]),
                    _optional_mean([_usage_value(d, "total_tokens") for d in diagnostics]),
                    _optional_mean([_usage_value(d, "cost") for d in diagnostics]),
                    category_rates,
                    api_chars,
                    authored_chars,
                )
            )
    return rows


def _ranking_rows(
    pair_rows: list[_PairRow], candidate_ids: Sequence[str], model_ids: Sequence[str], categories: Sequence[str]
) -> list[list[str | int | float | bool | None]]:
    baseline = max(
        1,
        next(
            (r.prompt_chars for r in pair_rows if r.candidate_id == "baseline" and r.prompt_chars),
            max((r.prompt_chars for r in pair_rows), default=1),
        ),
    )
    rendered: list[tuple[float, float, int, list[str | int | float | bool | None]]] = []
    for candidate_id in candidate_ids:
        rows = [r for r in pair_rows if r.candidate_id == candidate_id and r.model_id in model_ids]
        completed = sum(r.completed for r in rows)
        correct = sum(r.correct for r in rows)
        rate = _nullable_rate(correct, completed)
        model_rates = [r.correct_rate for r in rows if r.correct_rate is not None]
        min_model = min(model_rates, default=-1.0)
        prompt_chars = rows[0].prompt_chars if rows else 0
        categories_rate = [_nullable_mean([r.category_rates[c] for r in rows]) for c in categories]
        total_cells = sum(r.correct + r.incorrect + r.incomplete for r in rows)
        rendered.append(
            (
                rate if rate is not None else -1.0,
                min_model,
                prompt_chars,
                [
                    candidate_id,
                    correct,
                    sum(r.incorrect for r in rows),
                    sum(r.incomplete for r in rows),
                    completed,
                    _optional_round(rate),
                    _round(_rate(completed, total_cells)),
                    prompt_chars,
                    _round(prompt_chars / baseline),
                    *[_optional_round(v) for v in categories_rate],
                ],
            )
        )
    rendered.sort(key=lambda row: (-row[0], -row[1], row[2], str(row[3][0])))
    return [row[3] for row in rendered if row[0] >= 0]


def _categories(report_cases: Sequence[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]]) -> list[str]:
    return list(dict.fromkeys(_metadata_str(case, "category") for case in report_cases))


def _counts(traces: Sequence[CaseTrace]) -> dict[str, int]:
    return {
        "correct": sum(t.outcome.state == "correct" for t in traces),
        "incorrect": sum(t.outcome.state == "incorrect" for t in traces),
        "incomplete": sum(t.outcome.state == "incomplete" for t in traces),
        "completed": sum(t.outcome.state != "incomplete" for t in traces),
    }


def _metadata_str(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str) -> str:
    return str((report_case.metadata or {})[key])


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nullable_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float | int | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def _optional_mean(values: Sequence[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _nullable_mean(values: Sequence[float | None]) -> float | None:
    return _optional_mean(values)


def _usage_value(trace_diagnostics: object, key: str) -> float | None:
    usage = getattr(trace_diagnostics, "usage", None)
    value = usage.get(key) if isinstance(usage, dict) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def _round(value: float) -> float:
    return round(value, 3)
