"""Native pydantic-evals experiment for the candidate x model x case matrix."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

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
from llm_sandbox_evals.schema import CaseTrace, CheckResult, EvalCase, PromptCandidate
from llm_sandbox_evals.scoring import is_incomplete, mean_score

type MatrixCellMeta = dict[str, str | int]
type ProgressCallback = Callable[[int, int, str, CaseTrace, float], None]
"""Per-cell progress reporter: ``(index, total, cell_name, trace, elapsed_seconds)``.

``index`` is the 1-based completion order (unique per cell) and ``elapsed_seconds``
covers the single ``run_case`` call. Invoked synchronously from the event loop."""

_SIZE_TIE_EPSILON = 0.005


@dataclass(frozen=True, slots=True)
class MatrixCellRef:
    """JSON-native reference stored in pydantic-evals case inputs."""

    case_id: str
    candidate_id: str
    model_id: str
    home: str
    category: str


@dataclass(slots=True)
class SandboxOutcome(Evaluator[MatrixCellRef, CaseTrace, MatrixCellMeta]):
    """Expose deterministic sandbox scoring as native pydantic-evals results."""

    def evaluate(
        self, ctx: EvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]
    ) -> dict[str, bool | str | EvaluationReason]:
        """Return the score plus stable labels/assertions for one matrix cell."""
        trace = ctx.output
        required_passed = all(not (check.required and not check.passed) for check in trace.checks)
        failure_kind = _failure_kind(trace.checks)
        return {
            "score": EvaluationReason(value=trace.score, reason=_summarize(trace.checks)),
            "required_gates_passed": EvaluationReason(
                value=required_passed,
                reason=_failed_names(trace.checks),
            ),
            "model_error": str(any(check.name == "model_error" for check in trace.checks)).lower(),
            "outcome": "passed" if required_passed else "failed",
            "failure_kind": failure_kind,
            "error_type": _error_type(trace.error),
        }


@dataclass(slots=True)
class CandidateMatrixReport(ReportEvaluator[MatrixCellRef, CaseTrace, MatrixCellMeta]):
    """Aggregate native case reports into candidate/model leaderboard analyses."""

    candidate_ids: list[str]
    model_ids: list[str]
    prompt_sizes: dict[str, tuple[int, int]]

    def evaluate(self, ctx: ReportEvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[ReportAnalysis]:
        """Return native analysis tables for the full experiment report."""
        complete_cases = _complete_cases(ctx.report.cases)
        categories = _categories(ctx.report.cases)
        pair_rows = _pair_rows(complete_cases, self.candidate_ids, self.model_ids, categories, self.prompt_sizes)
        ranking_rows = _ranking_rows(pair_rows, self.candidate_ids, self.model_ids, categories)
        overall = mean_score([_case_score(case) for case in complete_cases])
        incomplete_count = sum(1 for case in ctx.report.cases if is_incomplete(case.output.checks))
        return [
            TableResult(
                title="Candidate ranking",
                columns=["Candidate", "Mean", "MinModel", "ToolCalls", "PromptChars", "SizeRatio", *categories],
                rows=ranking_rows,
            ),
            TableResult(
                title="Candidate x model means",
                columns=["Candidate", "Model", "Mean", "ToolCalls"],
                rows=[
                    [row.candidate_id, row.model_id, _round(row.mean), _round(row.mean_tool_calls)]
                    for row in pair_rows
                ],
            ),
            ScalarResult(title="Overall mean score", value=_round(overall)),
            ScalarResult(title="Incomplete cells", value=incomplete_count),
        ]


@dataclass(frozen=True, slots=True)
class _PairRow:
    candidate_id: str
    model_id: str
    mean: float
    mean_tool_calls: float
    category_means: dict[str, float]
    api_prompt_chars: int
    prompt_chars: int


def build_dataset(
    config: EvalConfig,
    candidates: Sequence[PromptCandidate],
    selected_cases: Sequence[EvalCase],
    run_id: str,
) -> Dataset[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Build one native dataset containing every candidate/model/case matrix cell."""
    cases: list[Case[MatrixCellRef, CaseTrace, MatrixCellMeta]] = []
    for candidate in candidates:
        for model_id in config.models:
            for case in selected_cases:
                ref = MatrixCellRef(
                    case_id=case.id,
                    candidate_id=candidate.id,
                    model_id=model_id,
                    home=case.home,
                    category=case.category,
                )
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
                candidate_ids=[candidate.id for candidate in candidates],
                model_ids=list(config.models),
                prompt_sizes={candidate.id: prompts.candidate_prompt_sizes(candidate) for candidate in candidates},
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
    on_complete: ProgressCallback | None = None,
) -> Callable[[MatrixCellRef], Awaitable[CaseTrace]]:
    """Build the pydantic-evals task that preserves run_case snapshot/tool semantics."""
    completed = 0

    async def task(cell: MatrixCellRef) -> CaseTrace:
        nonlocal completed
        candidate = candidate_by_id[cell.candidate_id]
        case = case_by_id[cell.case_id]
        name = f"{candidate.id}/{cell.model_id}/{case.id}"
        start = perf_counter()
        trace = await run_case(candidate, cell.model_id, case, config, profile=profile)
        elapsed = perf_counter() - start
        # Branch boundary: increment is synchronous (no await) so each cell gets a
        # unique completion index despite concurrent matrix evaluation.
        completed += 1
        if on_complete is not None:
            on_complete(completed, total, name, trace, elapsed)
        return trace

    return task


async def run_matrix(
    config: EvalConfig,
    *,
    run_id: str | None = None,
    logfire_enabled: bool = False,
    on_complete: ProgressCallback | None = None,
) -> EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Run the full matrix through one native pydantic-evals experiment."""
    # Branch boundary: direct callers may omit a run id, but each cell still gets one for Logfire/report joins.
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    if logfire_enabled:
        from llm_sandbox_evals.logfire_config import configure_logfire

        configure_logfire()
    profile = resolve_profile(config.prompt_profile)
    candidates = prompts.load_candidates(config.candidates, config.prompt_profile)
    selected_cases = _select_cases(config.cases, config.homes)
    dataset = build_dataset(config, candidates, selected_cases, run_id)
    task = make_matrix_task(
        config,
        profile,
        {candidate.id: candidate for candidate in candidates},
        {case.id: case for case in selected_cases},
        total=len(dataset.cases),
        on_complete=on_complete,
    )
    return await dataset.evaluate(
        task,
        name="matrix",
        max_concurrency=max(1, config.concurrency),
        progress=False,
        retry_task=None,
    )


def overall_mean(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> float:
    """Read the native scalar analysis.

    Provider/infra failures (``model_error``) are excluded so an outage does not
    read as a candidate scoring near zero.
    """
    for analysis in report.analyses:
        if isinstance(analysis, ScalarResult) and analysis.title == "Overall mean score":
            return float(analysis.value)
    raise ValueError("report is missing the Overall mean score analysis")


def matrix_summary_lines(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[str]:
    """Return machine-readable per-candidate/model summary lines from native analyses."""
    lines = [f"overall_mean: {overall_mean(report):.3f}"]
    for analysis in report.analyses:
        if isinstance(analysis, TableResult) and analysis.title == "Candidate x model means":
            for row in analysis.rows:
                mean = row[2]
                turns = row[3]
                if isinstance(mean, int | float) and isinstance(turns, int | float):
                    lines.append(f"{row[0]}/{row[1]}: mean={mean:.3f} tool_calls={turns:.3f}")
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
            cases = [case for case in report_cases if _metadata_str(case, "candidate_id") == candidate_id]
            cases = [case for case in cases if _metadata_str(case, "model_id") == model_id]
            api_prompt_chars, prompt_chars = prompt_sizes.get(candidate_id, (0, 0))
            rows.append(
                _PairRow(
                    candidate_id=candidate_id,
                    model_id=model_id,
                    mean=mean_score([_case_score(case) for case in cases]),
                    mean_tool_calls=mean_score([float(case.output.tool_call_count) for case in cases]),
                    category_means={
                        category: mean_score(
                            [_case_score(case) for case in cases if _metadata_str(case, "category") == category]
                        )
                        for category in categories
                    },
                    api_prompt_chars=api_prompt_chars,
                    prompt_chars=prompt_chars,
                )
            )
    return rows


def _ranking_rows(
    pair_rows: list[_PairRow], candidate_ids: Sequence[str], model_ids: Sequence[str], categories: Sequence[str]
) -> list[list[str | int | float | bool | None]]:
    baseline_prompt_chars = next(
        (row.prompt_chars for row in pair_rows if row.candidate_id == "baseline" and row.prompt_chars),
        max((row.prompt_chars for row in pair_rows), default=1),
    )
    baseline_prompt_chars = max(1, baseline_prompt_chars)
    rows: list[tuple[float, int, float, list[str | int | float | bool | None]]] = []
    for candidate_id in candidate_ids:
        candidate_rows = [row for row in pair_rows if row.candidate_id == candidate_id and row.model_id in model_ids]
        mean = mean_score([row.mean for row in candidate_rows])
        min_model = min((row.mean for row in candidate_rows), default=0.0)
        mean_tool_calls = mean_score([row.mean_tool_calls for row in candidate_rows])
        prompt_chars = candidate_rows[0].prompt_chars if candidate_rows else 0
        api_prompt_chars = candidate_rows[0].api_prompt_chars if candidate_rows else 0
        category_means = {
            category: mean_score([row.category_means[category] for row in candidate_rows]) for category in categories
        }
        rendered: list[str | int | float | bool | None] = [
            candidate_id,
            _round(mean),
            _round(min_model),
            _round(mean_tool_calls),
            prompt_chars,
            _round(prompt_chars / baseline_prompt_chars),
            *[_round(category_means[category]) for category in categories],
        ]
        rows.append((mean, api_prompt_chars, min_model, rendered))
    rows.sort(key=lambda row: (-round(row[0] / _SIZE_TIE_EPSILON), row[1], -row[2]))
    return [row[3] for row in rows]


def _complete_cases(
    report_cases: Sequence[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]],
) -> list[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]]:
    """Return report cases that are not provider/infra failures.

    Incomplete cells (``model_error``) are excluded from candidate/model mean
    denominators so an outage does not distort quality scores.
    """
    return [case for case in report_cases if not is_incomplete(case.output.checks)]


def _categories(report_cases: Sequence[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]]) -> list[str]:
    categories: list[str] = []
    for case in report_cases:
        category = _metadata_str(case, "category")
        if category not in categories:
            categories.append(category)
    return categories


def _case_score(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> float:
    score = report_case.scores.get("score")
    return 0.0 if score is None else float(score.value)


def _metadata_str(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str) -> str:
    metadata = report_case.metadata or {}
    value = metadata[key]
    return str(value)


def _round(value: float) -> float:
    return round(value, 3)


def _summarize(checks: Sequence[CheckResult]) -> str:
    failed = _failed_names(checks)
    return "passed" if not failed else f"failed: {failed}"


def _failed_names(checks: Sequence[CheckResult]) -> str:
    return ", ".join(check.name for check in checks if check.required and not check.passed)


def _failure_kind(checks: Sequence[CheckResult]) -> str:
    """Return a low-cardinality first failing required check label."""
    for check in checks:
        if check.required and not check.passed:
            return check.name
    return "none"


def _error_type(error: str | None) -> str:
    """Extract the exception class from ``check_name: Type: detail`` trace errors."""
    if error is None:
        return "none"
    parts = error.split(":", 2)
    if len(parts) < 2:
        return "unknown"
    candidate = parts[1].strip()
    return candidate or "unknown"
