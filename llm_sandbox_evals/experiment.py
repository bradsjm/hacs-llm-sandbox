"""Native pydantic-evals experiment for the candidate x model x case matrix."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from time import perf_counter
from typing import Literal

from custom_components.llm_sandbox.llm_api.prompts import PromptProfile, resolve_profile
from pydantic_evals import Case, Dataset, increment_eval_metric
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
from llm_sandbox_evals.phases import LanePhase, PhaseObservation
from llm_sandbox_evals.schema import (
    CaseTrace,
    EvalCase,
    ModelDescriptor,
    PromptCandidate,
    RunDescriptor,
    variant_label,
)

type MatrixCellMeta = dict[str, str | int | float | None]
type MatrixEventCallback = Callable[[MatrixProgressEvent], None]
type LanePhaseCallback = Callable[[LanePhaseEvent], None]
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
    reasoning_effort: str | None = None
    temperature: float | None = None


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


@dataclass(frozen=True, slots=True)
class LanePhaseEvent:
    """Payload-free phase observation associated with one matrix cell."""

    cell: MatrixCellRef
    phase: LanePhase
    tool_name: str | None = None


@dataclass(slots=True)
class SandboxOutcome(Evaluator[MatrixCellRef, CaseTrace, MatrixCellMeta]):
    """Expose scoring-v6 quality and operational labels as native eval results."""

    def evaluate(
        self, ctx: EvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]
    ) -> dict[str, bool | str | EvaluationReason]:
        """Publish binary quality and provider classification labels."""
        trace = ctx.output
        return {
            "score": EvaluationReason(value=trace.outcome.score, reason=trace.outcome.action_reason),
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
        pairs = _pair_rows(cases, self.candidate_ids, self.model_ids, self.prompt_sizes)
        ranking = _ranking_rows(pairs, self.candidate_ids, self.model_ids)
        counts = _counts([case.output for case in cases])
        return [
            TableResult(
                title="Candidate ranking",
                columns=[
                    "Candidate",
                    "Correct",
                    "Incorrect",
                    "Incomplete",
                    "Scored",
                    "Quality rate",
                    "Coverage rate",
                    "PromptChars",
                    "SizeRatio",
                ],
                rows=ranking,
            ),
            TableResult(
                title="Candidate x model outcomes",
                columns=[
                    "Candidate",
                    "Variant",
                    "Correct",
                    "Incorrect",
                    "Incomplete",
                    "Scored",
                    "Quality rate",
                    "Coverage rate",
                    "Calls",
                    "FailedCalls",
                    "Turns",
                    "Elapsed",
                    "P50 elapsed",
                    "P95 elapsed",
                    "Tokens",
                    "Cost",
                ],
                rows=[
                    [
                        row.candidate_id,
                        row.variant,
                        row.correct,
                        row.incorrect,
                        row.incomplete,
                        row.scored,
                        _optional_round(row.quality_rate),
                        _round(row.coverage_rate),
                        _round(row.mean_calls),
                        _round(row.mean_failed_calls),
                        _round(row.mean_turns),
                        _round(row.mean_elapsed),
                        _round(row.p50_elapsed),
                        _round(row.p95_elapsed),
                        _render_optional(row.total_tokens),
                        _render_optional(row.total_cost),
                    ]
                    for row in pairs
                ],
            ),
            ScalarResult(title="Quality rate", value=_round(_rate(counts["correct"], counts["scored"]))),
            ScalarResult(title="Correct cells", value=counts["correct"]),
            ScalarResult(title="Incorrect cells", value=counts["incorrect"]),
            ScalarResult(title="Incomplete cells", value=counts["incomplete"]),
            ScalarResult(title="Scored cells", value=counts["scored"]),
            ScalarResult(title="Coverage rate", value=_round(_rate(counts["scored"], len(cases)))),
        ]


@dataclass(frozen=True, slots=True)
class _PairRow:
    candidate_id: str
    model_id: str
    variant: str
    correct: int
    incorrect: int
    incomplete: int
    scored: int
    quality_rate: float | None
    coverage_rate: float
    mean_calls: float
    mean_failed_calls: float
    mean_turns: float
    mean_elapsed: float
    p50_elapsed: float
    p95_elapsed: float
    total_tokens: float | None
    total_cost: float | None
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
                ref = MatrixCellRef(
                    case.id,
                    candidate.id,
                    model_id,
                    case.home,
                    config.reasoning_effort,
                    config.temperature,
                )
                metadata: MatrixCellMeta = {
                    "run_id": run_id,
                    "case_id": ref.case_id,
                    "candidate_id": ref.candidate_id,
                    "model_id": ref.model_id,
                    "home": ref.home,
                    "reasoning_effort": ref.reasoning_effort,
                    "temperature": ref.temperature,
                    "variant_label": variant_label(ref.model_id, ref.reasoning_effort),
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
    on_phase: LanePhaseCallback | None = None,
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

        def on_phase_observation(observation: PhaseObservation) -> None:
            _emit_phase_event(on_phase, LanePhaseEvent(cell, observation.phase, observation.tool_name))

        trace = await run_case(
            candidate_by_id[cell.candidate_id],
            cell.model_id,
            case,
            config,
            profile=profile,
            on_tool_boundary=on_tool_boundary,
            on_response=on_response,
            on_phase=on_phase_observation,
        )
        _record_trace_metrics(trace)
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
    config: EvalConfig,
    *,
    run_id: str | None = None,
    descriptor: RunDescriptor | None = None,
    on_event: MatrixEventCallback | None = None,
    on_phase: LanePhaseCallback | None = None,
) -> EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Run the full matrix through one native pydantic-evals experiment."""
    if descriptor is not None:
        # Branch boundary: the caller-owned descriptor is the single lifecycle identity for a CLI run.
        run_id = descriptor.run_id
    elif run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    if os.environ.get("LOGFIRE_TOKEN"):
        from llm_sandbox_evals.logfire_config import configure_logfire

        configure_logfire()
    profile = resolve_profile(config.prompt_profile)
    candidates = prompts.load_candidates(config.candidates, config.prompt_profile)
    selected_cases = _select_cases(config.cases, config.homes)
    dataset = build_dataset(config, candidates, selected_cases, run_id)
    _emit_event(on_event, MatrixProgressEvent("matrix_started", total=len(dataset.cases)))
    if descriptor is None:
        descriptor = build_run_descriptor(config, run_id, selected_cases)
    return await dataset.evaluate(
        make_matrix_task(
            config,
            profile,
            {c.id: c for c in candidates},
            {c.id: c for c in selected_cases},
            total=len(dataset.cases),
            on_event=on_event,
            on_phase=on_phase,
        ),
        name="matrix",
        max_concurrency=max(1, config.concurrency),
        progress=False,
        retry_task=None,
        metadata=_descriptor_metadata(descriptor),
    )


def _emit_event(callback: MatrixEventCallback | None, event: MatrixProgressEvent) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - presentation observers must not fail evaluation.
        return


def _emit_phase_event(callback: LanePhaseCallback | None, event: LanePhaseEvent) -> None:
    """Isolate payload-free phase observers from matrix execution."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - phase observers must not fail evaluation.
        return


def _presentation_request(request: str) -> str:
    return " ".join(request.split())[:240]


def overall_correct_rate(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> float:
    """Return the scored-cell quality rate retained under the legacy helper name."""
    for analysis in report.analyses:
        if isinstance(analysis, ScalarResult) and analysis.title == "Quality rate":
            return float(analysis.value)
    raise ValueError("report is missing the Quality rate analysis")


def matrix_summary_lines(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[str]:
    """Return concise machine-readable outcome summaries."""
    counts = _counts([case.output for case in report.cases])
    lines = [
        f"quality_rate: {overall_correct_rate(report):.3f}",
        f"coverage_rate: {_rate(counts['scored'], counts['total']):.3f}",
        f"scored: {counts['scored']}",
    ]
    for analysis in report.analyses:
        if isinstance(analysis, TableResult) and analysis.title == "Candidate x model outcomes":
            lines.extend(
                f"{row[0]}/{row[1]}: quality_rate={row[6]} scored={row[5]} coverage_rate={row[7]}"
                for row in analysis.rows
            )
    return lines


def _pair_rows(
    report_cases: Sequence[ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]],
    candidate_ids: Sequence[str],
    model_ids: Sequence[str],
    prompt_sizes: dict[str, tuple[int, int]],
) -> list[_PairRow]:
    rows: list[_PairRow] = []
    for candidate_id in candidate_ids:
        for model_id in model_ids:
            selected = [
                c
                for c in report_cases
                if _metadata_str(c, "candidate_id") == candidate_id and _metadata_str(c, "model_id") == model_id
            ]
            traces = [case.output for case in selected]
            counts = _counts(traces)
            diagnostics = [trace.diagnostics for trace in traces]
            api_chars, authored_chars = prompt_sizes.get(candidate_id, (0, 0))
            rows.append(
                _PairRow(
                    candidate_id,
                    model_id,
                    variant_label(
                        model_id, _metadata_str_or_none(selected[0], "reasoning_effort") if selected else None
                    ),
                    counts["correct"],
                    counts["incorrect"],
                    counts["incomplete"],
                    counts["scored"],
                    _nullable_rate(counts["correct"], counts["scored"]),
                    _rate(counts["scored"], len(selected)),
                    _mean(
                        [
                            _metric_value(case, "tool_calls", d.tool_calls)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ]
                    ),
                    _mean(
                        [
                            _metric_value(case, "failed_tool_calls", d.failed_tool_calls)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ]
                    ),
                    _mean(
                        [
                            _metric_value(case, "model_turns", d.model_turns)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ]
                    ),
                    _mean(
                        [
                            _metric_value(case, "elapsed_seconds", d.elapsed_seconds)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ]
                    ),
                    _percentile(
                        [
                            _metric_value(case, "elapsed_seconds", d.elapsed_seconds)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ],
                        0.5,
                    ),
                    _percentile(
                        [
                            _metric_value(case, "elapsed_seconds", d.elapsed_seconds)
                            for case, d in zip(selected, diagnostics, strict=True)
                        ],
                        0.95,
                    ),
                    _optional_total([_metric_or_usage(case, "total_tokens") for case in selected]),
                    _optional_total([_metric_or_usage(case, "cost") for case in selected]),
                    api_chars,
                    authored_chars,
                )
            )
    return rows


def _ranking_rows(
    pair_rows: list[_PairRow], candidate_ids: Sequence[str], model_ids: Sequence[str]
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
        scored = sum(r.scored for r in rows)
        correct = sum(r.correct for r in rows)
        rate = _nullable_rate(correct, scored)
        model_rates = [r.quality_rate for r in rows if r.quality_rate is not None]
        min_model = min(model_rates, default=-1.0)
        prompt_chars = rows[0].prompt_chars if rows else 0
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
                    scored,
                    _optional_round(rate),
                    _round(_rate(scored, total_cells)),
                    prompt_chars,
                    _round(prompt_chars / baseline),
                ],
            )
        )
    rendered.sort(key=lambda row: (-row[0], -row[1], row[2], str(row[3][0])))
    return [row[3] for row in rendered if row[0] >= 0]


def _counts(traces: Sequence[CaseTrace]) -> dict[str, int]:
    return {
        "correct": sum(t.outcome.state == "correct" for t in traces),
        "incorrect": sum(t.outcome.state == "incorrect" for t in traces),
        "incomplete": sum(t.outcome.state == "incomplete" for t in traces),
        "scored": sum(t.outcome.state != "incomplete" for t in traces),
        "total": len(traces),
    }


def _metadata_str(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str) -> str:
    return str((report_case.metadata or {})[key])


def _metadata_str_or_none(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str) -> str | None:
    """Read nullable metadata without stringifying an absent resolved setting."""
    value = (report_case.metadata or {}).get(key)
    return str(value) if value is not None else None


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


def _optional_total(values: Sequence[float | int | None]) -> float | None:
    """Return the total known value, preserving unavailable provider metrics."""
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


def _percentile(values: Sequence[float | int | None], fraction: float) -> float:
    """Return a linear-interpolated percentile for elapsed timing diagnostics."""
    present = sorted(float(value) for value in values if value is not None)
    if not present:
        return 0.0
    position = (len(present) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(present) - 1)
    return present[lower] + (present[upper] - present[lower]) * (position - lower)


def _metric_value(
    report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str, fallback: float | int | None
) -> float | int | None:
    """Prefer native task metrics while retaining trace diagnostics for old provider values."""
    value = report_case.metrics.get(key)
    return value if isinstance(value, int | float) and not isinstance(value, bool) else fallback


def _metric_or_usage(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta], key: str) -> float | None:
    """Read persisted task metrics first, then the self-contained trace usage fallback."""
    metric = report_case.metrics.get(key)
    if isinstance(metric, int | float) and not isinstance(metric, bool):
        return float(metric)
    usage = report_case.output.diagnostics.usage
    value = usage.get(key) if isinstance(usage, dict) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _record_trace_metrics(trace: CaseTrace) -> None:
    """Record per-cell native metrics while the pydantic-evals task context is active."""
    diagnostics = trace.diagnostics
    for name, value in {
        "tool_calls": diagnostics.tool_calls,
        "successful_tool_calls": diagnostics.successful_tool_calls,
        "failed_tool_calls": diagnostics.failed_tool_calls,
        "model_turns": diagnostics.model_turns,
        "elapsed_seconds": diagnostics.elapsed_seconds,
        "total_tokens": _usage_value(diagnostics, "total_tokens"),
    }.items():
        # Branch boundary: unavailable provider usage must remain unavailable, not become zero.
        if isinstance(value, int | float) and not isinstance(value, bool):
            increment_eval_metric(name, value)


def _usage_value(trace_diagnostics: object, key: str) -> float | None:
    """Read one optional numeric usage field from diagnostics."""
    usage = getattr(trace_diagnostics, "usage", None)
    value = usage.get(key) if isinstance(usage, dict) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def build_run_descriptor(config: EvalConfig, run_id: str, selected_cases: Sequence[EvalCase]) -> RunDescriptor:
    """Build the native reload-safe run configuration snapshot."""
    return RunDescriptor(
        run_id,
        datetime.now(UTC).isoformat(),
        tuple(
            ModelDescriptor(
                model_id,
                config.reasoning_effort,
                config.temperature,
                variant_label(model_id, config.reasoning_effort),
            )
            for model_id in config.models
        ),
        tuple(config.candidates),
        tuple(case.id for case in selected_cases),
        config.prompt_profile,
        config.concurrency,
        config.model_timeout,
        config.max_tool_calls,
    )


def _descriptor_metadata(descriptor: RunDescriptor) -> dict[str, object]:
    """Convert the frozen descriptor to the JSON-native Dataset metadata contract."""
    return {
        "run_id": descriptor.run_id,
        "created_at": descriptor.created_at,
        "models": [
            {
                "model_id": model.model_id,
                "reasoning_effort": model.reasoning_effort,
                "temperature": model.temperature,
                "variant_label": model.variant_label,
            }
            for model in descriptor.models
        ],
        "candidates": list(descriptor.candidates),
        "cases": list(descriptor.cases),
        "prompt_profile": descriptor.prompt_profile,
        "concurrency": descriptor.concurrency,
        "model_timeout": descriptor.model_timeout,
        "max_tool_calls": descriptor.max_tool_calls,
    }


def _optional_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def _render_optional(value: float | None) -> float | str:
    """Render unavailable usage and cost as an explicit em dash."""
    return "—" if value is None else _round(value)


def _round(value: float) -> float:
    return round(value, 3)
