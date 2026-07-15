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
from pydantic_evals.reporting import EvaluationReport
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
from llm_sandbox_evals.statistics import nullable_rate

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
    request_variant_id: str
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
    """Expose scoring-v9 quality and operational labels as native eval results."""

    def evaluate(
        self, ctx: EvaluatorContext[MatrixCellRef, CaseTrace, MatrixCellMeta]
    ) -> dict[str, bool | str | EvaluationReason]:
        """Publish binary quality and provider classification labels."""
        trace = ctx.output
        return {
            "score": EvaluationReason(value=trace.outcome.score, reason=trace.outcome.score_reason),
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
        # Local import avoids reversing the existing presentation-to-runtime type dependency.
        from llm_sandbox_evals.presentation import ReportPresentationModel

        model = ReportPresentationModel.from_report(ctx.report)
        pairs = model.aggregates
        counts = model.counts
        baseline = max(
            1,
            self.prompt_sizes.get("baseline", (0, 0))[1]
            or max((authored for _, authored in self.prompt_sizes.values()), default=1),
        )
        ranking: list[tuple[float, float, int, list[str | int | float | bool | None]]] = []
        for candidate_id in self.candidate_ids:
            candidate_pairs = [
                pair for pair in pairs if pair.candidate_id == candidate_id and pair.model_id in self.model_ids
            ]
            candidate_counts = [pair.counts for pair in candidate_pairs]
            scored = sum(value.scored for value in candidate_counts)
            correct = sum(value.correct for value in candidate_counts)
            quality = nullable_rate(correct, scored)
            model_rates = [value.quality_rate for value in candidate_counts if value.quality_rate is not None]
            prompt_chars = self.prompt_sizes.get(candidate_id, (0, 0))[1]
            total = sum(value.total for value in candidate_counts)
            ranking.append(
                (
                    quality if quality is not None else -1.0,
                    min(model_rates, default=-1.0),
                    prompt_chars,
                    [
                        candidate_id,
                        correct,
                        sum(value.incorrect for value in candidate_counts),
                        sum(value.incomplete for value in candidate_counts),
                        scored,
                        _optional_round(quality),
                        _optional_round(nullable_rate(scored, total)),
                        prompt_chars,
                        _round(prompt_chars / baseline),
                    ],
                )
            )
        ranking.sort(key=lambda row: (-row[0], -row[1], row[2], str(row[3][0])))
        analyses: list[ReportAnalysis] = [
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
                rows=[row[3] for row in ranking if row[0] >= 0],
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
                        row.counts.correct,
                        row.counts.incorrect,
                        row.counts.incomplete,
                        row.counts.scored,
                        _optional_round(row.counts.quality_rate),
                        _optional_round(row.counts.coverage_rate),
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
            ScalarResult(title="Correct cells", value=counts.correct),
            ScalarResult(title="Incorrect cells", value=counts.incorrect),
            ScalarResult(title="Incomplete cells", value=counts.incomplete),
            ScalarResult(title="Scored cells", value=counts.scored),
        ]
        # Branch boundary: pydantic-evals scalar analyses cannot encode None, so unavailable rates are omitted.
        if counts.quality_rate is not None:
            analyses.append(ScalarResult(title="Quality rate", value=_round(counts.quality_rate)))
        if counts.coverage_rate is not None:
            analyses.append(ScalarResult(title="Coverage rate", value=_round(counts.coverage_rate)))
        return analyses


def build_dataset(
    config: EvalConfig, candidates: Sequence[PromptCandidate], selected_cases: Sequence[EvalCase], run_id: str
) -> Dataset[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Build one native dataset containing every candidate/model/task/request-variant cell."""
    cases: list[Case[MatrixCellRef, CaseTrace, MatrixCellMeta]] = []
    for candidate in candidates:
        for model_id in config.models:
            for case in selected_cases:
                for request_variant in case.requests:
                    ref = MatrixCellRef(
                        case.id,
                        request_variant.id,
                        candidate.id,
                        model_id,
                        case.home,
                        config.reasoning_effort,
                        config.temperature,
                    )
                    metadata: MatrixCellMeta = {
                        "run_id": run_id,
                        "case_id": ref.case_id,
                        "request_variant_id": ref.request_variant_id,
                        "candidate_id": ref.candidate_id,
                        "model_id": ref.model_id,
                        "home": ref.home,
                        "reasoning_effort": ref.reasoning_effort,
                        "temperature": ref.temperature,
                        "variant_label": variant_label(ref.model_id, ref.reasoning_effort),
                    }
                    cases.append(
                        Case(
                            name=f"{candidate.id}/{model_id}/{case.id}/{request_variant.id}",
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
        request_variant = next(variant for variant in case.requests if variant.id == cell.request_variant_id)
        request = _presentation_request(request_variant.text)
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
            request_variant,
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


def overall_correct_rate(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> float | None:
    """Return the scored-cell quality rate retained under the legacy helper name."""
    from llm_sandbox_evals.presentation import ReportPresentationModel

    return ReportPresentationModel.from_report(report).counts.quality_rate


def matrix_summary_lines(report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> list[str]:
    """Return concise machine-readable outcome summaries."""
    from llm_sandbox_evals.presentation import ReportPresentationModel

    model = ReportPresentationModel.from_report(report)
    counts = model.counts
    lines = [
        f"quality_rate: {_machine_rate(counts.quality_rate)}",
        f"coverage_rate: {_machine_rate(counts.coverage_rate)}",
        f"scored: {counts.scored}",
    ]
    lines.extend(
        f"{aggregate.candidate_id}/{aggregate.variant}: "
        f"quality_rate={_machine_rate(aggregate.counts.quality_rate)} "
        f"scored={aggregate.counts.scored} coverage_rate={_machine_rate(aggregate.counts.coverage_rate)}"
        for aggregate in model.aggregates
    )
    return lines


def _machine_rate(value: float | None) -> str:
    """Render one machine-summary rate without turning unavailable into zero."""
    return "—" if value is None else f"{value:.3f}"


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
