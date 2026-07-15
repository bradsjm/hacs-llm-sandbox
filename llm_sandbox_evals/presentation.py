"""Pure eval result semantics plus separate runtime and saved-report projections."""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Self, get_args

from pydantic_evals.reporting import EvaluationReport

from llm_sandbox_evals import statistics as _statistics
from llm_sandbox_evals.experiment import (
    LanePhaseEvent,
    MatrixCellMeta,
    MatrixCellRef,
    MatrixProgressEvent,
)
from llm_sandbox_evals.phases import LanePhase
from llm_sandbox_evals.schema import CaseTrace, variant_label

if TYPE_CHECKING:
    from collections.abc import Iterable

PairAggregate = _statistics.PairAggregate
CategoryAggregate = _statistics.CategoryAggregate
ResultCounts = _statistics.ResultCounts
TaskRobustness = _statistics.TaskRobustness
pair_aggregates = _statistics.pair_aggregates
rate = _statistics.rate
result_counts = _statistics.result_counts

type JudgeStatus = Literal["not_requested", "available", "failed", "unavailable"]

# Keep the native evaluator classification bounded without projecting its exception message.
_JUDGE_FAILURE_TYPE_MAX_CHARS = 500

# Valid phase vocabulary derived from the backend LanePhase alias, never hand-maintained.
_VALID_PHASES: frozenset[str] = frozenset(get_args(LanePhase.__value__))
# Only these phases carry an authoritative tool name resolved after wrapper selection; the
# provider-driven preparing_tool_call name is model-supplied arbitrary text and is never retained.
_TOOL_NAME_PHASES: frozenset[str] = frozenset({"running_tool", "processing_tool_result"})


def effective_cause(trace: CaseTrace) -> str:
    """Return the operational or scored cause without mixing their contracts."""
    if trace.diagnostics.cap_exhausted:
        return "cap_exhausted"
    if trace.outcome.state == "incomplete":
        return trace.diagnostics.failure or "unknown"
    return trace.outcome.score_reason or "unknown"


def result_label(trace: CaseTrace) -> str:
    """Return one compact stable label for presentation surfaces."""
    return f"{trace.outcome.state}·{effective_cause(trace)}"


@dataclass(frozen=True, slots=True)
class JudgeFailure:
    """Bounded code-judge failure classification without native error details."""

    error_type: str | None
    message: str | None


@dataclass(frozen=True, slots=True)
class JudgePresentation:
    """Advisory code-judge state projected from one native report case."""

    status: JudgeStatus
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    failure: JudgeFailure | None = None


_NOT_REQUESTED_JUDGE = JudgePresentation("not_requested")


@dataclass(frozen=True, slots=True)
class JudgeSummary:
    """Report-only advisory judge counts and score projections."""

    requested: int
    available: int
    passed: int
    evaluator_failed: int
    unavailable: int
    mean_score: float | None
    pass_rate: float | None


@dataclass(frozen=True, slots=True)
class JudgeAggregate:
    """Advisory judge summary for one candidate and display variant."""

    candidate_id: str
    variant: str
    requested: int
    available: int
    passed: int
    evaluator_failed: int
    unavailable: int
    mean_score: float | None
    pass_rate: float | None


@dataclass(frozen=True, slots=True)
class JudgeAttention:
    """Safe report-only detail for one advisory judge result needing review."""

    case_id: str
    request_variant_id: str
    category: str
    candidate_id: str
    variant: str
    status: JudgeStatus
    score: float | None
    passed: bool | None
    reason: str | None
    failure_error_type: str | None


@dataclass(frozen=True, slots=True)
class PresentationCell:
    """Normalized per-cell facts used by every presentation surface."""

    case_id: str
    request_variant_id: str
    category: str
    candidate_id: str
    model_id: str
    variant: str
    trace: CaseTrace
    metrics: dict[str, float | int]
    judge: JudgePresentation = _NOT_REQUESTED_JUDGE


@dataclass(frozen=True, slots=True)
class OperationalIssueGroup:
    """One deterministic operational-failure row shared by runtime and reports."""

    count: int
    cause: str
    variant: str
    cells: tuple[str, ...]
    exception_type: str
    status_code: int | None
    provider_code: str | None
    provider_model: str | None
    message: str | None
    detail: str


type _OperationalIssueKey = tuple[
    str,
    str,
    str,
    int | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


def operational_issue_groups(cells: Iterable[PresentationCell]) -> tuple[OperationalIssueGroup, ...]:
    """Group incomplete cells by variant, cause, and structured operational identity."""
    grouped: dict[_OperationalIssueKey, list[PresentationCell]] = {}
    for cell in cells:
        trace = cell.trace
        # Branch boundary: cap exhaustion is a scored outcome, not an incomplete operational issue.
        if trace.outcome.state != "incomplete" or trace.diagnostics.cap_exhausted:
            continue
        execution_error = trace.execution_error
        exception_type = execution_error.exception_type if execution_error is not None else "unknown"
        status_code = execution_error.status_code if execution_error is not None else None
        provider_code = execution_error.provider_code if execution_error is not None else None
        provider_model = execution_error.provider_model if execution_error is not None else None
        message = execution_error.message if execution_error is not None else None
        provider_detail = execution_error.provider_detail if execution_error is not None else None
        provider_error = trace.provider_error
        key = (
            cell.variant,
            effective_cause(trace),
            exception_type,
            status_code,
            provider_code,
            provider_model,
            message,
            provider_detail,
            provider_error,
        )
        grouped.setdefault(key, []).append(cell)

    groups: list[OperationalIssueGroup] = []
    for (
        variant,
        cause,
        exception_type,
        status_code,
        provider_code,
        provider_model,
        message,
        provider_detail,
        provider_error,
    ), values in grouped.items():
        ordered_values = tuple(sorted(values, key=lambda cell: (cell.candidate_id, cell.case_id)))
        identities = tuple(f"{cell.candidate_id}/{cell.case_id}" for cell in ordered_values)
        detail = provider_error or provider_detail or message or "No error detail"
        groups.append(
            OperationalIssueGroup(
                count=len(values),
                cause=cause,
                variant=variant,
                cells=identities,
                exception_type=exception_type,
                status_code=status_code,
                provider_code=provider_code,
                provider_model=provider_model,
                message=message,
                detail=detail,
            )
        )
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                -group.count,
                group.cause,
                group.variant,
                group.exception_type,
                group.status_code is None,
                group.status_code or 0,
                group.provider_code or "",
                group.provider_model or "",
                group.message or "",
                group.detail,
                group.cells,
            ),
        )
    )


@dataclass(slots=True)
class RuntimeLane:
    """Safe active-lane metadata; only a phase label and optional tool name are retained."""

    cell: MatrixCellRef
    request: str
    started_at: float
    timeout: float
    max_tool_calls: int
    tools_used: int = 0
    # Latest observed execution phase and its safe tool name; never any model content.
    phase: str | None = "queued"
    tool_name: str | None = None


@dataclass(slots=True)
class PresentationState:
    """Mutable event projection used only while an eval is running."""

    total: int = 0
    started_at: float = field(default_factory=perf_counter)
    lanes: dict[MatrixCellRef, RuntimeLane] = field(default_factory=dict)
    completed: list[PresentationCell] = field(default_factory=list)
    # Cached projections of `completed`, recomputed in ingest() on the mutating (main) thread so
    # the Live auto-refresh thread only reads these frozen tuples instead of iterating the live
    # `completed` list (or its length) mid-append. `completed` is append-only via ingest().
    _groups_cache: tuple[OperationalIssueGroup, ...] = ()
    _aggregates_cache: tuple[PairAggregate, ...] = ()

    def ingest(self, event: MatrixProgressEvent, *, timeout: float, max_tool_calls: int) -> None:
        """Apply one lifecycle event without allowing presentation to affect evaluation."""
        if event.state == "matrix_started":
            self.total = event.total or 0
        elif event.state == "cell_started" and event.cell is not None:
            # State mutation point: a started cell owns one active display lane.
            self.lanes[event.cell] = RuntimeLane(
                event.cell, event.request or event.cell.case_id, perf_counter(), timeout, max_tool_calls
            )
        elif event.state == "tool_started" and event.cell is not None:
            lane = self.lanes.get(event.cell)
            if lane is not None:
                lane.tools_used += 1
        elif event.state == "cell_finished" and event.cell is not None and event.trace is not None:
            # State mutation point: a terminal trace leaves the active set exactly once.
            self.lanes.pop(event.cell, None)
            self.completed.append(_presentation_cell(event.cell, event.trace, {}))
            # Recompute projections here (main thread) so the Live refresh thread only reads frozen
            # tuples; this is the sole mutation point for `completed`, so once per cell is sufficient.
            self._groups_cache = operational_issue_groups(self.completed)
            self._aggregates_cache = pair_aggregates(self.completed)

    def ingest_phase(self, event: LanePhaseEvent) -> bool:
        """Project a phase onto its active lane; return whether visible state actually changed.

        Coalesces the harness's repeated identical stream-delta phases (e.g. thinking/responding)
        so the terminal only rebuilds the Live frame on a real transition, never per delta.
        """
        lane = self.lanes.get(event.cell)
        # Branch boundary: phases for drained or unknown cells never mutate state or report a change.
        if lane is None or event.phase not in _VALID_PHASES:
            return False
        # Trusted tool name: retained only from authoritative running/processing phases; every other
        # phase (including provider-supplied preparing_tool_call) resolves to None and is never stored.
        trusted_tool_name = event.tool_name if event.phase in _TOOL_NAME_PHASES else None
        # Change detection covers exactly the projected lane fields; duplicate phase/name pairs are no-ops.
        changed = lane.phase != event.phase or lane.tool_name != trusted_tool_name
        # State mutation point: store the latest projected phase and trusted tool name for the lane.
        lane.phase = event.phase
        lane.tool_name = trusted_tool_name
        return changed

    @property
    def counts(self) -> ResultCounts:
        """Return finished-cell results against the matrix's planned coverage denominator."""
        finished = result_counts(cell.trace for cell in self.completed)
        return ResultCounts(self.total, finished.correct, finished.incorrect, finished.incomplete)

    @property
    def operational_issues(self) -> Counter[str]:
        """Group incomplete cells by their true operational cause."""
        counts: Counter[str] = Counter()
        for group in self.operational_issue_groups:
            counts[group.cause] += group.count
        return counts

    @property
    def operational_issue_groups(self) -> tuple[OperationalIssueGroup, ...]:
        """Return operational-failure rows, recomputed in ingest() on the mutating thread."""
        return self._groups_cache

    @property
    def aggregates(self) -> tuple[PairAggregate, ...]:
        """Return candidate by variant aggregates, recomputed in ingest() on the mutating thread."""
        return self._aggregates_cache


@dataclass(frozen=True, slots=True)
class ReportPresentationModel:
    """Immutable reload-safe projection of a persisted native EvaluationReport."""

    cells: tuple[PresentationCell, ...]
    descriptor: dict[str, object]

    @classmethod
    def from_report(cls, report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> Self:
        """Build the saved-report presentation model without runtime state."""
        cells = tuple(
            _presentation_cell(
                report_case.inputs,
                report_case.output,
                dict(report_case.metrics),
                judge=_judge_presentation(report_case),
            )
            for report_case in report.cases
        )
        return cls(cells, dict(report.experiment_metadata or {}))

    @property
    def counts(self) -> ResultCounts:
        """Return report-wide counts."""
        return result_counts(cell.trace for cell in self.cells)

    @property
    def operational_issues(self) -> Counter[str]:
        """Group incomplete report cells by their effective cause."""
        counts: Counter[str] = Counter()
        for group in self.operational_issue_groups:
            counts[group.cause] += group.count
        return counts

    @property
    def operational_issue_groups(self) -> tuple[OperationalIssueGroup, ...]:
        """Return deterministic operational-failure rows for report cells."""
        return operational_issue_groups(self.cells)

    @property
    def aggregates(self) -> tuple[PairAggregate, ...]:
        """Return immutable candidate by variant aggregates."""
        return pair_aggregates(self.cells)

    @property
    def category_aggregates(self) -> tuple[CategoryAggregate, ...]:
        """Return immutable candidate by variant by category aggregates."""
        return _statistics.category_aggregates(self.cells)

    @property
    def canonical_counts(self) -> ResultCounts:
        """Return counts for canonical request variants only."""
        return result_counts(cell.trace for cell in _statistics.canonical_cells(self.cells))

    @property
    def paraphrase_counts(self) -> ResultCounts:
        """Return counts for non-canonical request variants only."""
        return result_counts(cell.trace for cell in _statistics.paraphrase_cells(self.cells))

    @property
    def canonical_quality_interval(self) -> tuple[float | None, float | None]:
        """Return the Wilson 95% interval over scored canonical cells."""
        counts = self.canonical_counts
        return _statistics.wilson_interval(counts.correct, counts.scored)

    @property
    def task_robustness(self) -> tuple[TaskRobustness, ...]:
        """Return task-level robustness across request variants."""
        return _statistics.task_robustness(self.cells)

    @property
    def judge_results(self) -> tuple[JudgePresentation, ...]:
        """Return the per-cell advisory judge projections in report order."""
        return tuple(cell.judge for cell in self.cells)

    @property
    def judge_requested(self) -> bool:
        """Return whether any report cell requested advisory judging."""
        return any(result.status != "not_requested" for result in self.judge_results)

    @property
    def judge_summary(self) -> JudgeSummary:
        """Return advisory judge counts, with unavailable rates for an empty projection."""
        return judge_summary(self.cells)

    @property
    def judge_aggregates(self) -> tuple[JudgeAggregate, ...]:
        """Return sorted advisory aggregates for candidate and display variant pairs."""
        return judge_aggregates(self.cells)

    @property
    def judge_needs_attention(self) -> tuple[JudgeAttention, ...]:
        """Return unbounded, safely projected advisory results needing review."""
        return judge_attention(self.cells)


def _presentation_cell(
    cell: MatrixCellRef,
    trace: CaseTrace,
    metrics: dict[str, float | int],
    *,
    judge: JudgePresentation = _NOT_REQUESTED_JUDGE,
) -> PresentationCell:
    """Normalize a cell and trace into the common presentation shape."""
    return PresentationCell(
        cell.case_id,
        cell.request_variant_id,
        trace.category,
        cell.candidate_id,
        cell.model_id,
        variant_label(trace.model_id, trace.reasoning_effort),
        trace,
        metrics,
        judge,
    )


def _judge_presentation(report_case: object) -> JudgePresentation:
    """Project native judge results without exposing evaluator error details."""
    metadata = getattr(report_case, "metadata", None)
    # Branch boundary: only the explicit native opt-in requests an advisory judge projection.
    if not isinstance(metadata, dict) or metadata.get("judge_enabled") is not True:
        return _NOT_REQUESTED_JUDGE

    failures = getattr(report_case, "evaluator_failures", ())
    # Branch boundary: unrelated evaluator failures never change the code-judge presentation state.
    matching_failure = next(
        (failure for failure in failures if getattr(failure, "name", None) == "code_quality_judge"),
        None,
    )
    if matching_failure is not None:
        return JudgePresentation(
            "failed",
            failure=JudgeFailure(
                _bounded_judge_failure_type(getattr(matching_failure, "error_type", None)),
                None,
            ),
        )

    scores = getattr(report_case, "scores", {})
    assertions = getattr(report_case, "assertions", {})
    score_result = scores.get("code_quality_score") if isinstance(scores, dict) else None
    pass_result = assertions.get("code_quality_pass") if isinstance(assertions, dict) else None
    score = _judge_score(getattr(score_result, "value", None))
    passed = _judge_pass(getattr(pass_result, "value", None))
    # Branch boundary: a requested judge is available only when both native outputs are valid.
    if score is None or passed is None:
        return JudgePresentation("unavailable")

    score_reason = _judge_reason(getattr(score_result, "reason", None))
    pass_reason = _judge_reason(getattr(pass_result, "reason", None))
    return JudgePresentation(
        "available",
        score=score,
        passed=passed,
        reason=score_reason if score_reason is not None else pass_reason,
    )


def _judge_score(value: object) -> float | None:
    """Return a finite normalized judge score, rejecting malformed native values."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    score = float(value)
    return score if isfinite(score) and 0.0 <= score <= 1.0 else None


def _judge_pass(value: object) -> bool | None:
    """Return a native boolean judge result without accepting integer coercion."""
    return value if isinstance(value, bool) else None


def _judge_reason(value: object) -> str | None:
    """Return only the native textual judge reason."""
    return value if isinstance(value, str) else None


def _bounded_judge_failure_type(value: object) -> str | None:
    """Return only the bounded native evaluator failure classification."""
    if not isinstance(value, str):
        return None
    return value[:_JUDGE_FAILURE_TYPE_MAX_CHARS]


def judge_summary(cells: Iterable[PresentationCell]) -> JudgeSummary:
    """Summarize only requested advisory judge results without touching deterministic scores."""
    results = tuple(cell.judge for cell in cells if cell.judge.status != "not_requested")
    available = tuple(result for result in results if result.status == "available")
    scores = tuple(result.score for result in available if result.score is not None)
    passed = sum(result.passed is True for result in available)
    # Branch boundary: no available results retain an absent rate and mean instead of fake zeros.
    pass_rate = passed / len(available) if available else None
    mean_score = sum(scores) / len(scores) if scores else None
    return JudgeSummary(
        requested=len(results),
        available=len(available),
        passed=passed,
        evaluator_failed=sum(result.status == "failed" for result in results),
        unavailable=sum(result.status == "unavailable" for result in results),
        mean_score=mean_score,
        pass_rate=pass_rate,
    )


def judge_aggregates(cells: Iterable[PresentationCell]) -> tuple[JudgeAggregate, ...]:
    """Group requested advisory judge results by candidate and display variant."""
    grouped: dict[tuple[str, str], list[PresentationCell]] = {}
    for cell in cells:
        # Branch boundary: unrequested cells do not create empty advisory aggregate rows.
        if cell.judge.status == "not_requested":
            continue
        # State mutation point: each requested cell contributes to one sorted report group.
        grouped.setdefault((cell.candidate_id, cell.variant), []).append(cell)

    aggregates: list[JudgeAggregate] = []
    for (candidate_id, variant), values in sorted(grouped.items()):
        summary = judge_summary(values)
        aggregates.append(
            JudgeAggregate(
                candidate_id=candidate_id,
                variant=variant,
                requested=summary.requested,
                available=summary.available,
                passed=summary.passed,
                evaluator_failed=summary.evaluator_failed,
                unavailable=summary.unavailable,
                mean_score=summary.mean_score,
                pass_rate=summary.pass_rate,
            )
        )
    return tuple(aggregates)


def judge_attention(cells: Iterable[PresentationCell]) -> tuple[JudgeAttention, ...]:
    """Project requested failed, unavailable, and false advisory judge results safely."""
    attention: list[JudgeAttention] = []
    for cell in cells:
        judge = cell.judge
        # Branch boundary: only explicit false judgments or judge execution gaps need attention.
        if judge.status not in {"failed", "unavailable"} and not (
            judge.status == "available" and judge.passed is False
        ):
            continue
        failure_error_type = judge.failure.error_type if judge.failure is not None else None
        attention.append(
            JudgeAttention(
                case_id=cell.case_id,
                request_variant_id=cell.request_variant_id,
                category=cell.category,
                candidate_id=cell.candidate_id,
                variant=cell.variant,
                status=judge.status,
                score=judge.score,
                passed=judge.passed,
                reason=judge.reason,
                failure_error_type=failure_error_type,
            )
        )

    # Branch boundary: evaluator gaps precede false scores; false scores are ordered from lowest up.
    priority = {"failed": 0, "unavailable": 1, "available": 2}
    return tuple(
        sorted(
            attention,
            key=lambda item: (
                priority[item.status],
                item.score if item.status == "available" and item.score is not None else 0.0,
                item.case_id,
                item.request_variant_id,
                item.category,
                item.candidate_id,
                item.variant,
                item.status,
                item.reason or "",
                item.failure_error_type or "",
            ),
        )
    )
