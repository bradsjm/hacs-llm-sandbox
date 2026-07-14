"""Pure eval result semantics plus separate runtime and saved-report projections."""

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Self, get_args

from pydantic_evals.reporting import EvaluationReport

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


def rate(numerator: int, denominator: int) -> float:
    """Return a safe display rate."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class ResultCounts:
    """Count vocabulary shared by runtime and saved-report views."""

    total: int
    correct: int
    incorrect: int
    incomplete: int

    @property
    def scored(self) -> int:
        """Return cells whose action outcome is scoreable."""
        return self.correct + self.incorrect

    @property
    def quality_rate(self) -> float:
        """Return correct/scored."""
        return rate(self.correct, self.scored)

    @property
    def coverage_rate(self) -> float:
        """Return scored/total."""
        return rate(self.scored, self.total)


def result_counts(traces: Iterable[CaseTrace]) -> ResultCounts:
    """Aggregate terminal traces using the public scored vocabulary."""
    values = tuple(traces)
    return ResultCounts(
        total=len(values),
        correct=sum(trace.outcome.state == "correct" for trace in values),
        incorrect=sum(trace.outcome.state == "incorrect" for trace in values),
        incomplete=sum(trace.outcome.state == "incomplete" for trace in values),
    )


@dataclass(frozen=True, slots=True)
class PresentationCell:
    """Normalized per-cell facts used by every presentation surface."""

    case_id: str
    candidate_id: str
    model_id: str
    variant: str
    trace: CaseTrace
    metrics: dict[str, float | int]


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


@dataclass(frozen=True, slots=True)
class PairAggregate:
    """Candidate by resolved-variant aggregate for compact comparison views."""

    candidate_id: str
    variant: str
    counts: ResultCounts
    mean_calls: float
    mean_failed_calls: float
    mean_elapsed: float
    total_tokens: float | None
    total_cost: float | None


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


def pair_aggregates(cells: Iterable[PresentationCell]) -> tuple[PairAggregate, ...]:
    """Group normalized cells by candidate and display variant."""
    grouped: dict[tuple[str, str], list[PresentationCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.candidate_id, cell.variant), []).append(cell)
    aggregates: list[PairAggregate] = []
    for (candidate_id, variant), values in sorted(grouped.items()):
        traces = [cell.trace for cell in values]
        calls = [float(cell.metrics.get("tool_calls", cell.trace.diagnostics.tool_calls)) for cell in values]
        failures = [
            float(cell.metrics.get("failed_tool_calls", cell.trace.diagnostics.failed_tool_calls)) for cell in values
        ]
        elapsed = [
            float(cell.metrics.get("elapsed_seconds", cell.trace.diagnostics.elapsed_seconds or 0.0))
            for cell in values
        ]
        tokens = [_metric_or_usage(cell, "total_tokens") for cell in values]
        costs = [_metric_or_usage(cell, "cost") for cell in values]
        aggregates.append(
            PairAggregate(
                candidate_id,
                variant,
                result_counts(traces),
                sum(calls) / len(calls) if calls else 0.0,
                sum(failures) / len(failures) if failures else 0.0,
                sum(elapsed) / len(elapsed) if elapsed else 0.0,
                sum(value for value in tokens if value is not None)
                if any(value is not None for value in tokens)
                else None,
                sum(value for value in costs if value is not None)
                if any(value is not None for value in costs)
                else None,
            )
        )
    return tuple(aggregates)


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


def _metric_or_usage(cell: PresentationCell, name: str) -> float | None:
    value = cell.metrics.get(name)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    usage = cell.trace.diagnostics.usage
    fallback = usage.get(name) if usage else None
    return float(fallback) if isinstance(fallback, int | float) and not isinstance(fallback, bool) else None


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
    phase: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class PresentationState:
    """Mutable event projection used only while an eval is running."""

    total: int = 0
    started_at: float = field(default_factory=perf_counter)
    lanes: dict[MatrixCellRef, RuntimeLane] = field(default_factory=dict)
    completed: list[PresentationCell] = field(default_factory=list)
    # Sticky for the run: turns True only once a real thinking phase is seen on an active lane.
    activity_enabled: bool = False
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
        # Branch boundary: phases for drained or unknown cells never mutate state or activate.
        if lane is None or event.phase not in _VALID_PHASES:
            return False
        # Trusted tool name: retained only from authoritative running/processing phases; every other
        # phase (including provider-supplied preparing_tool_call) resolves to None and is never stored.
        trusted_tool_name = event.tool_name if event.phase in _TOOL_NAME_PHASES else None
        # Branch boundary: a real thinking phase is the only trigger for the sticky Activity column.
        activates = event.phase == "thinking" and not self.activity_enabled
        # Change detection over exactly the projected fields: lane phase, trusted tool name, or the
        # False->True sticky flip. A duplicate phase/name after activation reports no change.
        changed = lane.phase != event.phase or lane.tool_name != trusted_tool_name or activates
        # State mutation point: store the latest projected phase and trusted tool name for the lane.
        lane.phase = event.phase
        lane.tool_name = trusted_tool_name
        # State mutation point: flip the sticky column reveal exactly once, on first thinking.
        if activates:
            self.activity_enabled = True
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


def _presentation_cell(cell: MatrixCellRef, trace: CaseTrace, metrics: dict[str, float | int]) -> PresentationCell:
    """Normalize a cell and trace into the common presentation shape."""
    return PresentationCell(
        cell.case_id,
        cell.candidate_id,
        cell.model_id,
        variant_label(trace.model_id, trace.reasoning_effort),
        trace,
        metrics,
    )
