"""Shared numeric semantics and immutable eval aggregates."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

from llm_sandbox_evals.schema import CaseTrace


def nullable_rate(numerator: int, denominator: int) -> float | None:
    """Return a rate while preserving an unavailable denominator."""
    return numerator / denominator if denominator else None


def wilson_interval(correct: int, scored: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Return the Wilson score interval, or (None, None) when no cells were scored."""
    if scored == 0:
        return None, None
    # Branch boundary: preserve the specified conventional 95% z-score exactly.
    z = 1.96 if confidence == 0.95 else NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = correct / scored
    z_squared = z**2
    denominator = 1 + z_squared / scored
    center = (proportion + z_squared / (2 * scored)) / denominator
    margin = z * ((proportion * (1 - proportion) + z_squared / (4 * scored)) / scored) ** 0.5 / denominator
    return center - margin, center + margin


def rate(numerator: int, denominator: int) -> float:
    """Return a non-nullable rate for callers with a concrete zero fallback."""
    return numerator / denominator if denominator else 0.0


def mean(values: Sequence[float | int | None]) -> float:
    """Return the mean of available numeric values, or zero when none exist."""
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def optional_total(values: Sequence[float | int | None]) -> float | None:
    """Return the total known value while preserving unavailable metrics."""
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


def percentile(values: Sequence[float | int | None], fraction: float) -> float:
    """Return a linear-interpolated percentile for available values."""
    present = sorted(float(value) for value in values if value is not None)
    if not present:
        return 0.0
    position = (len(present) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(present) - 1)
    return present[lower] + (present[upper] - present[lower]) * (position - lower)


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
    def quality_rate(self) -> float | None:
        """Return correct/scored, or unavailable when no cells were scored."""
        return nullable_rate(self.correct, self.scored)

    @property
    def coverage_rate(self) -> float | None:
        """Return scored/total, or unavailable when no cells were planned."""
        return nullable_rate(self.scored, self.total)


def result_counts(traces: Iterable[CaseTrace]) -> ResultCounts:
    """Aggregate terminal traces using the public scored vocabulary."""
    values = tuple(traces)
    return ResultCounts(
        total=len(values),
        correct=sum(trace.outcome.state == "correct" for trace in values),
        incorrect=sum(trace.outcome.state == "incorrect" for trace in values),
        incomplete=sum(trace.outcome.state == "incomplete" for trace in values),
    )


class _AggregateCell(Protocol):
    """Structural input contract for presentation cells."""

    @property
    def candidate_id(self) -> str: ...

    @property
    def case_id(self) -> str: ...

    @property
    def request_variant_id(self) -> str: ...

    @property
    def category(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def variant(self) -> str: ...

    @property
    def trace(self) -> CaseTrace: ...

    @property
    def metrics(self) -> dict[str, float | int]: ...


@dataclass(frozen=True, slots=True)
class PairAggregate:
    """Candidate by resolved-model aggregate for comparison views."""

    candidate_id: str
    model_id: str
    variant: str
    counts: ResultCounts
    mean_calls: float
    mean_failed_calls: float
    mean_turns: float
    mean_elapsed: float
    p50_elapsed: float
    p95_elapsed: float
    total_tokens: float | None
    total_cost: float | None


@dataclass(frozen=True, slots=True)
class CategoryAggregate:
    """Candidate by variant by category result counts."""

    candidate_id: str
    variant: str
    category: str
    counts: ResultCounts


@dataclass(frozen=True, slots=True)
class TaskRobustness:
    """Request-variant pass count for one task, candidate, and model variant."""

    case_id: str
    candidate_id: str
    variant: str
    total_variants: int
    correct_variants: int
    all_passed: bool


def pair_aggregates(cells: Iterable[_AggregateCell]) -> tuple[PairAggregate, ...]:
    """Group normalized cells by candidate, model, and display variant."""
    grouped: dict[tuple[str, str, str], list[_AggregateCell]] = {}
    for cell in cells:
        # State mutation point: each normalized cell contributes to exactly one pair aggregate.
        grouped.setdefault((cell.candidate_id, cell.model_id, cell.variant), []).append(cell)
    aggregates: list[PairAggregate] = []
    for (candidate_id, model_id, variant), values in sorted(grouped.items()):
        calls = [_metric_value(cell, "tool_calls", cell.trace.diagnostics.tool_calls) for cell in values]
        failures = [
            _metric_value(cell, "failed_tool_calls", cell.trace.diagnostics.failed_tool_calls) for cell in values
        ]
        turns = [_metric_value(cell, "model_turns", cell.trace.diagnostics.model_turns) for cell in values]
        elapsed = [_metric_value(cell, "elapsed_seconds", cell.trace.diagnostics.elapsed_seconds) for cell in values]
        aggregates.append(
            PairAggregate(
                candidate_id=candidate_id,
                model_id=model_id,
                variant=variant,
                counts=result_counts(cell.trace for cell in values),
                mean_calls=mean(calls),
                mean_failed_calls=mean(failures),
                mean_turns=mean(turns),
                mean_elapsed=mean(elapsed),
                p50_elapsed=percentile(elapsed, 0.5),
                p95_elapsed=percentile(elapsed, 0.95),
                total_tokens=optional_total([_metric_or_usage(cell, "total_tokens") for cell in values]),
                total_cost=optional_total([_metric_or_usage(cell, "cost") for cell in values]),
            )
        )
    return tuple(aggregates)


def category_aggregates(cells: Iterable[_AggregateCell]) -> tuple[CategoryAggregate, ...]:
    """Group cells by candidate, variant, and category."""
    grouped: dict[tuple[str, str, str], list[_AggregateCell]] = {}
    for cell in cells:
        # State mutation point: each cell contributes to one category slice.
        grouped.setdefault((cell.candidate_id, cell.variant, cell.category), []).append(cell)
    return tuple(
        CategoryAggregate(candidate_id, variant, category, result_counts(cell.trace for cell in values))
        for (candidate_id, variant, category), values in sorted(grouped.items())
    )


def canonical_cells[CellT: _AggregateCell](cells: Iterable[CellT]) -> tuple[CellT, ...]:
    """Return only canonical-variant cells (request_variant_id == 'canonical')."""
    return tuple(cell for cell in cells if cell.request_variant_id == "canonical")


def paraphrase_cells[CellT: _AggregateCell](cells: Iterable[CellT]) -> tuple[CellT, ...]:
    """Return only non-canonical-variant cells."""
    return tuple(cell for cell in cells if cell.request_variant_id != "canonical")


def task_robustness(cells: Iterable[_AggregateCell]) -> tuple[TaskRobustness, ...]:
    """For each task/candidate/variant group, report how many request variants passed."""
    grouped: dict[tuple[str, str, str, str], list[_AggregateCell]] = {}
    for cell in cells:
        # State mutation point: model_id disambiguates display-label collisions while variant is rendered.
        grouped.setdefault((cell.case_id, cell.candidate_id, cell.model_id, cell.variant), []).append(cell)
    return tuple(
        TaskRobustness(
            case_id,
            candidate_id,
            variant,
            len(values),
            sum(cell.trace.outcome.state == "correct" for cell in values),
            bool(values) and all(cell.trace.outcome.state == "correct" for cell in values),
        )
        for (case_id, candidate_id, _model_id, variant), values in sorted(grouped.items())
    )


def _metric_value(cell: _AggregateCell, key: str, fallback: float | int | None) -> float | int | None:
    """Prefer native task metrics while retaining trace diagnostic fallbacks."""
    value = cell.metrics.get(key)
    return value if isinstance(value, int | float) and not isinstance(value, bool) else fallback


def _metric_or_usage(cell: _AggregateCell, key: str) -> float | None:
    """Read task metrics first, then the self-contained trace usage fallback."""
    metric = cell.metrics.get(key)
    if isinstance(metric, int | float) and not isinstance(metric, bool):
        return float(metric)
    usage = cell.trace.diagnostics.usage
    value = usage.get(key) if usage else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
