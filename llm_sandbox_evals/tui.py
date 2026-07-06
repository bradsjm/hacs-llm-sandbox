"""Rich terminal rendering for eval progress and saved reports."""

import sys
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from llm_sandbox_evals.reports import candidate_rows, matrix_rows, score_categories
from llm_sandbox_evals.schema import CandidateModelScore


@dataclass(slots=True)
class _PairProgress:
    """Mutable terminal progress state for one candidate/model pair."""

    candidate_id: str
    model_id: str
    total: int
    concurrency: int
    completed: int = 0
    scores: list[float] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    case_marks: dict[int, str] = field(default_factory=dict)
    error: str | None = None


def stderr_console() -> Console:
    """Return a Rich console that writes human UI to stderr."""
    return Console(file=sys.stderr)


class LiveReporter:
    """Rich live progress reporter for ``harness.run_matrix``."""

    def __init__(self, console: Console) -> None:
        """Initialize the reporter with the stderr console owned by the CLI."""
        self._console = console
        self._states: dict[tuple[str, str], _PairProgress] = {}
        self._live: Live | None = None

    def __enter__(self) -> Self:
        """Start the live display."""
        self._live = Live(self._render(), console=self._console, refresh_per_second=8, transient=False)
        self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the live display after rendering its final state."""
        del exc_type, exc, traceback
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def pair_started(self, *, candidate_id: str, model_id: str, total: int, concurrency: int) -> None:
        """Record that one candidate/model pair is starting."""
        # State mutation point: create or reset the progress row for the active pair.
        self._states[(candidate_id, model_id)] = _PairProgress(
            candidate_id=candidate_id,
            model_id=model_id,
            total=total,
            concurrency=concurrency,
        )
        self._refresh()

    def case_finished(
        self,
        *,
        candidate_id: str,
        model_id: str,
        case_id: str,
        index: int,
        total: int,
        score: float,
        turns: int,
    ) -> None:
        """Record one completed case for the live display."""
        del case_id, total
        state = self._states[(candidate_id, model_id)]
        # State mutation point: completion order is concurrent, so preserve original case index for the mark strip.
        state.completed += 1
        state.scores.append(score)
        state.turns.append(turns)
        state.case_marks[index] = "[green]✓[/]" if score > 0.0 else "[red]✗[/]"
        self._refresh()

    def model_error(self, *, candidate_id: str, model_id: str, case_id: str, detail: str) -> None:
        """Record provider diagnostics for one candidate/model pair."""
        del case_id
        state = self._states[(candidate_id, model_id)]
        # Branch boundary: the first provider error marks the whole pair as degraded.
        state.error = detail.splitlines()[0] if detail else "model error"
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live display if it has been started."""
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Table | Panel:
        """Render the current live progress table."""
        if not self._states:
            return Panel("Starting eval matrix...", title="llm_sandbox eval progress")

        table = Table(title="llm_sandbox eval progress", expand=True)
        table.add_column("Candidate", overflow="fold")
        table.add_column("Model", overflow="fold")
        table.add_column("Cases", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Turns", justify="right")
        table.add_column("Progress", overflow="fold")
        table.add_column("Status", overflow="fold")
        for state in self._states.values():
            status, style = _pair_status(state)
            table.add_row(
                escape(state.candidate_id),
                escape(state.model_id),
                f"{state.completed}/{state.total}",
                _format_optional(_mean(state.scores)),
                _format_optional(_mean_ints(state.turns)),
                "".join(state.case_marks.get(index, "·") for index in range(state.total)),
                status,
                style=style,
            )
        return table


def render_leaderboard(
    console: Console,
    *,
    scores: list[CandidateModelScore],
    run_id: str,
    created_at: str,
    case_count: int,
    candidate_ids: list[str],
    model_ids: list[str],
) -> None:
    """Render saved leaderboard scores as Rich tables."""
    categories = score_categories(scores)
    candidates = candidate_rows(scores, candidate_ids, model_ids)
    matrix = matrix_rows(scores, candidate_ids, model_ids)
    best_mean = max((row.mean for row in candidates), default=0.0)

    console.rule(f"Eval leaderboard {escape(run_id)}")
    console.print(f"Created: {escape(created_at)}   Cases: {case_count}")

    candidate_table = Table(title="Candidate ranking", expand=True)
    candidate_table.add_column("Rank", justify="right")
    candidate_table.add_column("Candidate", overflow="fold")
    candidate_table.add_column("Mean", justify="right")
    candidate_table.add_column("MinModel", justify="right")
    candidate_table.add_column("Turns", justify="right")
    candidate_table.add_column("PromptChars", justify="right")
    candidate_table.add_column("SizeRatio", justify="right")
    for category in categories:
        candidate_table.add_column(category, justify="right")
    for rank, row in enumerate(candidates, start=1):
        style = "bold green" if row.mean == best_mean and candidates else ""
        candidate_table.add_row(
            str(rank),
            escape(row.candidate_id),
            _format_score(row.mean),
            _format_score(row.min_model),
            _format_score(row.mean_turns),
            str(row.prompt_chars),
            _format_score(row.size_ratio),
            *[_format_score(row.category_means[category]) for category in categories],
            style=style,
        )
    console.print(candidate_table)

    matrix_table = Table(title="Candidate x model means", expand=True)
    matrix_table.add_column("Candidate", overflow="fold")
    matrix_table.add_column("Model", overflow="fold")
    matrix_table.add_column("Mean", justify="right")
    matrix_table.add_column("Turns", justify="right")
    for matrix_row in matrix:
        matrix_table.add_row(
            escape(matrix_row.candidate_id),
            escape(matrix_row.model_id),
            _format_score(matrix_row.mean),
            _format_score(matrix_row.mean_turns),
        )
    console.print(matrix_table)


def render_failures(console: Console, rows: list[dict[str, object]]) -> None:
    """Render failures from saved ``results.jsonl`` rows."""
    failures = [row for row in rows if _is_failure(row)]
    if not failures:
        console.print("[green]No zero-score or required-check failures.[/]")
        return

    table = Table(title="Failures", expand=True)
    table.add_column("Case", overflow="fold")
    table.add_column("Candidate", overflow="fold")
    table.add_column("Model", overflow="fold")
    table.add_column("Category", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Failed required checks", overflow="fold")
    for row in failures:
        table.add_row(
            escape(_text(row, "case_id")),
            escape(_text(row, "candidate_id")),
            escape(_text(row, "model_id")),
            escape(_text(row, "category")),
            _format_score(_number(row, "score")),
            escape(", ".join(_failed_required_checks(row)) or "score=0.0"),
            style="red",
        )
    console.print(table)


def _pair_status(state: _PairProgress) -> tuple[str, str]:
    """Return display status text and row style for one progress row."""
    if state.error is not None:
        return f"model error: {escape(state.error)}", "red"
    if state.completed >= state.total:
        return "done", "green"
    return f"running (concurrency={state.concurrency})", ""


def _format_score(value: float) -> str:
    """Format one score value for terminal display."""
    return f"{value:.3f}"


def _format_optional(value: float | None) -> str:
    """Format a possible aggregate value for live-progress rows."""
    return "-" if value is None else f"{value:.2f}"


def _mean(values: list[float]) -> float | None:
    """Return a mean for a list of floats, or ``None`` when empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _mean_ints(values: list[int]) -> float | None:
    """Return a mean for a list of integers, or ``None`` when empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _is_failure(row: dict[str, object]) -> bool:
    """Return whether a compact result row should appear in the failure table."""
    return _number(row, "score") == 0.0 or bool(_failed_required_checks(row))


def _failed_required_checks(row: dict[str, object]) -> list[str]:
    """Return failed required check names from a compact result row."""
    checks = row.get("checks")
    if not isinstance(checks, list):
        return []
    names: list[str] = []
    for check in checks:
        if isinstance(check, dict) and _bool(check, "required") and not _bool(check, "passed"):
            names.append(_text(check, "name"))
    return names


def _text(row: dict[str, object], key: str) -> str:
    """Return one string field from a decoded JSON object."""
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _bool(row: dict[str, object], key: str) -> bool:
    """Return one boolean field from a decoded JSON object."""
    value = row.get(key)
    return value if isinstance(value, bool) else False


def _number(row: dict[str, object], key: str) -> float:
    """Return one numeric field from a decoded JSON object."""
    value = row.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
