"""Safe stderr presentation for eval matrix lifecycle events."""

from time import perf_counter
from typing import Self

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixProgressEvent
from llm_sandbox_evals.presentation import PresentationState, result_label, variant_label

_ACTIVE = "#38b6ca"
_SUCCESS = "#55c97c"
_WARNING = "#d9a514"
_ERROR = "#f2705f"
_TRACK = "grey37"
_PERCENT = "#b8a1e8"
_RECENT_RESULTS = 10
_VARIANT_WIDTH = 22

# Outcome state → (glyph, color). Drives every colored result surface consistently.
_OUTCOME_STYLES: dict[str, tuple[str, str]] = {
    "correct": ("\u2713", _SUCCESS),
    "incorrect": ("\u2717", _WARNING),
    "incomplete": ("!", _ERROR),
}


class _LeftEllipsisText:
    """Render text left-truncated so the meaningful model/variant suffix survives narrow columns."""

    __slots__ = ("style", "value")

    def __init__(self, value: str, *, style: str = "") -> None:
        """Retain the raw value and style for width-aware truncation at render time."""
        self.value = value
        self.style = style

    def __rich_console__(self, _console: Console, options: ConsoleOptions) -> RenderResult:
        """Yield text truncated on the left to the column's actual render width."""
        yield Text(_left_ellipsis(self.value, options.max_width), style=self.style)


class MatrixTerminalReporter:
    """Render runtime state only in human TTY mode; observers never affect evaluation."""

    def __init__(self, config: EvalConfig, *, run_id: str, run_dir: str, human: bool, escape_available: bool) -> None:
        """Initialize the isolated runtime projection for one command invocation."""
        self._config = config
        self._run_id = run_id
        self._run_dir = run_dir
        self._human = human
        self._cancel_hint = "Press Escape to cancel" if escape_available else "Press Ctrl+C to cancel"
        self._console = Console(stderr=True)
        self._state = PresentationState()
        self._live: Live | None = None

    def __enter__(self) -> Self:
        """Print orientation then start a transient Live frame for interactive runs."""
        if self._human:
            # State mutation point: clear stale terminal output before Live owns the display.
            self._console.clear(home=True)
            self._console.print(self._orientation())
            # Separate the orientation header from the live frame with one blank line.
            self._console.print()
            self._live = Live(
                self._render(),
                console=self._console,
                # Faster refresh keeps the spinner and progress bar visibly animated.
                refresh_per_second=12,
                redirect_stdout=False,
                redirect_stderr=True,
                transient=True,
            )
            self._live.start()
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Always remove the transient frame before caller diagnostics are printed."""
        self._stop_live()

    def handle(self, event: MatrixProgressEvent) -> None:
        """Consume a safe lifecycle event and update the mutable runtime projection."""
        self._state.ingest(event, timeout=self._config.model_timeout, max_tool_calls=self._config.max_tool_calls)
        if self._live is not None:
            self._live.update(self._render(), refresh=True)
        elif not self._human:
            self._print_machine_event(event)

    def finish(self, *, run_dir: str, report_html: str) -> None:
        """Print exactly one durable human final after artifacts are complete."""
        if not self._human:
            return
        self._stop_live()
        self._console.print(render_durable_final(self._state, run_dir=run_dir, report_html=report_html))

    @property
    def state(self) -> PresentationState:
        """Expose runtime state to the CLI only for final lifecycle handling."""
        return self._state

    def _stop_live(self) -> None:
        """Stop Live before any permanent stderr output."""
        if self._live is None:
            return
        try:
            self._live.stop()
        finally:
            self._live = None

    def _orientation(self) -> Panel:
        """Return a full-width, form-aligned pre-run summary without model output or payloads."""
        temperature = self._config.temperature if self._config.temperature is not None else "default"
        models = Text()
        for index, model_id in enumerate(self._config.models):
            if index:
                models.append("\n")
            models.append(variant_label(model_id, self._config.reasoning_effort), style="bold")
            models.append(f"   temperature={temperature}", style="dim")
        # A two-column grid keeps labels right-aligned against left-aligned values like a form.
        form = Table.grid(padding=(0, 2))
        form.add_column(justify="right", style="dim", no_wrap=True)
        form.add_column(justify="left", overflow="fold")
        form.add_row("Run", self._run_id)
        form.add_row("Models", models)
        form.add_row("Candidates", ", ".join(self._config.candidates))
        form.add_row("Concurrency", str(self._config.concurrency))
        form.add_row("Timeout", f"{self._config.model_timeout:g}s")
        form.add_row("Tool cap", str(self._config.max_tool_calls))
        form.add_row("Artifacts", self._run_dir)
        # The cancellation hint lives only in the live frame footer, so it is not repeated here.
        return Panel(form, title="LLM Sandbox evaluation", border_style=_ACTIVE, box=box.ROUNDED, expand=True)

    def _render(self) -> Group:
        """Compose the transient frame from PresentationState only."""
        return Group(
            self._status_line(),
            self._overall_progress(),
            Text(),
            Rule("Running", style=_ACTIVE, characters="─"),
            self._lanes_table(),
            Text(),
            Rule("Recent", style=_ACTIVE, characters="─"),
            self._recent_table(),
            Text(),
            self._issues_panel(),
            Text(self._cancel_hint, style="dim"),
        )

    def _status_line(self) -> Text:
        """Return a compact colored counts line above the overall progress bar."""
        counts = self._state.counts
        finished = len(self._state.completed)
        running = len(self._state.lanes)
        queued = max(0, self._state.total - finished - running)
        line = Text()
        # State glyphs are the only colored counts so scanning stays fast.
        line.append(f"{_OUTCOME_STYLES['correct'][0]} {counts.correct}", style=_SUCCESS)
        line.append("  ")
        line.append(f"{_OUTCOME_STYLES['incorrect'][0]} {counts.incorrect}", style=_WARNING)
        line.append("  ")
        line.append(f"{_OUTCOME_STYLES['incomplete'][0]} {counts.incomplete}", style=_ERROR)
        line.append("    ")
        line.append(f"Running {running}", style=_ACTIVE)
        line.append(f"  Queued {queued}", style="dim")
        line.append(
            f"    Scored {counts.scored}  Quality {counts.quality_rate:.1%}  Coverage {counts.coverage_rate:.1%}",
            style="bold",
        )
        line.append(f"    Elapsed {_duration(perf_counter() - self._state.started_at)}", style=_WARNING)
        return line

    def _overall_progress(self) -> Progress:
        """Return a fresh single-task progress bar reflecting completed vs planned cells."""
        finished = len(self._state.completed)
        progress = Progress(
            TextColumn("[bold]Overall[/]"),
            BarColumn(bar_width=None, style=_TRACK, complete_style=_ACTIVE, finished_style=_SUCCESS),
            TaskProgressColumn(style=_PERCENT),
            TextColumn("{task.completed}/{task.total}", style=_SUCCESS),
            expand=True,
        )
        # A fresh task each render keeps the bar a pure function of current state.
        progress.add_task("overall", total=max(1, self._state.total), completed=finished)
        return progress

    def _lanes_table(self) -> Table:
        """Render active lanes with an animated spinner, variant tail, budget, and tool cap."""
        table = Table(box=None, expand=True, pad_edge=False, header_style=_ACTIVE)
        table.add_column("", width=2, no_wrap=True)
        table.add_column("Request", ratio=3, overflow="ellipsis", no_wrap=True)
        table.add_column("Variant", width=_VARIANT_WIDTH, style="dim", no_wrap=True)
        table.add_column("Elapsed / timeout", width=18, justify="right", no_wrap=True)
        table.add_column("Tools / cap", width=12, justify="right", no_wrap=True)
        for lane in self._state.lanes.values():
            table.add_row(
                # A live Spinner renderable animates every refresh without any simulated phase text.
                Spinner("dots", style=_ACTIVE),
                Text(lane.request),
                _LeftEllipsisText(variant_label(lane.cell.model_id, lane.cell.reasoning_effort), style="dim"),
                Text(f"{_duration(perf_counter() - lane.started_at)} / {lane.timeout:g}s"),
                Text(f"{lane.tools_used} / {lane.max_tool_calls}"),
            )
        if not self._state.lanes:
            # Idle placeholder keeps the section height stable between refreshes.
            table.add_row(
                Text(" "),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
            )
        return table

    def _recent_table(self) -> Table:
        """Render terminal results with a colored glyph, variant tail, and one semantic result."""
        table = Table(box=None, expand=True, pad_edge=False, header_style=_ACTIVE)
        table.add_column("", width=1, no_wrap=True)
        table.add_column("#", width=4, justify="right")
        table.add_column("Request", ratio=3, overflow="ellipsis", no_wrap=True)
        table.add_column("Variant", width=_VARIANT_WIDTH, style="dim", no_wrap=True)
        table.add_column("Result", width=26, no_wrap=True, overflow="ellipsis")
        table.add_column("Tools", width=6, justify="right")
        table.add_column("Elapsed", width=8, justify="right")
        for index, cell in enumerate(reversed(self._state.completed[-_RECENT_RESULTS:]), start=1):
            trace = cell.trace
            glyph, color = _OUTCOME_STYLES[trace.outcome.state]
            table.add_row(
                Text(glyph, style=color),
                str(len(self._state.completed) - index + 1),
                Text(trace.user_request or cell.case_id),
                _LeftEllipsisText(cell.variant, style="dim"),
                # Single Result column, colored by outcome, still reads as state·cause.
                Text(result_label(trace), style=color),
                str(trace.diagnostics.tool_calls),
                _duration(trace.diagnostics.elapsed_seconds or 0.0),
            )
        return table

    def _issues_panel(self) -> Panel:
        """Group actual operational causes without presenting them as scored mismatches."""
        issues = self._state.operational_issues
        if not issues:
            return Panel(Text("None", style="dim"), title="Operational issues", border_style="dim", expand=False)
        body = Text(", ".join(f"{cause}: {count}" for cause, count in sorted(issues.items())), style=_ERROR)
        return Panel(body, title="Operational issues", border_style=_ERROR, expand=False)

    def _print_machine_event(self, event: MatrixProgressEvent) -> None:
        """Keep redirected stderr lifecycle diagnostics concise and deterministic."""
        if event.state == "matrix_started":
            self._console.print(f"matrix_started total={event.total or 0}", markup=False)
        elif event.state == "cell_finished" and event.trace is not None:
            self._console.print(
                f"cell_finished index={event.completion_index or 0} result={result_label(event.trace)}",
                markup=False,
            )


def render_durable_final(state: PresentationState, *, run_dir: str, report_html: str) -> Panel:
    """Return the sole durable interactive final after Live has stopped."""
    counts = state.counts
    summary = Text()
    summary.append(f"Finished {len(state.completed)}/{state.total}\n", style="bold")
    summary.append("Quality ", style="bold")
    summary.append(f"{counts.correct}/{counts.scored} ({counts.quality_rate:.1%})\n", style=_SUCCESS)
    summary.append("Coverage ", style="bold")
    summary.append(f"{counts.scored}/{counts.total} ({counts.coverage_rate:.1%})\n", style=_ACTIVE)
    issues = state.operational_issues
    if issues:
        # Operational failures are surfaced separately from scored quality.
        summary.append(
            "Operational issues: " + ", ".join(f"{key}: {value}" for key, value in sorted(issues.items())) + "\n",
            style=_ERROR,
        )
    table = Table(box=box.SIMPLE_HEAD, expand=False, pad_edge=False, header_style=_ACTIVE)
    table.add_column("Candidate")
    table.add_column("Variant", overflow="fold")
    table.add_column("Quality", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Calls/failures", justify="right")
    table.add_column("Avg elapsed", justify="right")
    table.add_column("Tokens / cost", justify="right")
    for aggregate in state.aggregates:
        table.add_row(
            aggregate.candidate_id,
            aggregate.variant,
            Text(f"{aggregate.counts.quality_rate:.1%}", style=_SUCCESS),
            Text(f"{aggregate.counts.coverage_rate:.1%}", style=_ACTIVE),
            f"{aggregate.mean_calls:.1f}/{aggregate.mean_failed_calls:.1f}",
            _duration(aggregate.mean_elapsed),
            f"{_token_total(aggregate.total_tokens)} / cost "
            f"{aggregate.total_cost if aggregate.total_cost is not None else 'unavailable'}",
        )
    notable = [cell for cell in state.completed if cell.trace.outcome.state != "correct"]
    if notable:
        summary.append("Notable cells:\n")
        for cell in notable[:5]:
            glyph, color = _OUTCOME_STYLES[cell.trace.outcome.state]
            summary.append(f"  {glyph} ", style=color)
            summary.append(f"{cell.case_id} {result_label(cell.trace)}\n")
    summary.append(f"Artifacts: {run_dir}\n", style="dim")
    summary.append(f"report.html: {report_html}", style="dim")
    return Panel(
        Group(summary, Text(), table), title="Eval complete", border_style=_SUCCESS, box=box.ROUNDED, expand=False
    )


def _left_ellipsis(value: str, width: int) -> str:
    """Keep the rightmost identifier text when a bounded metadata field overflows."""
    if width <= 1 or len(value) <= width:
        return value
    return f"\u2026{value[-(width - 1) :]}"


def _token_total(tokens: float | None) -> str:
    """Format aggregate tokens as a nearest-thousand human terminal value."""
    if tokens is None:
        return "tokens unavailable"
    return f"tokens {int(tokens / 1_000 + 0.5):,}k"


def _duration(seconds: float) -> str:
    """Format monotonic durations compactly."""
    minutes, remaining = divmod(max(0, int(seconds)), 60)
    return f"{minutes}:{remaining:02d}" if minutes else f"{seconds:.1f}s"
