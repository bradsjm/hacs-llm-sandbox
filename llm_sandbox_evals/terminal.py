"""Safe stderr presentation for eval matrix lifecycle events."""

from time import perf_counter
from typing import Self

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixProgressEvent
from llm_sandbox_evals.presentation import PresentationState, result_label, variant_label

_ACTIVE = "#38b6ca"
_SUCCESS = "#55c97c"
_WARNING = "#d9a514"
_ERROR = "#f2705f"
_RECENT_RESULTS = 10


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
            self._console.print(self._orientation())
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=4,
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
        """Return a pre-run configuration summary without model output or payloads."""
        models = "\n".join(
            f"• {variant_label(model_id, self._config.reasoning_effort)}"
            f"  temperature={self._config.temperature if self._config.temperature is not None else 'default'}"
            for model_id in self._config.models
        )
        body = (
            f"run {self._run_id}\n"
            f"models\n{models}\n"
            f"candidates: {', '.join(self._config.candidates)}\n"
            f"concurrency: {self._config.concurrency}  timeout: {self._config.model_timeout:g}s  "
            f"tool cap: {self._config.max_tool_calls}\n"
            f"artifacts: {self._run_dir}\n\n{self._cancel_hint}."
        )
        return Panel(body, title="LLM Sandbox evaluation", border_style=_ACTIVE, box=box.ROUNDED, expand=False)

    def _render(self) -> Group:
        """Compose the transient frame from PresentationState only."""
        counts = self._state.counts
        finished = len(self._state.completed)
        header = Text(
            f"Finished {finished}/{self._state.total}  Running {len(self._state.lanes)}  "
            f"Queued {max(0, self._state.total - finished - len(self._state.lanes))}  "
            f"Scored {counts.scored}  Quality {counts.quality_rate:.1%}  Coverage {counts.coverage_rate:.1%}  "
            f"Elapsed {_duration(perf_counter() - self._state.started_at)}",
            style="bold",
        )
        return Group(
            header,
            self._lanes_table(),
            self._recent_table(),
            self._issues_panel(),
            Text(self._cancel_hint, style="dim"),
        )

    def _lanes_table(self) -> Table:
        """Render active lanes with only request, variant, time budget, and tool cap."""
        table = Table(title="Running", box=None, expand=True, header_style=_ACTIVE)
        table.add_column("Request", ratio=3, overflow="ellipsis", no_wrap=True)
        table.add_column("Variant", ratio=1, style="dim", overflow="ellipsis", no_wrap=True)
        table.add_column("Elapsed / timeout", width=18, justify="right", no_wrap=True)
        table.add_column("Tools / cap", width=12, justify="right", no_wrap=True)
        for lane in self._state.lanes.values():
            table.add_row(
                lane.request,
                variant_label(lane.cell.model_id, lane.cell.reasoning_effort),
                f"{_duration(perf_counter() - lane.started_at)} / {lane.timeout:g}s",
                f"{lane.tools_used} / {lane.max_tool_calls}",
            )
        if not self._state.lanes:
            table.add_row("—", "—", "—", "—")
        return table

    def _recent_table(self) -> Table:
        """Render terminal results with one semantic result column and no raw response."""
        table = Table(title="Recent", box=None, expand=True, header_style=_ACTIVE)
        table.add_column("#", width=4, justify="right")
        table.add_column("Request", ratio=3, overflow="ellipsis", no_wrap=True)
        table.add_column("Variant", ratio=1, style="dim", overflow="ellipsis", no_wrap=True)
        table.add_column("Result", width=28, overflow="ellipsis", no_wrap=True)
        table.add_column("Tools", width=6, justify="right")
        table.add_column("Elapsed", width=8, justify="right")
        for index, cell in enumerate(reversed(self._state.completed[-_RECENT_RESULTS:]), start=1):
            trace = cell.trace
            table.add_row(
                str(len(self._state.completed) - index + 1),
                trace.user_request or cell.case_id,
                cell.variant,
                result_label(trace),
                str(trace.diagnostics.tool_calls),
                _duration(trace.diagnostics.elapsed_seconds or 0.0),
            )
        return table

    def _issues_panel(self) -> Panel:
        """Group actual operational causes without presenting them as scored mismatches."""
        issues = self._state.operational_issues
        body = "None" if not issues else ", ".join(f"{cause}: {count}" for cause, count in sorted(issues.items()))
        return Panel(body, title="Operational issues", border_style=_ERROR if issues else "dim", expand=False)

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
    summary = Text(
        f"Finished {len(state.completed)}/{state.total}\n"
        f"Quality {counts.correct}/{counts.scored} ({counts.quality_rate:.1%})\n"
        f"Coverage {counts.scored}/{counts.total} ({counts.coverage_rate:.1%})\n"
    )
    issues = state.operational_issues
    if issues:
        summary.append(
            "Operational issues: " + ", ".join(f"{key}: {value}" for key, value in sorted(issues.items())) + "\n"
        )
    table = Table(box=None, expand=False, header_style=_ACTIVE)
    table.add_column("Candidate")
    table.add_column("Variant")
    table.add_column("Quality")
    table.add_column("Coverage")
    table.add_column("Calls/failures")
    table.add_column("Avg elapsed")
    table.add_column("Tokens/cost")
    for aggregate in state.aggregates:
        table.add_row(
            aggregate.candidate_id,
            aggregate.variant,
            f"{aggregate.counts.quality_rate:.1%}",
            f"{aggregate.counts.coverage_rate:.1%}",
            f"{aggregate.mean_calls:.1f}/{aggregate.mean_failed_calls:.1f}",
            _duration(aggregate.mean_elapsed),
            f"{aggregate.total_tokens if aggregate.total_tokens is not None else 'unavailable'}/"
            f"{aggregate.total_cost if aggregate.total_cost is not None else 'unavailable'}",
        )
    notable = [
        f"{cell.case_id} {result_label(cell.trace)}"
        for cell in state.completed
        if cell.trace.outcome.state != "correct"
    ]
    if notable:
        summary.append("Notable cells: " + "; ".join(notable[:5]) + "\n")
    summary.append(f"Artifacts: {run_dir}\nreport.html: {report_html}")
    return Panel(Group(summary, table), title="Eval complete", border_style=_SUCCESS, box=box.ROUNDED, expand=False)


def _duration(seconds: float) -> str:
    """Format monotonic durations compactly."""
    minutes, remaining = divmod(max(0, int(seconds)), 60)
    return f"{minutes}:{remaining:02d}" if minutes else f"{seconds:.1f}s"
