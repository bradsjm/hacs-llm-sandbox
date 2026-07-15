"""Safe stderr presentation for eval matrix lifecycle events."""

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Self

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import LanePhaseEvent, MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.presentation import (
    JudgeAggregate,
    JudgeAttention,
    JudgeSummary,
    OperationalIssueGroup,
    PresentationCell,
    PresentationState,
    ReportPresentationModel,
    result_label,
    variant_label,
)
from llm_sandbox_evals.statistics import (
    canonical_cells,
    category_aggregates,
    pair_aggregates,
    paraphrase_cells,
    result_counts,
    task_robustness,
    wilson_interval,
)

_ACTIVE = "#38b6ca"
_SUCCESS = "#55c97c"
_WARNING = "#d9a514"
_ERROR = "#f2705f"
_TOOL = "#e89b4f"
_TRACK = "grey37"
_PERCENT = "#b8a1e8"
_RECENT_RESULTS = 10
_OPERATIONAL_DETAIL_LIMIT = 320
_OPERATIONAL_CELL_PREVIEW = 3
# Bounded advisory-judge preview: at most five needs-attention rows so the durable final stays
# scannable and never grows with the judged-cell count.
_JUDGE_ATTENTION_LIMIT = 5
# Live timers display whole seconds, so 4 Hz clears the 1 Hz second-transition resolution with
# margin. Rich Live (non-screen mode) erases and re-prints the entire frame on every refresh —
# there is no cell-level diffing — so a lower rate directly reduces full-frame repaint flicker.
# The dots spinner's speed is scaled to match this rate (see _SPINNER_SPEED) so it advances one
# frame per refresh instead of jumping. Event-driven updates bypass this cadence via
# Live.update(refresh=True) but sample the same time-based spinner frame, so they stay smooth.
_LIVE_REFRESH_PER_SECOND = 4
# Rich "dots" spinner: 10 frames at 80 ms interval (12.5 fps native). See rich._spinners.SPINNERS.
_DOTS_INTERVAL_MS = 80
# Scale the spinner's effective frame rate to the Live refresh rate so each refresh advances
# exactly one frame. At 4 Hz this yields a smooth 4 fps animation (2.5 s per 10-frame cycle)
# instead of the jerky ~3-frame jumps that sampling a 12.5 fps animation at 4 Hz would produce.
_SPINNER_SPEED = _LIVE_REFRESH_PER_SECOND * _DOTS_INTERVAL_MS / 1000

# Fixed lane-table column widths (in cells). Named so the responsive breakpoint is derived from
# the real budget rather than a magic number.
_SPINNER_WIDTH = 2
_VARIANT_WIDTH = 22
_ELAPSED_WIDTH = 18
_TOOLS_WIDTH = 12
# Smallest width that renders the longest runtime label `result · execute_home_code` (26 cells)
# without ellipsis, so the executing tool name is always shown in full during running/processing.
_ACTIVITY_WIDTH = 26

# Rich Table uses default cell padding (0, 1): one space each side, i.e. two cells per column.
# pad_edge=False suppresses the outermost left and right pad columns, trimming two cells overall.
_CELL_PADDING = 2
_EDGE_TRIM = 2
# Keep Request at least this legible before abandoning the wide six-column layout.
_MIN_REQUEST_WIDTH = 12
# Six-column layout budget: every fixed column + Activity + inter-cell padding + a legible Request.
# Below this the ratio Request cannot absorb the deficit, so the lower-priority Variant is omitted.
_ACTIVITY_WIDE_MIN_COLUMNS = (
    _SPINNER_WIDTH
    + _ACTIVITY_WIDTH
    + _VARIANT_WIDTH
    + _ELAPSED_WIDTH
    + _TOOLS_WIDTH
    + (6 * _CELL_PADDING - _EDGE_TRIM)
    + _MIN_REQUEST_WIDTH
)  # = 80 fixed + 10 padding + 12 request = 102

# Outcome state → (glyph, color). Drives every colored result surface consistently.
_OUTCOME_STYLES: dict[str, tuple[str, str]] = {
    "correct": ("\u2713", _SUCCESS),
    "incorrect": ("\u2717", _WARNING),
    "incomplete": ("!", _ERROR),
}

# Phase → concise, content-free activity verb. Truthful labels sourced only from stream facts.
_ACTIVITY_LABELS: dict[str, str] = {
    "queued": "queued",
    "awaiting_model": "awaiting model",
    "thinking": "thinking",
    "preparing_tool_call": "preparing",
    "running_tool": "run",
    "processing_tool_result": "result",
    "responding": "responding",
    "scoring": "scoring",
    "judging": "judging",
    "finished": "finished",
}
# Phase → semantic terminal style. Unknown/missing phases fall back to the neutral track color.
_ACTIVITY_STYLES: dict[str, str] = {
    "queued": _WARNING,
    "awaiting_model": _WARNING,
    "thinking": _ACTIVE,
    "preparing_tool_call": _TOOL,
    "running_tool": _TOOL,
    "processing_tool_result": _TOOL,
    "responding": _SUCCESS,
    "scoring": _PERCENT,
    "judging": _PERCENT,
    "finished": _SUCCESS,
}


class _DynamicLiveFrame:
    """Rebuild the Live frame on every Rich refresh so timers keep moving."""

    def __init__(self, render: Callable[[], RenderableType]) -> None:
        self._render = render

    def __rich_console__(self, _console: Console, _options: ConsoleOptions) -> RenderResult:
        yield self._render()


# Only the authoritative running/processing phases append the safe tool name; preparing_tool_call
# carries a provider-supplied name and stays a bare label.
_TOOL_PHASES: frozenset[str] = frozenset({"running_tool", "processing_tool_result"})


def _activity_label(phase: str | None, tool_name: str | None) -> str:
    """Return a concise activity label; em dash for no phase, tool name only for authoritative phases."""
    # Branch boundary: an unobserved phase renders as an em dash, never as a simulated wait.
    if phase is None:
        return "\u2014"
    base = _ACTIVITY_LABELS.get(phase, phase)
    # Branch boundary: only running/processing append the executing tool name; preparing never does.
    if phase in _TOOL_PHASES and tool_name:
        return f"{base} \u00b7 {tool_name}"
    return base


def _activity_style(phase: str | None) -> str:
    """Return the semantic Activity color, falling back safely for absent or future phases."""
    # Branch boundary: absent or unrecognized phases stay neutral instead of raising or implying progress.
    return _ACTIVITY_STYLES.get(phase, _TRACK) if phase is not None else _TRACK


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
        self._spinners: dict[MatrixCellRef, Spinner] = {}

    def __enter__(self) -> Self:
        """Print orientation then start a transient Live frame for interactive runs."""
        if self._human:
            # State mutation point: clear stale terminal output before Live owns the display.
            self._console.clear(home=True)
            self._console.print(self._orientation())
            # Separate the orientation header from the live frame with one blank line.
            self._console.print()
            self._live = Live(
                self._live_frame(),
                console=self._console,
                refresh_per_second=_LIVE_REFRESH_PER_SECOND,
                redirect_stdout=False,
                redirect_stderr=True,
                transient=True,
                # Branch boundary: operational provider payloads may exceed the terminal height;
                # prefer complete human error detail over Rich's default vertical ellipsis.
                vertical_overflow="visible",
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
            self._live.update(self._live_frame(), refresh=True)
        elif not self._human:
            self._print_machine_event(event)

    def handle_phase(self, event: LanePhaseEvent) -> None:
        """Project a payload-free phase; refresh Live only on a real change, never emitting machine output."""
        changed = self._state.ingest_phase(event)
        # Branch boundary: coalesce repeated stream-delta phases — rebuild the Live frame only when the
        # projected state actually changed, so zero-buffer stream deltas do not backpressure the model.
        if changed and self._live is not None:
            self._live.update(self._live_frame(), refresh=True)

    def finish(self, *, run_dir: str, report_html: str, report_model: ReportPresentationModel) -> None:
        """Print exactly one durable human final after artifacts are complete."""
        if not self._human:
            return
        self._stop_live()
        # The immutable report model is the sole source for the advisory judge section; the runtime
        # PresentationState still drives every deterministic, live, and operational surface.
        self._console.print(
            render_durable_final(self._state, run_dir=run_dir, report_html=report_html, report_model=report_model)
        )

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
            _operational_issues_panel(self._state.operational_issue_groups),
            Text(self._cancel_hint, style="dim"),
        )

    def _live_frame(self) -> _DynamicLiveFrame:
        """Return a renderable that recomputes time-sensitive values on every Live refresh."""
        return _DynamicLiveFrame(self._render)

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
            f"    Scored {counts.scored}  Quality {_percentage(counts.quality_rate)}  "
            f"Coverage {_percentage(counts.coverage_rate)}",
            style="bold",
        )
        line.append(f"    Elapsed {_live_duration(perf_counter() - self._state.started_at)}", style=_WARNING)
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
        """Render active lanes with baseline Activity and width-responsive Variant visibility."""
        # Branch boundary: below the six-column budget, drop only Variant so spinner + Request +
        # Activity keep their full widths and the tool name never ellipsizes.
        narrow = self._console.width < _ACTIVITY_WIDE_MIN_COLUMNS
        show_variant = not narrow
        table = Table(box=None, expand=True, pad_edge=False, header_style=_ACTIVE)
        table.add_column("", width=_SPINNER_WIDTH, no_wrap=True)
        table.add_column("Request", ratio=3, overflow="ellipsis", no_wrap=True)
        # Activity is a baseline lane fact and remains visible before any phase transition.
        table.add_column("Activity", width=_ACTIVITY_WIDTH, overflow="ellipsis", no_wrap=True)
        # Branch boundary: Variant is the only column suppressed on the narrow layout.
        if show_variant:
            table.add_column("Variant", width=_VARIANT_WIDTH, style="dim", no_wrap=True)
        table.add_column("Elapsed / timeout", width=_ELAPSED_WIDTH, justify="right", no_wrap=True)
        table.add_column("Tools / cap", width=_TOOLS_WIDTH, justify="right", no_wrap=True)
        # Prune spinners for lanes that have finished so the cache stays bounded to active lanes.
        # All _spinners access happens in this render method (under Live's render lock in
        # production), so the prune cannot race a main-thread mutation.
        for stale in [cell for cell in self._spinners if cell not in self._state.lanes]:
            del self._spinners[stale]
        for lane in self._state.lanes.values():
            # Phase color is shared by the persistent spinner and the payload-free Activity label.
            activity_style = _activity_style(lane.phase)
            spinner = self._spinners.get(lane.cell)
            if spinner is None:
                spinner = Spinner("dots", style=activity_style, speed=_SPINNER_SPEED)
                self._spinners[lane.cell] = spinner
            else:
                # Preserve the spinner clock while still reflecting phase color changes.
                spinner.style = activity_style
            row: list[RenderableType] = [
                # A persistent Spinner renderable animates every refresh without any simulated phase text.
                spinner,
                Text(lane.request),
            ]
            # Activity text is derived only from stored phase facts, never model content.
            row.append(Text(_activity_label(lane.phase, lane.tool_name), style=activity_style))
            # Branch boundary: Variant renders only when the wide layout is in effect.
            if show_variant:
                row.append(
                    _LeftEllipsisText(variant_label(lane.cell.model_id, lane.cell.reasoning_effort), style="dim")
                )
            row.extend(
                (
                    Text(f"{_live_duration(perf_counter() - lane.started_at)} / {lane.timeout:g}s"),
                    Text(f"{lane.tools_used} / {lane.max_tool_calls}"),
                )
            )
            table.add_row(*row)
        # Reserve the effective matrix concurrency in rows so the Running section does not shrink
        # as lanes finish. Before matrix_started the total is unknown, so reserve one safe row.
        capacity = 1 if self._state.total == 0 else min(self._config.concurrency, self._state.total)
        for _ in range(max(0, capacity - len(self._state.lanes))):
            placeholder: list[RenderableType] = [Text(" "), Text("—", style="dim")]
            placeholder.append(Text("—", style="dim"))
            if show_variant:
                placeholder.append(Text("—", style="dim"))
            placeholder.extend((Text("—", style="dim"), Text("—", style="dim")))
            table.add_row(*placeholder)
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
        recent = self._state.completed[-_RECENT_RESULTS:]
        first_recent_index = len(self._state.completed) - len(recent) + 1
        for index, cell in enumerate(recent, start=first_recent_index):
            trace = cell.trace
            glyph, color = _OUTCOME_STYLES[trace.outcome.state]
            table.add_row(
                Text(glyph, style=color),
                str(index),
                Text(trace.request_text or cell.case_id),
                _LeftEllipsisText(cell.variant, style="dim"),
                # Single Result column, colored by outcome, still reads as state·cause.
                Text(result_label(trace), style=color),
                str(trace.diagnostics.tool_calls),
                _duration(trace.diagnostics.elapsed_seconds or 0.0),
            )
        return table

    def _print_machine_event(self, event: MatrixProgressEvent) -> None:
        """Keep redirected stderr lifecycle diagnostics concise and deterministic."""
        if event.state == "matrix_started":
            self._console.print(f"matrix_started total={event.total or 0}", markup=False)
        elif event.state == "cell_finished" and event.trace is not None:
            self._console.print(
                f"cell_finished index={event.completion_index or 0} result={result_label(event.trace)}",
                markup=False,
            )


def _operational_issues_panel(groups: tuple[OperationalIssueGroup, ...]) -> Panel:
    """Render grouped operational failures with durable provider details preserved."""
    # Branch boundary: an empty run still reserves the full-width issues section for scanability.
    if not groups:
        return Panel(Text("None", style="dim"), title="Operational issues", border_style="dim", expand=True)

    table = Table(expand=True, box=box.SIMPLE_HEAD, show_lines=True, header_style=_ACTIVE)
    table.add_column("#", width=3, justify="right", no_wrap=True)
    table.add_column("Cause", width=10, overflow="fold")
    table.add_column("Variant", width=20, overflow="fold")
    table.add_column("Cells", width=18, overflow="fold")
    table.add_column("Exception", width=16, overflow="fold")
    table.add_column("HTTP / provider code", width=22, overflow="fold")
    table.add_column("Detail", min_width=28, overflow="fold", ratio=1)
    for group in groups:
        table.add_row(
            Text(str(group.count)),
            Text(group.cause, style=_ERROR),
            _operational_issue_variant(group),
            _operational_issue_cells(group.cells),
            Text(group.exception_type),
            _http_provider_code(group),
            Text(_operational_issue_detail(group)),
        )
    return Panel(table, title="Operational issues", border_style=_ERROR, expand=True)


def _operational_issue_variant(group: OperationalIssueGroup) -> Text:
    """Return the display variant plus the concrete provider model when supplied."""
    value = Text(group.variant)
    # Branch boundary: provider_model is optional structured metadata, so omit the line entirely when absent.
    if group.provider_model:
        value.append("\nprovider model: ", style="dim")
        value.append(group.provider_model, style="dim")
    return value


def _http_provider_code(group: OperationalIssueGroup) -> Text:
    """Combine available HTTP status and provider code without synthesizing missing values."""
    parts: list[str] = []
    if group.status_code is not None:
        parts.append(str(group.status_code))
    if group.provider_code:
        parts.append(group.provider_code)
    return Text("\n".join(parts))


def _operational_issue_detail(group: OperationalIssueGroup) -> str:
    """Return a bounded terminal summary while retaining the full failure in errors.log."""
    raw_traceback = "Traceback (most recent call last):" in group.detail
    # Branch boundary: structured messages replace raw tracebacks, but provider payloads remain actionable.
    detail = group.message if raw_traceback and group.message else group.detail
    suffix = " [full: errors.log]" if raw_traceback else ""
    compact = " ".join(detail.split())
    if len(compact) + len(suffix) > _OPERATIONAL_DETAIL_LIMIT:
        compact = compact[: _OPERATIONAL_DETAIL_LIMIT - len(suffix) - 1].rstrip() + "…"
        suffix = " [full: errors.log]"
    return f"{compact}{suffix}"


def _operational_issue_cells(cells: tuple[str, ...]) -> Text:
    """Return a compact cell preview while the per-trace errors.log retains every occurrence."""
    preview = cells[:_OPERATIONAL_CELL_PREVIEW]
    value = Text("\n".join(preview))
    remaining = len(cells) - len(preview)
    if remaining:
        value.append(f"\n+ {remaining} more", style="dim")
    return value


def render_durable_final(
    state: PresentationState,
    *,
    run_dir: str,
    report_html: str,
    report_model: ReportPresentationModel | None = None,
) -> Panel:
    """Return the sole durable interactive final after Live has stopped.

    ``report_model`` supplies the advisory code-judge section only; ``None`` (and any run whose
    cells never requested judging) leaves the deterministic final structurally unchanged.
    """
    canonical = canonical_cells(state.completed)
    counts = result_counts(cell.trace for cell in canonical)
    paraphrases = result_counts(cell.trace for cell in paraphrase_cells(state.completed))
    robustness = task_robustness(state.completed)
    summary = Text()
    summary.append(f"Finished {len(state.completed)}/{state.total}\n", style="bold")
    summary.append("Canonical quality ", style="bold")
    summary.append(
        f"{counts.correct}/{counts.scored} ({_percentage(counts.quality_rate)}) "
        f"Wilson 95% CI {_interval(wilson_interval(counts.correct, counts.scored))}\n",
        style=_SUCCESS,
    )
    summary.append("Canonical coverage ", style="bold")
    summary.append(f"{counts.scored}/{counts.total} ({_percentage(counts.coverage_rate)})\n", style=_ACTIVE)
    summary.append("Paraphrase quality ", style="bold")
    summary.append(
        f"{paraphrases.correct}/{paraphrases.scored} ({_percentage(paraphrases.quality_rate)}) "
        f"utterance-level Wilson 95% CI {_interval(wilson_interval(paraphrases.correct, paraphrases.scored))}\n",
        style=_PERCENT,
    )
    robust_tasks = sum(value.all_passed for value in robustness)
    summary.append("Task robustness ", style="bold")
    summary.append(
        f"{robust_tasks}/{len(robustness)} candidate/model task groups pass all request variants\n",
        style=_ACTIVE,
    )
    for value in robustness:
        # Branch boundary: only tasks with a request-variant regression need durable detail.
        if not value.all_passed:
            summary.append(
                f"  {value.candidate_id}/{value.case_id} · {value.variant} "
                f"{value.correct_variants}/{value.total_variants}\n",
                style=_WARNING,
            )
    table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style=_ACTIVE)
    table.add_column("Candidate")
    table.add_column("Variant", overflow="fold")
    table.add_column("Quality", justify="right")
    table.add_column("Wilson 95% CI", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Calls/failures", justify="right")
    table.add_column("Avg elapsed", justify="right")
    table.add_column("Tokens / cost", justify="right")
    for aggregate in pair_aggregates(canonical):
        table.add_row(
            aggregate.candidate_id,
            aggregate.variant,
            Text(_percentage(aggregate.counts.quality_rate), style=_SUCCESS),
            Text(_interval(wilson_interval(aggregate.counts.correct, aggregate.counts.scored)), style=_SUCCESS),
            Text(_percentage(aggregate.counts.coverage_rate), style=_ACTIVE),
            f"{aggregate.mean_calls:.1f}/{aggregate.mean_failed_calls:.1f}",
            _duration(aggregate.mean_elapsed),
            f"{_token_total(aggregate.total_tokens)} / cost "
            f"{aggregate.total_cost if aggregate.total_cost is not None else 'unavailable'}",
        )
    summary.append(f"Artifacts: {run_dir}\n", style="dim")
    summary.append(f"report.html: {report_html}", style="dim")
    category_table = _category_breakdown_table(state)
    notable_table = _notable_table(state.completed)
    # Branch boundary: fully correct runs do not reserve an empty notable-cells section.
    notable_section: tuple[RenderableType, ...] = (Text(), notable_table) if notable_table is not None else ()
    # Branch boundary: the advisory judge section is appended last, after every deterministic and
    # operational section, and only when some report cell requested judging — otherwise the final is
    # byte-for-byte the pre-judge layout.
    judge_section: tuple[RenderableType, ...] = (
        (Text(), _judge_section(report_model)) if report_model is not None and report_model.judge_requested else ()
    )
    return Panel(
        Group(
            summary,
            Text(),
            table,
            *notable_section,
            Text(),
            category_table,
            Text(),
            _operational_issues_panel(state.operational_issue_groups),
            *judge_section,
        ),
        title="Eval complete",
        border_style=_SUCCESS,
        box=box.ROUNDED,
        expand=True,
    )


def _category_breakdown_table(state: PresentationState) -> Table:
    """Render the additive candidate by variant by category breakdown."""
    table = Table(
        title="By category (all request variants)",
        box=box.SIMPLE_HEAD,
        expand=True,
        pad_edge=False,
        header_style=_ACTIVE,
    )
    table.add_column("Candidate")
    table.add_column("Variant", overflow="fold")
    table.add_column("Category")
    table.add_column("Quality", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Scored", justify="right")
    for aggregate in category_aggregates(state.completed):
        table.add_row(
            aggregate.candidate_id,
            aggregate.variant,
            aggregate.category,
            Text(_percentage(aggregate.counts.quality_rate), style=_SUCCESS),
            Text(_percentage(aggregate.counts.coverage_rate), style=_ACTIVE),
            str(aggregate.counts.scored),
        )
    return table


def _notable_table(cells: Sequence[PresentationCell], *, limit: int = 8) -> Table | None:
    """Render non-correct cells with request-variant identity and honest truncation."""
    notable = sorted(
        (cell for cell in cells if cell.trace.outcome.state != "correct"),
        key=lambda cell: (
            cell.request_variant_id != "canonical",
            cell.candidate_id,
            cell.model_id,
            cell.case_id,
            cell.request_variant_id,
        ),
    )
    # Branch boundary: successful runs retain the compact final without an empty failure table.
    if not notable:
        return None
    table = Table(title="Notable cells", box=None, expand=True, pad_edge=False, header_style=_ACTIVE)
    table.add_column("Cell / model variant / result", overflow="fold")
    for cell in notable[:limit]:
        glyph, color = _OUTCOME_STYLES[cell.trace.outcome.state]
        row = Text(glyph, style=color)
        row.append(f" {cell.candidate_id}/{cell.case_id}/{cell.request_variant_id}")
        row.append(f"\n  {cell.variant}", style="dim")
        row.append("\n  ")
        row.append(result_label(cell.trace), style=color)
        table.add_row(row)
    remaining = len(notable) - limit
    # Branch boundary: bounded terminal output still reports every omitted result numerically.
    if remaining > 0:
        table.add_row(Text(f"+ {remaining} more", style="dim"))
    return table


def _judge_section(report_model: ReportPresentationModel) -> Panel:
    """Return the advisory code-judge panel, visually distinct from deterministic quality/ranking."""
    summary = report_model.judge_summary
    attention_table = _judge_attention_table(report_model.judge_needs_attention)
    # Branch boundary: omit the needs-attention block entirely when no result requires review.
    attention_section: tuple[RenderableType, ...] = (Text(), attention_table) if attention_table is not None else ()
    return Panel(
        Group(
            # An explicit advisory disclaimer keeps this section from reading as deterministic scoring.
            Text("Advisory only — does not affect quality, ranking, coverage, or the verdict.", style="dim"),
            _judge_identity_text(report_model.descriptor),
            _judge_summary_text(summary),
            Text(),
            _judge_aggregates_table(report_model.judge_aggregates),
            *attention_section,
        ),
        title="Code judge · advisory",
        border_style=_PERCENT,
        box=box.ROUNDED,
        expand=True,
    )


def _judge_identity_text(descriptor: dict[str, object]) -> Text:
    """Return the judge model plus rubric identity, using the em dash for absent descriptor facts."""
    model = descriptor.get("judge_model")
    rubric_id = descriptor.get("judge_rubric_id")
    version = descriptor.get("judge_rubric_version")
    text = Text()
    text.append("Judge model ", style="bold")
    text.append("\u2014" if model is None else str(model), style=_PERCENT)
    # Branch boundary: the rubric identity is shown only when the descriptor actually carries it.
    if rubric_id is not None:
        rubric = str(rubric_id) if version is None else f"{rubric_id} v{version}"
        text.append("   Rubric ", style="bold")
        text.append(rubric, style="dim")
    return text


def _judge_summary_text(summary: JudgeSummary) -> Text:
    """Return the compact overall judge facts, with an em dash for absent denominators and means."""
    # Branch boundary: with nothing judged, the pass denominator is unavailable rather than zero.
    passed_denominator = str(summary.available) if summary.available else "\u2014"
    text = Text()
    text.append("Judged ", style="bold")
    text.append(f"{summary.available}/{summary.requested}", style=_PERCENT)
    text.append("   Passed ", style="bold")
    text.append(f"{summary.passed}/{passed_denominator} ({_percentage(summary.pass_rate)})", style=_PERCENT)
    text.append("   Mean score ", style="bold")
    text.append(_percentage(summary.mean_score), style=_PERCENT)
    text.append("   Evaluator failures ", style="bold")
    text.append(str(summary.evaluator_failed), style=_ERROR if summary.evaluator_failed else "dim")
    text.append("   Unavailable ", style="bold")
    text.append(str(summary.unavailable), style=_WARNING if summary.unavailable else "dim")
    return text


def _judge_aggregates_table(aggregates: Sequence[JudgeAggregate]) -> Table:
    """Render the per-candidate/variant advisory aggregates with readable, ANSI-free widths."""
    table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style=_PERCENT)
    table.add_column("Candidate")
    table.add_column("Variant", overflow="fold")
    table.add_column("Judged", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Mean score", justify="right")
    table.add_column("Evaluator failures", justify="right")
    table.add_column("Unavailable", justify="right")
    for aggregate in aggregates:
        # Branch boundary: an aggregate with nothing judged shows an em-dash pass denominator, not zero.
        passed_denominator = str(aggregate.available) if aggregate.available else "\u2014"
        table.add_row(
            aggregate.candidate_id,
            _LeftEllipsisText(aggregate.variant, style="dim"),
            f"{aggregate.available}/{aggregate.requested}",
            f"{aggregate.passed}/{passed_denominator}",
            Text(_percentage(aggregate.mean_score), style=_PERCENT),
            str(aggregate.evaluator_failed),
            str(aggregate.unavailable),
        )
    return table


# Advisory needs-attention status → semantic style. Evaluator gaps read as errors; unavailable and
# explicit non-passing scores read as warnings, never as deterministic-quality success colors.
_JUDGE_ATTENTION_STYLES: dict[str, str] = {
    "failed": _ERROR,
    "unavailable": _WARNING,
    "available": _WARNING,
}


def _judge_attention_table(attention: Sequence[JudgeAttention]) -> Table | None:
    """Render a bounded advisory preview of judged results needing review, or None when empty."""
    # Branch boundary: a clean advisory pass reserves no empty needs-attention table.
    if not attention:
        return None
    # One stacked cell per item in a single fold column keeps every field legible at Console width 80,
    # where a six-column layout would collapse the wrapped Detail to one character per line.
    table = Table(title="Needs attention", box=None, expand=True, pad_edge=False, header_style=_PERCENT)
    table.add_column("Case / candidate variant / judge", overflow="fold")
    for item in attention[:_JUDGE_ATTENTION_LIMIT]:
        style = _JUDGE_ATTENTION_STYLES.get(item.status, _WARNING)
        # Branch boundary: only an available result carries a real score; gaps show the em dash.
        score = _percentage(item.score) if item.status == "available" else "\u2014"
        # Line 1: case + request variant; line 2: candidate + display variant; line 3: judge verdict.
        row = Text(f"{item.case_id}/{item.request_variant_id}")
        row.append(f"\n  {item.candidate_id} · ", style="dim")
        row.append(item.variant, style="dim")
        row.append("\n  ")
        row.append(f"{item.status} · {score} · ", style=style)
        row.append(_judge_attention_detail(item), style=style)
        table.add_row(row)
    remaining = len(attention) - _JUDGE_ATTENTION_LIMIT
    # Branch boundary: bounded output still reports every omitted advisory result numerically.
    if remaining > 0:
        table.add_row(Text(f"+ {remaining} more", style="dim"))
    return table


def _judge_attention_detail(item: JudgeAttention) -> str:
    """Return one safe fixed advisory detail; never the judge reason or any provider-produced text."""
    # Branch boundary: an evaluator failure exposes only its safe classification, never native text.
    if item.status == "failed":
        return item.failure_error_type or "unknown"
    # Branch boundary: an unavailable judge has no score or reason, only its status meaning.
    if item.status == "unavailable":
        return "judge output unavailable"
    # A non-passing available result renders a fixed classification; the full judge reason stays in
    # the HTML, Markdown, and report.json surfaces and is never shown in the terminal.
    return "code quality did not pass"


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


def _percentage(value: float | None) -> str:
    """Format one shared rate without coercing unavailable data to zero."""
    return "—" if value is None else f"{value:.1%}"


def _interval(interval: tuple[float | None, float | None]) -> str:
    """Format one optional Wilson interval for terminal output."""
    low, high = interval
    return "—" if low is None or high is None else f"[{low:.1%}, {high:.1%}]"


def _duration(seconds: float) -> str:
    """Format monotonic durations compactly."""
    minutes, remaining = divmod(max(0, int(seconds)), 60)
    return f"{minutes}:{remaining:02d}" if minutes else f"{seconds:.1f}s"


def _live_duration(seconds: float) -> str:
    """Format live elapsed durations in whole seconds for stable 4 Hz Live refresh."""
    whole = max(0, int(seconds))
    minutes, remaining = divmod(whole, 60)
    return f"{minutes}:{remaining:02d}" if minutes else f"{whole}s"
