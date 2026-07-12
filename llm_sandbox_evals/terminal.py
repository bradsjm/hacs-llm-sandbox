"""Stderr-only terminal presentation for native eval matrix lifecycle events."""

from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Self

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.schema import CheckResult
from llm_sandbox_evals.scoring import is_incomplete

_ACTIVE = "#38b6ca"
_SUCCESS = "#55c97c"
_WARNING = "#d9a514"
_ERROR = "#f2705f"
_FAIL_GLYPH = "\u2717"
_REFRESH_PER_SECOND = 4
_RECENT_RESULTS = 10
_MAX_DIAGNOSTIC_CHARS = 500
_MODEL_METADATA_WIDTH = 18
_RECENT_MODEL_WIDTH = 11
_CATEGORY_WIDTH = 11
# Rich adds two cells of padding to each Progress column, so phase also funds Category's padding.
_PHASE_WIDTH = 32 - _CATEGORY_WIDTH - 2
_GATE_WIDTH = 10
_NARROW_GATE_WIDTH = 8
_RECENT_RESPONSE_MIN_WIDTH = 100
_RECENT_NARROW_OUTCOME_WIDTH = 15
_RECENT_WIDE_OUTCOME_WIDTH = 3
_RECENT_WIDE_OUTCOME_MAX_WIDTH = 18


@dataclass(slots=True)
class _Lane:
    """Current visible state for one concurrently evaluated matrix cell."""

    cell: MatrixCellRef
    request: str
    started_at: float
    active_tools: Counter[str] = field(default_factory=Counter)
    response: str | None = None


@dataclass(frozen=True, slots=True)
class _RecentResult:
    """Compact completed-cell data retained by the terminal presentation."""

    cell: MatrixCellRef
    request: str
    outcome: str
    reason: str
    tool_calls: int
    completion_index: int
    response: str
    checks: tuple[CheckResult, ...]
    tool_call_par: int | None
    elapsed: float


@dataclass(frozen=True, slots=True)
class _LeftEllipsisText:
    """Render terminal text with its most useful suffix retained at any column width."""

    value: str
    style: str = ""

    def __rich_console__(self, _console: Console, options: ConsoleOptions) -> RenderResult:
        """Yield Rich text left-truncated to the column's actual render width."""
        yield Text(_left_ellipsis(self.value, options.max_width), style=self.style)


class _PhaseColumn(ProgressColumn):
    """Render normal lane phases or a dim, suffix-preserving final response tail."""

    def __init__(self, *, table_column: Column) -> None:
        """Configure the shared phase/response column width."""
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> RenderableType:
        """Return the response tail for response rows, otherwise the current phase."""
        response = task.fields.get("response")
        if isinstance(response, str):
            return _LeftEllipsisText(response, style="dim")
        phase = task.fields["phase"]
        assert isinstance(phase, str)
        return Text.from_markup(phase)


class _LaneSpinnerColumn(SpinnerColumn):
    """Render the active-lane spinner while reserving its cell for response rows."""

    def render(self, task: Task) -> RenderableType:
        """Return a blank cell for final responses and the cyan spinner otherwise."""
        if task.fields.get("show_spinner") is False:
            return Text(" ")
        return super().render(task)


class _ElapsedColumn(ProgressColumn):
    """Render a monotonic task duration every time Rich refreshes a progress row."""

    def __init__(self, prefix: str = "", *, style: str = "dim", table_column: Column | None = None) -> None:
        """Initialize an optional neutral label before the duration."""
        super().__init__(table_column=table_column)
        self._prefix = prefix
        self._style = style

    def render(self, task: Task) -> Text:
        """Return elapsed time from the stable task start rather than a stale field."""
        if task.fields.get("show_elapsed") is False:
            return Text()
        started_at = task.fields["started_at"]
        assert isinstance(started_at, float)
        return Text(f"{self._prefix}{_duration(perf_counter() - started_at)}", style=self._style)


class MatrixTerminalReporter:
    """Render matrix lifecycle events to stderr without affecting evaluation."""

    def __init__(self) -> None:
        """Initialize isolated display state for one eval command invocation."""
        self._console = Console(stderr=True)
        self._live: Live | None = None
        self._started_at = perf_counter()
        self._total = 0
        self._completed = 0
        self._passes = 0
        self._failures = 0
        self._incomplete = 0
        self._tool_calls = 0
        self._lanes: dict[MatrixCellRef, _Lane] = {}
        self._recent: deque[_RecentResult] = deque(maxlen=_RECENT_RESULTS)

    def __enter__(self) -> Self:
        """Start exactly one Live composition when stderr is a terminal."""
        if self._console.is_terminal:
            # State mutation point: clear stale terminal output before Live owns the screen.
            self._console.clear(home=True)
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=_REFRESH_PER_SECOND,
                redirect_stdout=False,
                # Rich routes provider diagnostics above the Live region while stdout
                # remains untouched for the eval command's machine-readable contract.
                redirect_stderr=True,
                transient=True,
            )
            self._live.start()
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Restore the terminal without suppressing a caller exception."""
        self._stop_live()

    def handle(self, event: MatrixProgressEvent) -> None:
        """Consume one observer event, retaining only presentation-safe metadata."""
        diagnostic: str | None = None
        if event.state == "matrix_started":
            self._total = event.total or 0
        elif event.state == "cell_started" and event.cell is not None:
            # State mutation point: a cell becomes an active lane until its completion event.
            self._lanes[event.cell] = _Lane(event.cell, event.request or event.cell.case_id, perf_counter())
        elif (
            event.state in {"tool_started", "tool_finished"} and event.cell is not None and event.tool_name is not None
        ):
            lane = self._lanes.get(event.cell)
            if lane is not None:
                # Branch boundary: parallel tool calls can overlap, so retain per-tool counts.
                if event.state == "tool_started":
                    lane.active_tools[event.tool_name] += 1
                elif lane.active_tools[event.tool_name] > 1:
                    lane.active_tools[event.tool_name] -= 1
                else:
                    lane.active_tools.pop(event.tool_name, None)
        elif event.state == "response_received" and event.cell is not None:
            lane = self._lanes.get(event.cell)
            if lane is not None:
                # State mutation point: retain only the final assistant response for presentation.
                lane.response = _response_display(event.response)
        elif event.state == "cell_finished" and event.cell is not None and event.trace is not None:
            self._finish_cell(event)
            diagnostic = _compact_diagnostic(event.trace.error)

        if self._live is not None:
            # Refresh the replacement state before a durable write can make Live redraw stale lanes.
            self._live.update(self._render(), refresh=True)
            if diagnostic is not None:
                self._print_live_diagnostic(event, diagnostic)
        elif not self._console.is_terminal:
            self._print_line(event, diagnostic=diagnostic)

    def finish(self, *, overall_mean: float, run_dir: str, report_html: str) -> None:
        """Print the successful post-artifact summary after the Live display ends."""
        elapsed = _duration(perf_counter() - self._started_at)
        if self._console.is_terminal:
            self._stop_live()
            summary = Text()
            _append_status(summary, "✓", "passed", self._passes, _SUCCESS)
            summary.append("  ")
            _append_status(summary, _FAIL_GLYPH, "needs attention", self._failures, _WARNING)
            summary.append("  ")
            _append_status(summary, "!", "incomplete", self._incomplete, _ERROR)
            summary.append(
                f"\ncompleted {self._completed}/{self._total}  tool calls {self._tool_calls}  "
                f"elapsed {elapsed}  overall mean {overall_mean:.3f}"
            )
            summary.append(f"\nreport.html {report_html}")
            summary.append(f"\nreport.json {run_dir}/report.json", style="dim")
            self._console.print(
                Panel(summary, title="Eval complete", border_style=_ACTIVE, box=box.ROUNDED, expand=False)
            )
        else:
            self._console.print(
                f"matrix complete completed={self._completed}/{self._total} pass={self._passes} "
                f"failed={self._failures} incomplete={self._incomplete} tool_calls={self._tool_calls} "
                f"elapsed={elapsed} overall_mean={overall_mean:.3f} run_dir={run_dir} report_html={report_html}",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )

    def _stop_live(self) -> None:
        """Stop the active Live composition before a permanent stderr write."""
        if self._live is None:
            return
        # State cleanup point: always restore a Live-managed terminal before propagating errors.
        try:
            self._live.stop()
        finally:
            self._live = None

    def _print_live_diagnostic(self, event: MatrixProgressEvent, diagnostic: str) -> None:
        """Write a completed-cell error above Live without exposing tool payloads."""
        assert event.cell is not None
        line = Text()
        line.append("error ", style=f"bold {_ERROR}")
        line.append(f"{_cell_name(event)} ({_cell_metadata(event.cell)}): ")
        line.append(diagnostic)
        # Rich's Console cooperates with Live to keep this durable diagnostic in scrollback.
        self._console.print(line, soft_wrap=True)

    def _finish_cell(self, event: MatrixProgressEvent) -> None:
        trace = event.trace
        cell = event.cell
        assert trace is not None
        assert cell is not None
        # State mutation point: terminal totals are updated exactly once per completed cell.
        self._completed = event.completion_index or self._completed + 1
        lane = self._lanes.pop(cell, None)
        request = event.request or (lane.request if lane is not None else cell.case_id)
        self._tool_calls += trace.tool_call_count
        if is_incomplete(trace.checks):
            outcome = "Could not complete"
            self._incomplete += 1
        elif trace.score > 0:
            outcome = "Completed"
            self._passes += 1
        else:
            outcome = "Needs attention"
            self._failures += 1
        self._recent.append(
            _RecentResult(
                cell,
                request,
                outcome,
                _outcome_detail(trace.checks),
                trace.tool_call_count,
                event.completion_index or self._completed,
                lane.response if lane is not None and lane.response is not None else "—",
                trace.checks,
                event.tool_call_par,
                event.elapsed or 0.0,
            )
        )

    def _render(self) -> Group:
        return Group(
            Text("LLM Sandbox evaluation", style="bold"),
            self._overall_progress(),
            Text(),
            Rule("Running now", style="dim"),
            self._lanes_progress(),
            Text(),
            Rule("Recent", style="dim"),
            self._recent_table(),
            Text(),
            self._totals(),
            _gate_legend(),
        )

    def _overall_progress(self) -> Progress:
        progress = Progress(
            TextColumn("Overall", table_column=Column(no_wrap=True)),
            BarColumn(
                bar_width=None,
                style="grey37",
                complete_style=_ACTIVE,
                finished_style=_ACTIVE,
            ),
            TaskProgressColumn(style="#b8a1e8"),
            TextColumn(
                "{task.fields[completed_display]}/{task.fields[total_display]}",
                style="#a6d95a",
                table_column=Column(no_wrap=True),
            ),
            _ElapsedColumn("elapsed ", style="#d9a514"),
            expand=True,
        )
        progress.add_task(
            "overall",
            total=max(1, self._total),
            completed=self._completed,
            total_display=str(self._total),
            completed_display=str(self._completed),
            started_at=self._started_at,
        )
        return progress

    def _lanes_progress(self) -> Progress:
        progress = Progress(
            _LaneSpinnerColumn(style=_ACTIVE),
            TextColumn(
                "{task.description}",
                markup=True,
                table_column=Column(ratio=4, no_wrap=True, overflow="ellipsis"),
            ),
            TextColumn(
                "{task.fields[metadata]}",
                markup=True,
                style="dim",
                table_column=Column(width=_MODEL_METADATA_WIDTH, no_wrap=True, overflow="ellipsis"),
            ),
            TextColumn(
                "{task.fields[category]}",
                markup=True,
                style="dim",
                table_column=Column(width=_CATEGORY_WIDTH, no_wrap=True, overflow="ellipsis"),
            ),
            _PhaseColumn(table_column=Column(width=_PHASE_WIDTH, justify="right", no_wrap=True, overflow="ellipsis")),
            _ElapsedColumn(table_column=Column(width=6, no_wrap=True)),
            expand=True,
        )
        for lane in self._lanes.values():
            progress.add_task(
                escape(lane.request),
                total=None,
                metadata=escape(_left_ellipsis(lane.cell.model_id, _MODEL_METADATA_WIDTH)),
                category=escape(_left_ellipsis(lane.cell.category, _CATEGORY_WIDTH)),
                phase=escape(_lane_phase(lane.active_tools)),
                started_at=lane.started_at,
            )
            if lane.response is not None:
                progress.add_task(
                    "[dim]Response[/]",
                    total=1,
                    completed=1,
                    metadata="",
                    category="",
                    phase="",
                    response=lane.response,
                    show_spinner=False,
                    show_elapsed=False,
                    started_at=lane.started_at,
                )
        return progress

    def _recent_table(self) -> Table:
        show_response = self._console.width >= _RECENT_RESPONSE_MIN_WIDTH
        request_ratio = 3 if show_response else 1
        response_ratio = 4
        # Branch boundary: bounded Outcome width pays for Category before request/response space is allocated.
        outcome_width = (
            min(
                _RECENT_WIDE_OUTCOME_MAX_WIDTH,
                _RECENT_WIDE_OUTCOME_WIDTH + (self._console.width - _RECENT_RESPONSE_MIN_WIDTH) // 3,
            )
            if show_response
            else _RECENT_NARROW_OUTCOME_WIDTH
        )
        gate_width = _GATE_WIDTH if show_response else _NARROW_GATE_WIDTH
        table = Table(
            box=None,
            expand=True,
            pad_edge=False,
            show_header=True,
            header_style=Style(color=_ACTIVE, bold=False),
        )
        table.add_column("", width=1, no_wrap=True)
        table.add_column("Eval", width=4, justify="right", no_wrap=True)
        table.add_column("Model", width=_RECENT_MODEL_WIDTH, no_wrap=True, overflow="ellipsis")
        table.add_column("Category", width=_CATEGORY_WIDTH, no_wrap=True, overflow="ellipsis")
        table.add_column("Request", ratio=request_ratio, no_wrap=True, overflow="ellipsis")
        if show_response:
            table.add_column("Response", ratio=response_ratio, no_wrap=True, overflow="ellipsis")
        table.add_column("Outcome", width=outcome_width, no_wrap=True, overflow="ellipsis")
        table.add_column("Gates", width=gate_width, no_wrap=True)
        table.add_column("Calls", width=5, justify="right", no_wrap=True)
        table.add_column("Elapsed", width=7, justify="right", no_wrap=True)
        for result in reversed(self._recent):
            glyph, style = _outcome_glyph(result.outcome)
            row: list[RenderableType] = [
                Text(glyph, style=style),
                str(result.completion_index),
                _LeftEllipsisText(result.cell.model_id, style="dim"),
                _LeftEllipsisText(result.cell.category, style="dim"),
                _safe_text(result.request),
            ]
            if show_response:
                row.append(_LeftEllipsisText(result.response, style="dim"))
            row.extend(
                [
                    _outcome_text(result.outcome, result.reason, style),
                    _gate_text(result.checks, width=gate_width),
                    _call_text(result.tool_calls, result.tool_call_par),
                    _cell_duration(result.elapsed),
                ]
            )
            table.add_row(
                *row,
            )
        return table

    def _totals(self) -> Text:
        footer = Text()
        _append_status(footer, "✓", "passed", self._passes, _SUCCESS)
        footer.append("  ")
        _append_status(footer, _FAIL_GLYPH, "needs attention", self._failures, _WARNING)
        footer.append("  ")
        _append_status(footer, "!", "incomplete", self._incomplete, _ERROR)
        footer.append(f"  completed {self._completed}/{self._total}  tool calls {self._tool_calls}", style="dim")
        return footer

    def _print_line(self, event: MatrixProgressEvent, *, diagnostic: str | None = None) -> None:
        if event.state == "matrix_started":
            line = f"matrix started total={event.total or 0}"
        elif event.cell is None:
            return
        elif event.state == "cell_started":
            line = f"cell started {_cell_name(event)}"
        elif event.state in {"tool_started", "tool_finished"}:
            line = f"tool {'started' if event.state == 'tool_started' else 'finished'} {_cell_name(event)} {event.tool_name}"
        elif event.trace is not None:
            line = (
                f"cell finished {event.completion_index}/{event.total} {_cell_name(event)} "
                f"score={event.trace.score:.3f} tools={event.trace.tool_call_count} elapsed={_duration(event.elapsed or 0)}"
            )
            if diagnostic is not None:
                line = f"{line} error={diagnostic}"
        else:
            return
        self._console.print(line, markup=False, highlight=False, soft_wrap=True)


def _append_status(text: Text, glyph: str, label: str, count: int, style: str) -> None:
    """Append a colored status glyph while leaving surrounding text neutral."""
    text.append(glyph, style=style)
    text.append(f" {label} {count}")


def _gate_legend() -> Text:
    """Render a compact dim key for the Recent-table scoring gate glyphs."""
    legend = Text("Gates: ", style="dim")
    legend.append("■", style="dim")
    legend.append(" required, ", style="dim")
    legend.append("□", style="dim")
    legend.append(" optional; ", style="dim")
    legend.append("■", style=_SUCCESS)
    legend.append(" passed, ", style="dim")
    legend.append("■", style=_ERROR)
    legend.append(" failed", style="dim")
    return legend


def _compact_diagnostic(error: str | None) -> str | None:
    """Normalize and bound a trace error for durable terminal diagnostics."""
    if error is None:
        return None
    compact = " ".join(error.split())
    if len(compact) <= _MAX_DIAGNOSTIC_CHARS:
        return compact
    return f"{compact[: _MAX_DIAGNOSTIC_CHARS - 3]}..."


def _cell_metadata(cell: MatrixCellRef) -> str:
    """Return the dim secondary identity for a matrix cell."""
    return f"{cell.candidate_id} · {cell.model_id} · {cell.case_id}"


def _left_ellipsis(value: str, width: int) -> str:
    """Keep the rightmost model identifier text when a bounded metadata field overflows."""
    if len(value) <= width:
        return value
    return f"…{value[-(width - 1) :]}"


def _gate_text(checks: tuple[CheckResult, ...], *, width: int = _GATE_WIDTH) -> Text:
    """Render ordered scoring gates as compact requiredness/result glyphs."""
    gates = Text()
    visible_count = _visible_gate_count(len(checks), width)
    overflow = len(checks) - visible_count
    for check in checks[:visible_count]:
        glyph = "■" if check.required else "□"
        gates.append(glyph, style=_SUCCESS if check.passed else _ERROR)
    if overflow > 0:
        gates.append(f"+{overflow}", style="dim")
    return gates


def _call_text(tool_calls: int, par: int | None) -> Text:
    """Render tool-call efficiency relative to the scorer's supplied par."""
    if par is None:
        return Text(str(tool_calls))
    return Text(str(tool_calls), style=_SUCCESS if tool_calls <= par else _WARNING)


def _outcome_text(outcome: str, detail: str, style: str) -> Text:
    """Render failure detail directly, otherwise a semantic outcome label and explanation."""
    if outcome == "Needs attention":
        return Text(detail or outcome, style=style)
    rendered = Text()
    rendered.append(outcome, style=style)
    if detail:
        rendered.append(" — ")
        rendered.append(detail, style="not bold")
    return rendered


def _cell_duration(seconds: float) -> str:
    """Format one completed-cell duration compactly for the Recent table."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return _duration(seconds)


def _visible_gate_count(total: int, width: int = _GATE_WIDTH) -> int:
    """Return the maximum glyph count whose actual omitted-count suffix still fits."""
    if total <= width:
        return total
    for visible_count in range(width, -1, -1):
        omitted = total - visible_count
        if visible_count + len(str(omitted)) + 1 <= width:
            return visible_count
    raise AssertionError("gate overflow suffix must fit the configured width")


def _response_display(response: str | None) -> str:
    """Normalize final assistant text while keeping empty provider failures visually quiet."""
    if response is None:
        return "—"
    normalized = " ".join(response.split())
    return normalized or "—"


def _outcome_detail(checks: tuple[CheckResult, ...]) -> str:
    """Project the first failed required gate into a compact Recent-table explanation."""
    for check in checks:
        if check.required and not check.passed:
            return _humanize_required_failure(check)
    return ""


def _humanize_required_failure(check: CheckResult) -> str:
    """Translate known scoring feedback tokens without exposing raw tool payloads."""
    base = check.name.rsplit("_", 1)[0] if check.name.rsplit("_", 1)[-1].isdigit() else check.name
    if base == "model_error":
        return "Provider/model issue"
    if base == "meaningful_oracle":
        return "No scoring evidence defined"
    if base == "provenance_evidence_present":
        return _missing_feedback(check.feedback, "Missing evidence")
    if base == "tool_result_check":
        tool = _feedback_value(check.feedback, "tool") or "Tool"
        failures = _feedback_value(check.feedback, "failures")
        failure = failures.split(",")[0] if failures else "result_mismatch"
        return f"{tool}: {_humanize_tool_failure(failure)}"
    if base == "execution_ok":
        error = _feedback_value(check.feedback, "error")
        return f"Final tool failed ({_clean_token(error)})" if error else "Final tool failed"
    if base == "actions_match":
        return _humanize_action_failure(check.feedback)
    if base == "blocked_outcome":
        return _humanize_blocked_failure(check.feedback)
    if base == "tool_calls_within_max":
        calls = _feedback_value(check.feedback, "calls")
        maximum = _feedback_value(check.feedback, "max")
        return f"Tool-call limit exceeded ({calls}/{maximum})" if calls and maximum else "Tool-call limit exceeded"
    return f"{_clean_token(base)}: {_compact_feedback(check.feedback)}"


def _missing_feedback(feedback: str, prefix: str) -> str:
    """Return a missing-value projection from stable ``missing=`` scoring feedback."""
    missing = _feedback_value(feedback, "missing")
    return f"No {missing}" if missing else prefix


def _humanize_tool_failure(code: str) -> str:
    """Map one structured tool-result failure code using the HTML report vocabulary."""
    kind, separator, detail = code.partition(":")
    label = {
        "missing_entry_value": "missing expected value",
        "missing_entry_entity": "missing entity",
        "missing_entity": "missing entity",
        "missing_statistic": "missing statistic",
        "missing_automation": "missing automation",
        "empty_entity": "no rows for",
        "empty_statistic": "no rows for",
        "missing_successful_tool_result": "no successful result",
        "empty_output": "empty result",
        "empty_rows": "too few rows",
        "empty_entries": "too few logbook entries",
        "missing_entries": "no logbook entries",
        "missing_statistics": "no statistics",
        "missing_automations": "no automations",
        "unexpected_results": "unexpected results",
        "unverified_query_scope": "unverified query scope",
    }.get(kind, _clean_token(kind))
    return f"{label} {detail}" if separator and detail else label


def _humanize_action_failure(feedback: str) -> str:
    """Return the first stable action mismatch without surfacing raw action payloads."""
    key, _, value = feedback.partition("=")
    labels = {
        "unmatched": "Missing action",
        "extra": "Unexpected action",
        "duplicates": "Repeated action",
        "target_mismatch": "Wrong action target",
        "unexpected": "Unexpected actions",
    }
    label = labels.get(key, "Action mismatch")
    return f"{label}: {value}" if value else label


def _humanize_blocked_failure(feedback: str) -> str:
    """Summarize structured blocked-action failures without action payload details."""
    key, _, value = feedback.partition("=")
    if key == "successful_actions":
        return "Disallowed action succeeded"
    if key == "missing_rejected_action":
        return "No rejected action was observed"
    if key == "attempts":
        return f"Too many blocked attempts ({value.replace(' max=', '/')})"
    if key == "error_keys":
        return f"Unexpected block error: {value}"
    return "Blocked-action mismatch"


def _feedback_value(feedback: str, key: str) -> str | None:
    """Extract one stable ``key=value`` token from scoring feedback."""
    prefix = f"{key}="
    for token in feedback.replace(";", " ").split():
        if token.startswith(prefix):
            return token.removeprefix(prefix)
    return None


def _clean_token(value: str | None) -> str:
    """Convert one internal token into compact human-readable text."""
    return "unknown issue" if not value else value.replace("_", " ").replace(":", ": ")


def _compact_feedback(feedback: str) -> str:
    """Keep an unknown structured feedback contract brief enough for one table cell."""
    compact = " ".join(feedback.split())
    return compact[:80] if compact else "failed"


def _lane_phase(active_tools: Counter[str]) -> str:
    """Return the current human-readable phase for an active matrix lane."""
    if not active_tools:
        return "Waiting"
    phases = {
        "execute_home_code": "Using the Home Assistant sandbox",
        "get_history": "Reading history",
        "get_statistics": "Calculating statistics",
        "get_logbook": "Reading the logbook",
        "get_automation": "Reading automations",
    }
    return " + ".join(phases.get(tool_name, "Reading current home information") for tool_name in active_tools)


def _outcome_glyph(outcome: str) -> tuple[str, str]:
    """Return a semantic glyph and restrained style for a completed result."""
    return {
        "Completed": ("✓", _SUCCESS),
        "Needs attention": (_FAIL_GLYPH, _WARNING),
        "Could not complete": ("!", _ERROR),
    }[outcome]


def _safe_text(value: str) -> Text:
    """Render a dynamic string without treating user text as Rich markup."""
    return Text.from_markup(escape(value))


def _cell_name(event: MatrixProgressEvent) -> str:
    """Return a fallback-safe identity for non-TTY lifecycle lines."""
    if event.request is not None:
        return event.request
    assert event.cell is not None
    return f"{event.cell.candidate_id}/{event.cell.model_id}/{event.cell.case_id}"


def _duration(seconds: float) -> str:
    """Format monotonic durations compactly for terminal presentation."""
    minutes, remaining = divmod(max(0, int(seconds)), 60)
    return f"{minutes}:{remaining:02d}" if minutes else f"{seconds:.1f}s"
