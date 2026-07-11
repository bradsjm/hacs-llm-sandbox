"""Stderr-only terminal presentation for native eval matrix lifecycle events."""

from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Self

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TextColumn
from rich.rule import Rule
from rich.table import Column, Table
from rich.text import Text

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.schema import CheckResult
from llm_sandbox_evals.scoring import is_incomplete

_ACTIVE = "#38b6ca"
_PASS = "#55c97c"
_FAIL = "#f2705f"
_INCOMPLETE = "#d9a514"
_FAIL_GLYPH = "\u2717"
_REFRESH_PER_SECOND = 4
_RECENT_RESULTS = 5


@dataclass(slots=True)
class _Lane:
    """Current visible state for one concurrently evaluated matrix cell."""

    cell: MatrixCellRef
    request: str
    started_at: float
    active_tools: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class _RecentResult:
    """Compact completed-cell data retained by the terminal presentation."""

    cell: MatrixCellRef
    request: str
    outcome: str
    reason: str
    tool_calls: int


class _ElapsedColumn(ProgressColumn):
    """Render a monotonic task duration every time Rich refreshes a progress row."""

    def __init__(self, prefix: str = "", *, table_column: Column | None = None) -> None:
        """Initialize an optional neutral label before the duration."""
        super().__init__(table_column=table_column)
        self._prefix = prefix

    def render(self, task: Task) -> Text:
        """Return elapsed time from the stable task start rather than a stale field."""
        started_at = task.fields["started_at"]
        assert isinstance(started_at, float)
        return Text(f"{self._prefix}{_duration(perf_counter() - started_at)}", style="dim")


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
        elif event.state == "cell_finished" and event.cell is not None and event.trace is not None:
            self._finish_cell(event)

        if self._live is not None:
            self._live.update(self._render())
        elif not self._console.is_terminal:
            self._print_line(event)

    def finish(self, *, overall_mean: float, run_dir: str, report_html: str) -> None:
        """Print the successful post-artifact summary after the Live display ends."""
        elapsed = _duration(perf_counter() - self._started_at)
        if self._console.is_terminal:
            self._stop_live()
            summary = Text()
            _append_status(summary, "✓", "passed", self._passes, _PASS)
            summary.append("  ")
            _append_status(summary, _FAIL_GLYPH, "needs attention", self._failures, _FAIL)
            summary.append("  ")
            _append_status(summary, "!", "incomplete", self._incomplete, _INCOMPLETE)
            summary.append(
                f"\ncompleted {self._completed}/{self._total}  elapsed {elapsed}  overall mean {overall_mean:.3f}"
            )
            summary.append(f"\nreport.json {run_dir}/report.json", style="dim")
            summary.append(f"\nreport.html {report_html}", style="dim")
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
            outcome, reason = "Could not complete", _failure_reason(trace.checks)
            self._incomplete += 1
        elif trace.score > 0:
            outcome, reason = "Completed", "All required checks passed"
            self._passes += 1
        else:
            outcome, reason = "Needs attention", _failure_reason(trace.checks)
            self._failures += 1
        self._recent.append(_RecentResult(cell, request, outcome, reason, trace.tool_call_count))

    def _render(self) -> Group:
        return Group(
            Text("LLM Sandbox evaluation", style="bold"),
            self._overall_progress(),
            Rule("Running now", style="dim"),
            self._lanes_progress(),
            Rule("Recent", style="dim"),
            self._recent_table(),
            self._totals(),
        )

    def _overall_progress(self) -> Progress:
        progress = Progress(
            TextColumn("Overall", table_column=Column(no_wrap=True)),
            BarColumn(bar_width=None, complete_style=_ACTIVE, finished_style=_ACTIVE),
            TextColumn(
                "{task.fields[completed_display]}/{task.fields[total_display]}",
                table_column=Column(no_wrap=True),
            ),
            _ElapsedColumn("elapsed "),
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
            SpinnerColumn(style=_ACTIVE),
            TextColumn(
                "{task.description}",
                markup=True,
                table_column=Column(ratio=4, no_wrap=True, overflow="ellipsis"),
            ),
            TextColumn(
                "{task.fields[metadata]}",
                markup=True,
                style="dim",
                table_column=Column(ratio=1, no_wrap=True, overflow="ellipsis"),
            ),
            TextColumn(
                "{task.fields[phase]}",
                markup=True,
                table_column=Column(ratio=2, no_wrap=True, overflow="ellipsis"),
            ),
            _ElapsedColumn(table_column=Column(width=6, no_wrap=True)),
            expand=True,
        )
        for lane in self._lanes.values():
            progress.add_task(
                escape(lane.request),
                total=None,
                metadata=escape(_cell_metadata(lane.cell)),
                phase=escape(_lane_phase(lane.active_tools)),
                started_at=lane.started_at,
            )
        return progress

    def _recent_table(self) -> Table:
        table = Table(box=None, expand=True, pad_edge=False, show_header=True, header_style="dim")
        table.add_column("", width=1, no_wrap=True)
        table.add_column("Request", ratio=4, no_wrap=True, overflow="ellipsis")
        table.add_column("Outcome", no_wrap=True, overflow="ellipsis")
        table.add_column("Reason", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column("Tools", justify="right", no_wrap=True)
        for result in reversed(self._recent):
            glyph, style = _outcome_glyph(result.outcome)
            table.add_row(
                Text(glyph, style=style),
                _safe_text(result.request),
                result.outcome,
                result.reason,
                str(result.tool_calls),
            )
        return table

    def _totals(self) -> Text:
        footer = Text()
        _append_status(footer, "✓", "passed", self._passes, _PASS)
        footer.append("  ")
        _append_status(footer, _FAIL_GLYPH, "needs attention", self._failures, _FAIL)
        footer.append("  ")
        _append_status(footer, "!", "incomplete", self._incomplete, _INCOMPLETE)
        footer.append(f"  completed {self._completed}/{self._total}  tool calls {self._tool_calls}", style="dim")
        return footer

    def _print_line(self, event: MatrixProgressEvent) -> None:
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
        else:
            return
        self._console.print(line, markup=False, highlight=False, soft_wrap=True)


def _append_status(text: Text, glyph: str, label: str, count: int, style: str) -> None:
    """Append a colored status glyph while leaving surrounding text neutral."""
    text.append(glyph, style=style)
    text.append(f" {label} {count}")


def _cell_metadata(cell: MatrixCellRef) -> str:
    """Return the dim secondary identity for a matrix cell."""
    return f"{cell.candidate_id} · {cell.model_id} · {cell.case_id}"


def _failure_reason(checks: tuple[CheckResult, ...]) -> str:
    """Return a short friendly first-failure label without exposing raw feedback."""
    for check in checks:
        if check.required and not check.passed:
            return {
                "model_error": "Model or provider unavailable",
                "tool_calls_exceeded": "Tool-call limit reached",
                "execution_ok": "Last tool call did not succeed",
                "actions_match": "Requested action did not match",
                "blocked_outcome": "Action safety check did not pass",
                "tool_calls_within_max": "Too many tool calls",
            }.get(check.name, "Missing required result")
    return "Missing required result"


def _lane_phase(active_tools: Counter[str]) -> str:
    """Return the current human-readable phase for an active matrix lane."""
    if not active_tools:
        return "Waiting for the assistant"
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
        "Completed": ("✓", _PASS),
        "Needs attention": (_FAIL_GLYPH, _FAIL),
        "Could not complete": ("!", _INCOMPLETE),
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
