from io import StringIO
from typing import Literal

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.schema import CaseTrace, CheckResult
from llm_sandbox_evals.terminal import _WARNING, MatrixTerminalReporter, _outcome_text
import pytest
from rich.console import Console, Group
from rich.progress import Progress
from rich.table import Table


@pytest.mark.parametrize(
    ("width", "prior_request_width", "prior_response_width", "prior_running_request_width"),
    [
        pytest.param(80, 4, None, 20, id="narrow"),
        pytest.param(120, 20, 26, 60, id="response-threshold"),
        pytest.param(200, 42, 55, 140, id="wide"),
    ],
)
def test_terminal_category_columns_preserve_request_space(
    width: int,
    prior_request_width: int,
    prior_response_width: int | None,
    prior_running_request_width: int,
) -> None:
    reporter = MatrixTerminalReporter()
    reporter._console = Console(width=width, force_terminal=True)
    cell = MatrixCellRef("case", "candidate", "model", "home", "state_read")
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="request"))

    recent = reporter._recent_table()
    recent_widths = _column_widths(recent, reporter._console)
    running = reporter._lanes_progress()
    running_widths = running.make_tasks_table(running.tasks)._calculate_column_widths(
        reporter._console, reporter._console.options
    )

    assert [column.header for column in recent.columns] == [
        "",
        "Eval",
        "Model",
        "Category",
        "Request",
        *(["Response"] if prior_response_width is not None else []),
        "Outcome",
        "Gates",
        "Calls",
        "Elapsed",
    ]
    assert recent_widths["Category"] == 13
    assert recent_widths["Request"] >= prior_request_width
    if prior_response_width is not None:
        assert recent_widths["Response"] >= prior_response_width
    assert running_widths[1] >= prior_running_request_width


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        pytest.param("Missing action: light.turn_on", "Missing action: light.turn_on", id="humanized-detail"),
        pytest.param("", "Needs attention", id="empty-detail-fallback"),
    ],
)
def test_needs_attention_outcome_uses_amber_detail_only(detail: str, expected: str) -> None:
    outcome = _outcome_text("Needs attention", detail, _WARNING)

    assert outcome.plain == expected
    assert outcome.style == _WARNING


def test_terminal_counts_tool_call_limit_as_failure_without_error_diagnostic() -> None:
    stream = StringIO()
    reporter = MatrixTerminalReporter()
    reporter._console = Console(file=stream, force_terminal=False)
    cell = MatrixCellRef("case", "candidate", "model", "home", "state_read")
    trace = CaseTrace(
        case_id=cell.case_id,
        category=cell.category,
        candidate_id=cell.candidate_id,
        model_id=cell.model_id,
        score=0.0,
        output="",
        tool_call_count=3,
        recorded_actions=(),
        checks=(CheckResult("tool_calls_exceeded", False, True, "limit reached"),),
        error=None,
    )

    reporter.handle(MatrixProgressEvent("matrix_started", total=1))
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="request"))
    reporter.handle(MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=1, total=1))
    reporter.finish(overall_mean=0.0, run_dir="run", report_html="report.html")

    output = stream.getvalue()
    assert "failed=1" in output
    assert "incomplete=0" in output
    assert "error=" not in output


def test_live_refreshes_replaced_lanes_before_printing_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Literal["diagnostic", "update"]] = []
    updates: list[tuple[object, bool]] = []
    reporter = MatrixTerminalReporter()
    completed = MatrixCellRef("completed", "candidate", "model", "home", "state_read")
    replacement = MatrixCellRef("replacement", "candidate", "model", "home", "state_read")
    trace = CaseTrace(
        case_id=completed.case_id,
        category=completed.category,
        candidate_id=completed.candidate_id,
        model_id=completed.model_id,
        score=0.0,
        output="",
        tool_call_count=0,
        recorded_actions=(),
        checks=(CheckResult("execution_ok", False, True, ""),),
        error="provider failure",
    )

    def update(renderable: object, *, refresh: bool = False) -> None:
        events.append("update")
        updates.append((renderable, refresh))

    def print_diagnostic(_event: MatrixProgressEvent, _diagnostic: str) -> None:
        events.append("diagnostic")

    reporter._live = type("LiveCapture", (), {"update": staticmethod(update)})()
    monkeypatch.setattr(reporter, "_print_live_diagnostic", print_diagnostic)
    reporter.handle(MatrixProgressEvent("cell_started", cell=completed, request="completed request"))
    reporter.handle(MatrixProgressEvent("cell_started", cell=replacement, request="replacement request"))
    events.clear()
    updates.clear()

    reporter.handle(MatrixProgressEvent("cell_finished", cell=completed, trace=trace, completion_index=1, total=2))

    assert events == ["update", "diagnostic"]
    assert updates[0][1] is True
    captured_renderable = updates[0][0]
    assert isinstance(captured_renderable, Group)
    rendered_lanes = next(
        renderable
        for renderable in captured_renderable.renderables
        if isinstance(renderable, Progress) and renderable.tasks and renderable.tasks[0].total is None
    )
    assert len(rendered_lanes.tasks) == 1
    assert replacement in reporter._lanes
    assert completed not in reporter._lanes
    assert rendered_lanes.tasks[0].fields["started_at"] == reporter._lanes[replacement].started_at


def _column_widths(table: Table, console: Console) -> dict[str, int]:
    """Return Rich's actual terminal allocation, including each column's padding."""
    widths = table._calculate_column_widths(console, console.options)
    return dict(zip((str(column.header) for column in table.columns), widths, strict=True))
