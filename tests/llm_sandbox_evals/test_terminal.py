from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.terminal import _WARNING, MatrixTerminalReporter, _outcome_text
import pytest
from rich.console import Console
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


def _column_widths(table: Table, console: Console) -> dict[str, int]:
    """Return Rich's actual terminal allocation, including each column's padding."""
    widths = table._calculate_column_widths(console, console.options)
    return dict(zip((str(column.header) for column in table.columns), widths, strict=True))
