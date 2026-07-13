from io import StringIO
from typing import Literal

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EvalDiagnostics,
    RequiredAction,
)
from llm_sandbox_evals.terminal import MatrixTerminalReporter, _cell_duration, _outcome_text
import pytest
from rich.console import Console


@pytest.mark.parametrize(
    ("outcome", "detail", "expected"),
    [
        pytest.param("correct", "ok", "correct — ok", id="correct"),
        pytest.param("incorrect", "action_mismatch", "action_mismatch", id="incorrect"),
        pytest.param("incomplete", "provider_error", "incomplete — provider_error", id="incomplete"),
    ],
)
def test_terminal_renders_action_outcomes_without_operational_scoring(
    outcome: str, detail: str, expected: str
) -> None:
    rendered = _outcome_text(outcome, detail, "style")

    assert rendered.plain == expected
    assert "par" not in rendered.plain


def test_terminal_reports_cap_as_incorrect_diagnostic() -> None:
    stream = StringIO()
    reporter = MatrixTerminalReporter()
    reporter._console = Console(file=stream, force_terminal=False)
    cell = MatrixCellRef("case", "candidate", "model", "home")
    trace = _trace("incorrect", cap_exhausted=True, tool_calls=3)

    reporter.handle(MatrixProgressEvent("matrix_started", total=1))
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="request"))
    reporter.handle(MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=1, total=1))
    reporter.finish(overall_correct_rate=0.0, run_dir="run", report_html="report.html")

    output = stream.getvalue()
    assert "incorrect=1" in output
    assert "incomplete=0" in output
    assert "Gates" not in output
    assert "par" not in output


def test_terminal_recent_table_places_calls_and_elapsed_in_their_columns() -> None:
    reporter = MatrixTerminalReporter()
    console = Console(width=200, force_terminal=False, record=True)
    reporter._console = console
    cell = MatrixCellRef("case", "candidate", "model", "home")
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="request"))
    reporter.handle(
        MatrixProgressEvent(
            "cell_finished", cell=cell, trace=_trace("correct", tool_calls=3, elapsed=2.0), completion_index=1, total=1
        )
    )

    table = reporter._recent_table()
    assert [column.header for column in table.columns][-3:] == ["Reason", "Calls", "Elapsed"]
    console.print(table)
    output = console.export_text()
    assert "Calls" in output
    assert "Elapsed" in output
    assert "3" in output
    assert "2s" in output


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(0.4, "0.4s", id="subsecond"),
        pytest.param(10.4, "10s", id="seconds"),
        pytest.param(61.0, "1:01", id="minute"),
    ],
)
def test_cell_duration_is_compact(seconds: float, expected: str) -> None:
    assert _cell_duration(seconds) == expected


def _trace(
    state: Literal["correct", "incorrect", "incomplete"],
    *,
    cap_exhausted: bool = False,
    tool_calls: int = 0,
    elapsed: float | None = None,
) -> CaseTrace:
    return CaseTrace(
        case_id="case",
        candidate_id="candidate",
        model_id="model",
        answer=None,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        outcome=CaseOutcome(state, "ok" if state == "correct" else "action_mismatch"),
        action_result=ActionResult(state == "correct", "ok" if state == "correct" else "action_mismatch"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(tool_calls=tool_calls, cap_exhausted=cap_exhausted, elapsed_seconds=elapsed),
    )
