from io import StringIO
from pathlib import Path

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EvalDiagnostics,
    RequiredAction,
    ToolEvent,
)
from llm_sandbox_evals.terminal import MatrixTerminalReporter, _duration
import pytest
from rich.console import Console


def _config() -> EvalConfig:
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile="balanced",
        cases=None,
        homes=None,
        runs_dir=Path("runs/eval-test"),
        max_tool_calls=10,
        model_timeout=75.0,
    )


def _reporter(*, human: bool = True, escape_available: bool = True) -> MatrixTerminalReporter:
    return MatrixTerminalReporter(
        _config(), run_id="run-1", run_dir="runs/run-1", human=human, escape_available=escape_available
    )


def _cell(case_id: str = "case", reasoning_effort: str | None = None) -> MatrixCellRef:
    return MatrixCellRef(case_id, "baseline", "stub", "home_minimal", reasoning_effort=reasoning_effort)


def _trace(
    *,
    state: str = "correct",
    action_reason: str | None = "ok",
    failure: str | None = None,
    cap_exhausted: bool = False,
    tool_calls: int = 0,
    elapsed: float | None = None,
    request: str = "Turn on bedroom light",
    answer: str | None = None,
) -> CaseTrace:
    return CaseTrace(
        case_id="case",
        candidate_id="baseline",
        model_id="stub",
        answer=answer,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        outcome=CaseOutcome(state, action_reason),
        action_result=ActionResult(state == "correct", action_reason or "ok"),
        action_ledger=ActionLedger(),
        tool_events=(
            ToolEvent(
                "execute_home_code",
                {"code": "secret-tool-arg"},
                {"actions": [{"secret": "payload"}]},
            ),
        ),
        diagnostics=EvalDiagnostics(
            tool_calls=tool_calls,
            cap_exhausted=cap_exhausted,
            failure=failure,
            elapsed_seconds=elapsed,
        ),
        user_request=request,
    )


def _feed(reporter: MatrixTerminalReporter, *cells: tuple[MatrixCellRef, CaseTrace]) -> None:
    reporter._state.total = len(cells)
    for index, (cell, trace) in enumerate(cells, start=1):
        reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request=trace.user_request, total=len(cells)))
        reporter.handle(
            MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=index, total=len(cells))
        )


def test_human_render_has_no_phase_activity_waiting_or_response_row() -> None:
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (
            _cell(),
            _trace(
                state="incomplete",
                action_reason=None,
                failure="timeout",
                request="Turn on bedroom light",
                answer="raw model response text",
            ),
        ),
    )

    console = Console(width=160, force_terminal=False, record=True)
    console.print(reporter._render())
    output = console.export_text()

    # No phase or Activity column exists in the initial terminal (streaming deferred).
    assert "Activity" not in output
    assert "Phase" not in output
    # The misleading Waiting label and the ephemeral response row are removed.
    assert "Waiting" not in output
    assert "Response" not in output
    # Raw model response, tool args, and payloads never reach Live (the request is shown, the answer is not).
    assert "raw model response text" not in output
    assert "secret-tool-arg" not in output
    assert "payload" not in output


def test_recent_table_has_a_single_result_column_with_state_and_cause() -> None:
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (_cell(), _trace(state="incomplete", action_reason=None, failure="provider_error", tool_calls=2, elapsed=3.0)),
    )

    table = reporter._recent_table()
    column_headers = [column.header for column in table.columns]
    # Exactly one semantic Result column; no separate Outcome + Reason split.
    assert column_headers.count("Result") == 1
    assert "Reason" not in column_headers
    assert "Outcome" not in column_headers

    console = Console(width=160, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()
    # The Result column renders state·cause, so operational failures never read as action_mismatch.
    assert "incomplete·provider_error" in output
    assert "action_mismatch" not in output


def test_lanes_show_variant_and_tool_cap_without_phase() -> None:
    reporter = _reporter(human=True)
    reporter._state.total = 1
    reporter.handle(MatrixProgressEvent("cell_started", cell=_cell(reasoning_effort="high"), request="Turn on light"))

    table = reporter._lanes_table()
    column_headers = [column.header for column in table.columns]
    assert "Variant" in column_headers
    assert "Tools / cap" in column_headers
    assert "Activity" not in column_headers

    console = Console(width=160, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()
    assert "stub(high)" in output
    assert "0 / 10" in output


def test_machine_events_emit_stable_kv_without_raw_payload() -> None:
    stream = Console(width=200, force_terminal=False, file=__import__("io").StringIO())
    reporter = _reporter(human=False)
    reporter._console = stream
    cell = _cell()
    trace = _trace(state="incomplete", action_reason=None, failure="timeout", request="secret request body")

    reporter.handle(MatrixProgressEvent("matrix_started", total=1))
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="secret request body"))
    reporter.handle(
        MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=1, total=1)
    )

    output = stream.file.getvalue()
    assert "matrix_started total=1" in output
    assert "cell_finished index=1 result=incomplete·timeout" in output
    # Redirected output stays deterministic KV with no raw request text.
    assert "secret request body" not in output


def test_durable_final_emits_counts_and_artifact_path_once() -> None:
    from llm_sandbox_evals.terminal import render_durable_final

    reporter = _reporter(human=True)
    _feed(reporter, (_cell(), _trace(state="incorrect", action_reason="wrong_target", tool_calls=2)))

    console = Console(width=160, force_terminal=False, record=True)
    console.print(render_durable_final(reporter._state, run_dir="runs/run-1", report_html="runs/run-1/report.html"))
    output = console.export_text()

    assert "Quality" in output
    assert "Coverage" in output
    # The artifact directory and report.html each appear on one dedicated line in the durable final.
    assert "Artifacts: runs/run-1" in output
    assert "report.html: runs/run-1/report.html" in output


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        pytest.param(0.4, "0.4s", id="subsecond"),
        pytest.param(10.4, "10.4s", id="seconds"),
        pytest.param(61.0, "1:01", id="minute"),
    ],
)
def test_duration_is_compact(seconds: float, expected: str) -> None:
    assert _duration(seconds) == expected


@pytest.mark.parametrize(
    ("escape_available", "expected_hint", "absent_hint"),
    [
        pytest.param(True, "Escape", "Ctrl+C", id="escape-available"),
        pytest.param(False, "Ctrl+C", "Escape", id="ctrl-c-only"),
    ],
)
def test_cancel_hint_distinguishes_escape_from_ctrl_c_only(
    escape_available: bool, expected_hint: str, absent_hint: str
) -> None:
    interactive = _reporter(human=True, escape_available=escape_available)
    redirected = _reporter(human=False, escape_available=escape_available)

    # The orientation panel surfaces the correct cancellation mechanism for the active stream mode.
    orientation = Console(width=160, force_terminal=False, record=True)
    orientation.print(interactive._orientation())
    orientation_text = orientation.export_text()
    assert expected_hint in orientation_text
    assert absent_hint not in orientation_text

    # The transient Live frame carries the same hint while a run is active.
    frame = Console(width=160, force_terminal=False, record=True)
    frame.print(interactive._render())
    assert expected_hint in frame.export_text()

    # Redirected (non-TTY) runs emit deterministic KV only and never print a human cancel hint.
    stream = StringIO()
    redirected._console = Console(width=200, force_terminal=False, file=stream)
    redirected.handle(MatrixProgressEvent("matrix_started", total=1))
    assert expected_hint not in stream.getvalue()
