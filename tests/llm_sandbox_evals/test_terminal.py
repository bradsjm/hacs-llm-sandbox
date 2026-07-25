from io import StringIO
from pathlib import Path
import unicodedata

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import LanePhaseEvent, MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.phases import LanePhase
from llm_sandbox_evals.presentation import (
    JudgeFailure,
    JudgePresentation,
    PresentationCell,
    PresentationState,
    ReportPresentationModel,
)
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalDiagnostics,
    ExecutionError,
    RequiredAction,
    ToolEvent,
)
from llm_sandbox_evals.terminal import (
    MatrixTerminalReporter,
    _operational_issue_cells,
    _operational_issue_detail,
    render_durable_final,
)
import pytest
from rich.console import Console
from rich.panel import Panel


def _config(*, concurrency: int = 5) -> EvalConfig:
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile="balanced",
        cases=None,
        homes=None,
        runs_dir=Path("runs/eval-test"),
        max_tool_calls=10,
        model_timeout=75.0,
        concurrency=concurrency,
    )


def _reporter(*, human: bool = True, concurrency: int = 5) -> MatrixTerminalReporter:
    return MatrixTerminalReporter(
        _config(concurrency=concurrency),
        run_id="run-1",
        run_dir="runs/run-1",
        human=human,
        escape_available=True,
    )


def _cell(
    case_id: str = "case",
    reasoning_effort: str | None = None,
    request_variant_id: str = "canonical",
    model_id: str = "stub",
) -> MatrixCellRef:
    return MatrixCellRef(
        case_id, request_variant_id, "baseline", model_id, "home_minimal", reasoning_effort=reasoning_effort
    )


def _trace(
    *,
    case_id: str = "case",
    candidate_id: str = "baseline",
    state: str = "correct",
    score_reason: str | None = "ok",
    scoring_mode: str | None = "actions",
    failure: str | None = None,
    cap_exhausted: bool = False,
    tool_calls: int = 0,
    elapsed: float | None = None,
    request: str = "Turn on bedroom light",
    answer: str | None = None,
    model_id: str = "stub",
    request_variant_id: str = "canonical",
    reasoning_effort: str | None = None,
    provider_error: str | None = None,
    execution_error: ExecutionError | None = None,
) -> CaseTrace:
    return CaseTrace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        request_variant_id=request_variant_id,
        request_text=request,
        category="test",
        reasoning_effort=reasoning_effort,
        answer=answer,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        desired_entities=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome(state, scoring_mode, score_reason),
        action_result=ActionResult(state == "correct", score_reason or "ok"),
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
        provider_error=provider_error,
        execution_error=execution_error,
    )


def _feed(reporter: MatrixTerminalReporter, *cells: tuple[MatrixCellRef, CaseTrace]) -> None:
    reporter._state.total = len(cells)
    for index, (cell, trace) in enumerate(cells, start=1):
        reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request=trace.request_text, total=len(cells)))
        reporter.handle(
            MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=index, total=len(cells))
        )


def _report_cell(
    case_id: str,
    *,
    candidate_id: str = "baseline",
    model_id: str = "stub",
    request_variant_id: str = "canonical",
    reasoning_effort: str | None = None,
    request: str = "Turn on bedroom light",
    provider_error: str | None = None,
    judge: JudgePresentation,
) -> PresentationCell:
    trace = _trace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        request_variant_id=request_variant_id,
        reasoning_effort=reasoning_effort,
        request=request,
        provider_error=provider_error,
    )
    return PresentationCell(
        case_id,
        request_variant_id,
        trace.category,
        candidate_id,
        model_id,
        f"{model_id}({reasoning_effort or 'default'})",
        trace,
        {},
        judge,
    )


def _report_model(*cells: PresentationCell) -> ReportPresentationModel:
    return ReportPresentationModel(
        cells,
        {
            "judge_model": "openai-chat:judge-1",
            "judge_rubric_id": "llm_sandbox_code_quality",
            "judge_rubric_version": 7,
        },
    )


def _report_state(report_model: ReportPresentationModel) -> PresentationState:
    return PresentationState(total=len(report_model.cells), completed=list(report_model.cells))


def _normalized_rich_text(output: str) -> str:
    without_borders = "".join(
        " " if unicodedata.name(character, "").startswith("BOX DRAWINGS") else character for character in output
    )
    return " ".join(without_borders.split())


def _assert_actionable_operational_issue_output(output: str, *, detail_fragments: tuple[str, ...]) -> None:
    normalized = _normalized_rich_text(output)
    assert "Operational issues" in normalized
    assert "#" in normalized
    assert "2" in normalized.split()
    assert "rate_limit" in normalized
    assert "cerebras-llama-3.3" in normalized
    assert "429" in normalized
    assert "token_quota_exceeded" in normalized
    assert "alpha/case-a" in normalized
    assert "zeta/case-z" in normalized
    for fragment in detail_fragments:
        assert fragment in normalized
    assert "…" not in output
    assert "..." not in output


def test_human_render_shows_initial_queued_activity_without_waiting_or_response_row() -> None:
    reporter = _reporter(human=True)
    reporter.handle(MatrixProgressEvent("cell_started", cell=_cell(), request="Turn on bedroom light"))

    console = Console(width=160, force_terminal=False, record=True)
    console.print(reporter._render())
    output = console.export_text()

    assert "Activity" in output
    assert "queued" in output
    # The misleading Waiting label and the ephemeral response row are removed.
    assert "Waiting" not in output
    assert "Response" not in output


def test_recent_table_has_a_single_result_column_with_state_and_cause() -> None:
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (_cell(), _trace(state="incomplete", score_reason=None, failure="provider_error", tool_calls=2, elapsed=3.0)),
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


def test_recent_table_renders_newest_result_at_bottom() -> None:
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (_cell("case-oldest"), _trace(case_id="case-oldest", request="oldest request")),
        (_cell("case-middle"), _trace(case_id="case-middle", request="middle request")),
        (_cell("case-newest"), _trace(case_id="case-newest", request="newest request")),
    )

    console = Console(width=160, force_terminal=False, record=True)
    console.print(reporter._recent_table())
    output = console.export_text()

    assert output.index("oldest request") < output.index("middle request") < output.index("newest request")


def test_lanes_start_queued_with_activity_variant_and_tool_cap() -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=160, force_terminal=False, file=StringIO())
    reporter._state.total = 1
    reporter.handle(MatrixProgressEvent("cell_started", cell=_cell(reasoning_effort="high"), request="Turn on light"))

    table = reporter._lanes_table()
    column_headers = [column.header for column in table.columns]
    assert "Variant" in column_headers
    assert "Tools / cap" in column_headers
    assert "Activity" in column_headers

    console = Console(width=160, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()
    assert "stub(high)" in output
    assert "queued" in output
    assert "0 / 10" in output


def test_judging_phase_renders_its_phase_label() -> None:
    reporter = _reporter(human=True)
    reporter._state.total = 1
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    reporter.handle_phase(LanePhaseEvent(cell, "scoring"))
    reporter._lanes_table()

    reporter.handle_phase(LanePhaseEvent(cell, "judging"))
    table = reporter._lanes_table()
    console = Console(width=160, force_terminal=False, record=True)
    console.print(table)

    assert "judging" in console.export_text()


@pytest.mark.parametrize(
    ("phase", "expected_activity"),
    [
        pytest.param("running_tool", "run · execute_home_code", id="running-tool"),
        pytest.param("processing_tool_result", "result · execute_home_code", id="processing-tool-result"),
    ],
)
def test_real_phase_renders_safe_tool_activity_without_stream_payloads(
    phase: LanePhase, expected_activity: str
) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=102, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light"))
    reporter.handle_phase(LanePhaseEvent(cell, "preparing_tool_call", "provider-private-tool-name"))
    preparing_console = Console(width=102, force_terminal=False, record=True)
    preparing_console.print(reporter._lanes_table())
    preparing_output = preparing_console.export_text()
    assert reporter.state.lanes[cell].tool_name is None
    reporter.handle(
        MatrixProgressEvent("response_received", cell=cell, response="private reasoning and response text")
    )
    reporter.handle_phase(LanePhaseEvent(cell, phase, "execute_home_code"))

    active_console = Console(width=102, force_terminal=False, record=True)
    active_console.print(reporter._lanes_table())
    active_output = active_console.export_text()
    assert reporter.state.lanes[cell].tool_name == "execute_home_code"
    reporter.handle(
        MatrixProgressEvent(
            "cell_finished",
            cell=cell,
            trace=_trace(answer="private reasoning and response text"),
            completion_index=1,
            total=1,
        )
    )
    finished_console = Console(width=102, force_terminal=False, record=True)
    finished_console.print(reporter._render())
    output = active_output + finished_console.export_text()

    assert "Activity" in output
    assert expected_activity in output
    assert "…" not in active_output
    assert "execute_" in output
    assert "provider-private-tool-name" not in preparing_output
    assert "provider-private-tool-name" not in output
    assert "private reasoning and response text" not in output
    assert "secret-tool-arg" not in output
    assert "payload" not in output


@pytest.mark.parametrize(
    ("width", "expected_headers"),
    [
        pytest.param(
            80,
            ["", "Request", "Activity", "Elapsed / timeout", "Tools / cap"],
            id="narrow",
        ),
        pytest.param(
            102,
            ["", "Request", "Activity", "Variant", "Elapsed / timeout", "Tools / cap"],
            id="wide",
        ),
    ],
)
def test_activity_idle_placeholder_renders_at_narrow_and_wide_widths(width: int, expected_headers: list[str]) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=width, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    reporter.handle(MatrixProgressEvent("cell_finished", cell=cell, trace=_trace(), completion_index=1, total=1))

    table = reporter._lanes_table()
    console = Console(width=width, force_terminal=False, record=True)
    console.print(table)

    assert [str(column.header) for column in table.columns] == expected_headers
    assert "Activity" in console.export_text()


def test_lanes_table_reserves_one_row_before_matrix_started() -> None:
    """Before matrix_started, reserve one safe row regardless of configured concurrency."""
    reporter = _reporter(human=True, concurrency=100)
    reporter._console = Console(width=80, force_terminal=False, file=StringIO())

    table = reporter._lanes_table()

    assert len(table.rows) == 1


@pytest.mark.parametrize(
    ("concurrency", "total", "active", "finished", "expected_rows", "width"),
    [
        pytest.param(5, 2, 1, 0, 2, 160, id="total-below-concurrency-one-active"),
        pytest.param(5, 2, 1, 1, 2, 160, id="total-below-concurrency-all-finished"),
        pytest.param(2, 5, 1, 0, 2, 160, id="total-above-concurrency-one-active"),
        pytest.param(2, 5, 2, 0, 2, 160, id="total-above-concurrency-two-active"),
        pytest.param(2, 5, 2, 2, 2, 80, id="total-above-concurrency-all-finished-narrow"),
        pytest.param(3, 3, 2, 0, 3, 160, id="total-equals-concurrency-partial-active"),
    ],
)
def test_lanes_table_reserves_capacity_after_matrix_started(
    concurrency: int, total: int, active: int, finished: int, expected_rows: int, width: int
) -> None:
    """Running section always has min(concurrency, total) rows: active lanes plus placeholders."""
    reporter = _reporter(human=True, concurrency=concurrency)
    reporter._console = Console(width=width, force_terminal=False, file=StringIO())
    reporter.handle(MatrixProgressEvent("matrix_started", total=total))
    cells = [_cell(f"case-{index}") for index in range(active)]
    for cell in cells:
        reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light", total=total))
    for index, cell in enumerate(cells[:finished], start=1):
        reporter.handle(
            MatrixProgressEvent("cell_finished", cell=cell, trace=_trace(), completion_index=index, total=total)
        )

    table = reporter._lanes_table()
    assert len(table.rows) == expected_rows

    # Render to verify the full placeholder count renders without a column-arity error.
    console = Console(width=width, force_terminal=False, record=True)
    console.print(table)


@pytest.mark.parametrize(
    ("phase", "expected_activity"),
    [
        pytest.param("running_tool", "run · execute_home_code", id="running-tool"),
        pytest.param("processing_tool_result", "result · execute_home_code", id="processing-tool-result"),
    ],
)
def test_narrow_activity_renders_full_safe_tool_labels(phase: LanePhase, expected_activity: str) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=80, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light"))
    reporter.handle_phase(LanePhaseEvent(cell, phase, "execute_home_code"))

    table = reporter._lanes_table()
    console = Console(width=80, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()

    column_headers = [str(column.header) for column in table.columns]
    assert "Activity" in column_headers
    assert "Variant" not in column_headers
    assert expected_activity in output
    assert "…" not in output


def test_operational_issues_group_rate_limits_with_full_actionable_detail() -> None:
    detail_start = (
        "Provider payload: Daily token quota exceeded for the Cerebras deployment. "
        "The request was rejected before model execution,"
    )
    detail_end = (
        "and the account must wait for the quota window to reset before retrying this evaluation cell. "
        "No partial response was produced by the provider."
    )
    detail = f"{detail_start} {detail_end}"
    detail_fragments = ("Provider payload: Daily token quota exceeded for the Cerebras deployment.",)
    execution_error = ExecutionError(
        exception_type="ModelHTTPError",
        message="Provider rate limit exceeded while starting the model request",
        status_code=429,
        provider_code="token_quota_exceeded",
        provider_model="cerebras-llama-3.3",
        provider_detail=detail,
    )
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (
            MatrixCellRef("case-z", "canonical", "zeta", "cerebras", "home_minimal"),
            _trace(
                case_id="case-z",
                candidate_id="zeta",
                state="incomplete",
                score_reason=None,
                failure="rate_limit",
                model_id="cerebras",
                request="quota case",
                execution_error=execution_error,
            ),
        ),
        (
            MatrixCellRef("case-a", "canonical", "alpha", "cerebras", "home_minimal"),
            _trace(
                case_id="case-a",
                candidate_id="alpha",
                state="incomplete",
                score_reason=None,
                failure="rate_limit",
                model_id="cerebras",
                request="quota case",
                execution_error=execution_error,
            ),
        ),
    )

    groups = reporter.state.operational_issue_groups
    assert len(groups) == 1
    assert groups[0].count == 2
    assert groups[0].cells == ("alpha/case-a", "zeta/case-z")
    assert groups[0].detail == detail
    assert _operational_issue_detail(groups[0]) == detail

    frame = reporter._render()
    issues_panel = next(renderable for renderable in frame.renderables if isinstance(renderable, Panel))
    assert issues_panel.expand is True
    live_console = Console(width=240, force_terminal=False, record=True)
    live_console.print(issues_panel)
    _assert_actionable_operational_issue_output(live_console.export_text(), detail_fragments=detail_fragments)

    durable_console = Console(width=240, force_terminal=False, record=True)
    durable_console.print(
        render_durable_final(reporter.state, run_dir="runs/run-1", report_html="runs/run-1/report.html")
    )
    _assert_actionable_operational_issue_output(durable_console.export_text(), detail_fragments=detail_fragments)


def test_operational_issues_empty_state_remains_visible_live_and_durable() -> None:
    reporter = _reporter(human=True)
    live_console = Console(width=160, force_terminal=False, record=True)
    live_console.print(reporter._render())
    live_output = _normalized_rich_text(live_console.export_text())

    durable_console = Console(width=160, force_terminal=False, record=True)
    durable_console.print(
        render_durable_final(reporter.state, run_dir="runs/run-1", report_html="runs/run-1/report.html")
    )
    durable_output = _normalized_rich_text(durable_console.export_text())

    assert "Operational issues" in live_output
    assert "None" in live_output
    assert "Operational issues" in durable_output
    assert "None" in durable_output


def test_operational_issues_bound_raw_traceback_detail_at_terminal_width() -> None:
    traceback_detail = "\n".join(
        (
            "Traceback (most recent call last):",
            *(f"  File 'eval.py', line {index}, in run stack-frame-{index}" for index in range(120)),
            "TimeoutError: provider did not respond",
        )
    )
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (
            _cell("timeout-case"),
            _trace(
                case_id="timeout-case",
                state="incomplete",
                score_reason=None,
                failure="timeout",
                provider_error=traceback_detail,
                execution_error=ExecutionError(
                    exception_type="TimeoutError",
                    message="Provider timed out after=75s",
                    status_code=None,
                    provider_code=None,
                    provider_model=None,
                    provider_detail=None,
                ),
            ),
        ),
    )

    frame = reporter._render()
    issues_panel = next(renderable for renderable in frame.renderables if isinstance(renderable, Panel))
    live_console = Console(width=136, force_terminal=False, record=True)
    live_console.print(issues_panel)
    live_output = live_console.export_text()

    durable_console = Console(width=136, force_terminal=False, record=True)
    durable_console.print(
        render_durable_final(reporter.state, run_dir="runs/run-1", report_html="runs/run-1/report.html")
    )
    durable_output = durable_console.export_text()
    detail = _operational_issue_detail(reporter.state.operational_issue_groups[0])

    assert detail == "Provider timed out after=75s [full: errors.log]"

    for output in (live_output, durable_output):
        assert any("Detail" in line for line in output.splitlines())
        normalized = _normalized_rich_text(output)
        assert "Provider timed" in normalized
        assert "after=75s" in normalized
        assert "[full: errors.log]" in normalized
        assert "stack-frame-119" not in output


def test_operational_issues_preview_cells_when_group_has_many_occurrences() -> None:
    execution_error = ExecutionError(
        exception_type="ModelHTTPError",
        message="Provider rate limit exceeded",
        status_code=429,
        provider_code="rate_limit_exceeded",
        provider_model="provider/model",
        provider_detail=None,
    )
    reporter = _reporter(human=True)
    _feed(
        reporter,
        *tuple(
            (
                MatrixCellRef(f"case-{index}", "canonical", "baseline", "provider", "home_minimal"),
                _trace(
                    case_id=f"case-{index}",
                    state="incomplete",
                    score_reason=None,
                    failure="rate_limit",
                    model_id="provider",
                    execution_error=execution_error,
                ),
            )
            for index in range(5)
        ),
    )

    group = reporter.state.operational_issue_groups[0]

    assert group.count == 5
    assert _operational_issue_cells(group.cells).plain == (
        "baseline/case-0\nbaseline/case-1\nbaseline/case-2\n+ 2 more"
    )


def test_operational_issues_render_provider_markup_as_literal_text_live_and_durable() -> None:
    detail = "Provider payload [bold]literal-detail[/bold] keeps [unterminated as literal text."
    provider_code = "[x]code[/x]"
    provider_model = "[/red]"
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (
            MatrixCellRef("literal-markup", "canonical", "baseline", "cerebras", "home_minimal"),
            _trace(
                case_id="literal-markup",
                state="incomplete",
                score_reason=None,
                failure="rate_limit",
                model_id="cerebras",
                request="literal markup",
                execution_error=ExecutionError(
                    exception_type="ModelHTTPError",
                    message="Provider rejected literal markup payload",
                    status_code=429,
                    provider_code=provider_code,
                    provider_model=provider_model,
                    provider_detail=detail,
                ),
            ),
        ),
    )

    live_console = Console(width=320, force_terminal=False, record=True)
    live_console.print(reporter._render())
    live_output = _normalized_rich_text(live_console.export_text())

    durable_console = Console(width=320, force_terminal=False, record=True)
    durable_console.print(
        render_durable_final(reporter.state, run_dir="runs/run-1", report_html="runs/run-1/report.html")
    )
    durable_output = _normalized_rich_text(durable_console.export_text())

    assert detail in live_output
    assert "".join(provider_code.split()) in "".join(live_output.split())
    assert provider_model in live_output
    assert detail in durable_output
    assert "".join(provider_code.split()) in "".join(durable_output.split())
    assert provider_model in durable_output




def test_machine_phase_events_produce_no_terminal_output() -> None:
    stream = StringIO()
    reporter = _reporter(human=False)
    reporter._console = Console(width=200, force_terminal=False, file=stream)
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    reporter.handle_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))

    assert stream.getvalue() == ""


def test_machine_events_emit_stable_kv_without_raw_payload() -> None:
    output_buffer = StringIO()
    stream = Console(width=200, force_terminal=False, file=output_buffer)
    reporter = _reporter(human=False)
    reporter._console = stream
    cell = _cell()
    trace = _trace(
        state="incomplete",
        score_reason=None,
        failure="rate_limit",
        request="secret request body",
        provider_error="Traceback (most recent call last): raw-traceback-sensitive",
        execution_error=ExecutionError(
            exception_type="ModelHTTPError",
            message="provider-sensitive-message",
            status_code=429,
            provider_code="token_quota_exceeded",
            provider_model="cerebras-llama-3.3",
            provider_detail="provider-sensitive-body",
        ),
    )

    reporter.handle(MatrixProgressEvent("matrix_started", total=1))
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="secret request body"))
    reporter.handle_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))
    reporter.handle(MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=1, total=1))

    output = output_buffer.getvalue()
    assert output.splitlines() == [
        "matrix_started total=1",
        "cell_finished index=1 result=incomplete·rate_limit",
    ]
    # Redirected output stays deterministic KV with no diagnostic or model-facing content.
    assert "provider-sensitive-body" not in output
    assert "provider-sensitive-message" not in output
    assert "raw-traceback-sensitive" not in output
    assert "secret request body" not in output
    assert "secret-tool-arg" not in output
    assert "payload" not in output


def test_durable_final_redacts_raw_advisory_code_judge_details() -> None:
    raw_judge_reason = (
        "RAW_JUDGE_REASON request=RAW_JUDGE_REQUEST_PAYLOAD code=RAW_JUDGE_CODE_PAYLOAD tool=RAW_JUDGE_TOOL_PAYLOAD"
    )
    report_model = _report_model(
        _report_cell(
            "alpha-passed",
            candidate_id="alpha",
            model_id="alpha-model",
            judge=JudgePresentation("available", score=0.9, passed=True, reason="clear"),
        ),
        _report_cell(
            "alpha-failed-score",
            candidate_id="alpha",
            model_id="alpha-model",
            judge=JudgePresentation("available", score=0.5, passed=False, reason=raw_judge_reason),
        ),
        _report_cell(
            "judge-failure",
            candidate_id="alpha",
            model_id="alpha-model",
            request="RAW_REQUEST_PAYLOAD",
            provider_error="RAW_NATIVE_PROVIDER_TRACE",
            judge=JudgePresentation(
                "failed",
                failure=JudgeFailure("JudgeProviderTimeout", "RAW_PROVIDER_FAILURE_REQUEST_PAYLOAD"),
            ),
        ),
        _report_cell(
            "judge-unavailable",
            candidate_id="alpha",
            model_id="alpha-model",
            judge=JudgePresentation("unavailable"),
        ),
        _report_cell(
            "beta-low-score",
            candidate_id="beta",
            model_id="beta-model",
            reasoning_effort="high",
            judge=JudgePresentation("available", score=0.1, passed=False, reason=raw_judge_reason),
        ),
        _report_cell(
            "beta-mid-score",
            candidate_id="beta",
            model_id="beta-model",
            reasoning_effort="high",
            judge=JudgePresentation("available", score=0.2, passed=False, reason=raw_judge_reason),
        ),
        _report_cell(
            "gamma-low-score",
            candidate_id="gamma",
            model_id="gamma-model",
            request_variant_id="paraphrase",
            judge=JudgePresentation("available", score=0.3, passed=False, reason=raw_judge_reason),
        ),
        _report_cell(
            "gamma-mid-score",
            candidate_id="gamma",
            model_id="gamma-model",
            request_variant_id="paraphrase",
            judge=JudgePresentation("available", score=0.4, passed=False, reason=raw_judge_reason),
        ),
    )

    console = Console(width=80, force_terminal=False, record=True)
    console.print(
        render_durable_final(
            _report_state(report_model),
            run_dir="runs/run-1",
            report_html="runs/run-1/report.html",
            report_model=report_model,
        )
    )
    output = console.export_text()

    summary = report_model.judge_summary
    assert (summary.requested, summary.available, summary.passed) == (8, 6, 1)
    assert (summary.evaluator_failed, summary.unavailable) == (1, 1)
    assert summary.pass_rate == pytest.approx(1 / 6)
    assert summary.mean_score == pytest.approx(0.4)
    assert raw_judge_reason not in output
    for sentinel in (
        "RAW_JUDGE_REASON",
        "RAW_JUDGE_REQUEST_PAYLOAD",
        "RAW_JUDGE_CODE_PAYLOAD",
        "RAW_JUDGE_TOOL_PAYLOAD",
    ):
        assert sentinel not in output
    assert "RAW_PROVIDER_FAILURE_REQUEST_PAYLOAD" not in output
    assert "RAW_NATIVE_PROVIDER_TRACE" not in output
    assert "RAW_REQUEST_PAYLOAD" not in output
    assert "secret-tool-arg" not in output
    assert "payload" not in output


def test_recent_variant_keeps_meaningful_suffix_not_head() -> None:
    reporter = _reporter(human=True)
    long_model = "openai-chat:oc/deepseek-v4-flash-free"
    _feed(reporter, (_cell(), _trace(model_id=long_model, reasoning_effort="xhigh")))

    console = Console(width=160, force_terminal=False, record=True)
    console.print(reporter._recent_table())
    output = console.export_text()

    # The variant column is left-truncated, so the meaningful reasoning suffix survives.
    assert "(xhigh)" in output
    # The uninformative provider prefix is dropped rather than kept with a trailing ellipsis.
    assert "openai-chat" not in output


def test_render_includes_overall_progress_bar() -> None:
    reporter = _reporter(human=True)
    reporter._state.total = 2
    _feed_completed = MatrixProgressEvent("cell_finished", cell=_cell(), trace=_trace(), completion_index=1, total=2)
    reporter.handle(MatrixProgressEvent("cell_started", cell=_cell(), request="Turn on light", total=2))
    reporter.handle(_feed_completed)

    console = Console(width=160, force_terminal=False, record=True)
    console.print(reporter._render())
    output = console.export_text()

    # The restored overall progress bar reports completion against the planned matrix total.
    assert "Overall" in output
    assert "1/2" in output


def test_lanes_table_live_elapsed_displays_whole_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active-lane elapsed displays whole seconds, not tenths, matching the 4 Hz Live cadence."""
    current_time = 100.0
    monkeypatch.setattr("llm_sandbox_evals.terminal.perf_counter", lambda: current_time)
    monkeypatch.setattr("llm_sandbox_evals.presentation.perf_counter", lambda: current_time)
    reporter = _reporter(human=True)
    reporter._console = Console(width=160, force_terminal=False, file=StringIO())
    reporter.handle(MatrixProgressEvent("matrix_started", total=1))
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light", total=1))

    current_time = 102.4
    table = reporter._lanes_table()
    console = Console(width=160, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()

    assert "2s / 75s" in output
    assert "2.4s" not in output


