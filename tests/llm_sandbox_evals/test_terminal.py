from io import StringIO
from pathlib import Path
from typing import cast
import unicodedata

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import LanePhaseEvent, MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.phases import LanePhase
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EvalDiagnostics,
    ExecutionError,
    RequiredAction,
    ToolEvent,
)
from llm_sandbox_evals.terminal import (
    MatrixTerminalReporter,
    _duration,
    _left_ellipsis,
    _token_total,
    render_durable_final,
)
import pytest
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


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
    case_id: str = "case",
    candidate_id: str = "baseline",
    state: str = "correct",
    action_reason: str | None = "ok",
    failure: str | None = None,
    cap_exhausted: bool = False,
    tool_calls: int = 0,
    elapsed: float | None = None,
    request: str = "Turn on bedroom light",
    answer: str | None = None,
    model_id: str = "stub",
    reasoning_effort: str | None = None,
    provider_error: str | None = None,
    execution_error: ExecutionError | None = None,
) -> CaseTrace:
    return CaseTrace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
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
        provider_error=provider_error,
        execution_error=execution_error,
        user_request=request,
    )


def _feed(reporter: MatrixTerminalReporter, *cells: tuple[MatrixCellRef, CaseTrace]) -> None:
    reporter._state.total = len(cells)
    for index, (cell, trace) in enumerate(cells, start=1):
        reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request=trace.user_request, total=len(cells)))
        reporter.handle(
            MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=index, total=len(cells))
        )


def _lane_column_metadata(table: Table) -> list[tuple[str, int | None, int | None]]:
    return [(str(column.header), column.width, column.ratio) for column in table.columns]


def _normalized_rich_text(output: str) -> str:
    without_borders = "".join(
        " " if unicodedata.name(character, "").startswith("BOX DRAWINGS") else character for character in output
    )
    return " ".join(without_borders.split())


def _assert_actionable_operational_issue_output(output: str, *, detail_fragments: tuple[str, ...]) -> None:
    normalized = _normalized_rich_text(output)
    assert "Operational issues" in normalized
    assert "#" in normalized
    assert "Cause" in normalized
    assert "Variant" in normalized
    assert "Cells" in normalized
    assert "Exception" in normalized
    assert "HTTP / provider code" in normalized
    assert "Detail" in normalized
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


class _LiveRecorder:
    """Capture phase-triggered Live redraws without starting Rich's background loop."""

    def __init__(self) -> None:
        self.update_count = 0

    def update(self, _renderable: object, *, refresh: bool) -> None:
        self.update_count += 1


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


def test_dynamic_live_frame_preserves_spinner_between_refreshes() -> None:
    reporter = _reporter(human=True)
    reporter._state.total = 1
    reporter.handle(MatrixProgressEvent("cell_started", cell=_cell(), request="Turn on light"))

    reporter._lanes_table()
    first_spinner = reporter._spinners[_cell()]
    reporter._lanes_table()
    second_spinner = reporter._spinners[_cell()]

    assert first_spinner is second_spinner


@pytest.mark.parametrize("width", [pytest.param(80, id="narrow"), pytest.param(102, id="wide")])
def test_lanes_keep_five_columns_without_activity_for_non_thinking_phases(width: int) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=width, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    for phase in (
        "awaiting_model",
        "preparing_tool_call",
        "running_tool",
        "processing_tool_result",
        "responding",
        "scoring",
    ):
        reporter.handle_phase(LanePhaseEvent(cell, phase, "execute_home_code"))

    table = reporter._lanes_table()
    console = Console(width=width, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()

    assert _lane_column_metadata(table) == [
        ("", 2, None),
        ("Request", None, 3),
        ("Variant", 22, None),
        ("Elapsed / timeout", 18, None),
        ("Tools / cap", 12, None),
    ]
    assert "Activity" not in output
    assert "Waiting" not in output


@pytest.mark.parametrize(
    ("phase", "expected_activity"),
    [
        pytest.param("running_tool", "run · execute_home_code", id="running-tool"),
        pytest.param("processing_tool_result", "result · execute_home_code", id="processing-tool-result"),
    ],
)
def test_thinking_enables_safe_tool_activity_without_stream_payloads(phase: LanePhase, expected_activity: str) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=102, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light"))
    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))
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
    ("width", "expected_columns"),
    [
        pytest.param(
            80,
            [
                ("", 2, None),
                ("Request", None, 3),
                ("Activity", 26, None),
                ("Elapsed / timeout", 18, None),
                ("Tools / cap", 12, None),
            ],
            id="narrow",
        ),
        pytest.param(
            102,
            [
                ("", 2, None),
                ("Request", None, 3),
                ("Activity", 26, None),
                ("Variant", 22, None),
                ("Elapsed / timeout", 18, None),
                ("Tools / cap", 12, None),
            ],
            id="wide",
        ),
    ],
)
def test_activity_idle_placeholder_renders_at_narrow_and_wide_widths(
    width: int, expected_columns: list[tuple[str, int | None, int | None]]
) -> None:
    reporter = _reporter(human=True)
    reporter._console = Console(width=width, force_terminal=False, file=StringIO())
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))
    reporter.handle(MatrixProgressEvent("cell_finished", cell=cell, trace=_trace(), completion_index=1, total=1))

    table = reporter._lanes_table()
    console = Console(width=width, force_terminal=False, record=True)
    console.print(table)

    assert _lane_column_metadata(table) == expected_columns
    assert "Activity" in console.export_text()


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
    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))
    reporter.handle_phase(LanePhaseEvent(cell, phase, "execute_home_code"))

    table = reporter._lanes_table()
    console = Console(width=80, force_terminal=False, record=True)
    console.print(table)
    output = console.export_text()

    assert _lane_column_metadata(table) == [
        ("", 2, None),
        ("Request", None, 3),
        ("Activity", 26, None),
        ("Elapsed / timeout", 18, None),
        ("Tools / cap", 12, None),
    ]
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
    detail_fragments = (
        "Provider payload: Daily token quota exceeded for the Cerebras deployment.",
        "The request was rejected before model",
        "execution,",
        "the account must wait for the quota window to reset before retrying this evaluation cell.",
        "response was produced by the provider.",
    )
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
            MatrixCellRef("case-z", "zeta", "cerebras", "home_minimal"),
            _trace(
                case_id="case-z",
                candidate_id="zeta",
                state="incomplete",
                action_reason=None,
                failure="rate_limit",
                model_id="cerebras",
                request="quota case",
                execution_error=execution_error,
            ),
        ),
        (
            MatrixCellRef("case-a", "alpha", "cerebras", "home_minimal"),
            _trace(
                case_id="case-a",
                candidate_id="alpha",
                state="incomplete",
                action_reason=None,
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


def test_operational_issues_render_provider_markup_as_literal_text_live_and_durable() -> None:
    detail = "Provider payload [bold]literal-detail[/bold] keeps [unterminated as literal text."
    provider_code = "[bold]literal-code[/bold]"
    provider_model = "[/red]"
    reporter = _reporter(human=True)
    _feed(
        reporter,
        (
            MatrixCellRef("literal-markup", "baseline", "cerebras", "home_minimal"),
            _trace(
                case_id="literal-markup",
                state="incomplete",
                action_reason=None,
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
    assert provider_code in live_output
    assert provider_model in live_output
    assert detail in durable_output
    assert provider_code in durable_output
    assert provider_model in durable_output


def test_reporter_constructs_live_with_visible_vertical_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor_kwargs: dict[str, object] = {}
    lifecycle: list[str] = []
    live_renderables: list[object] = []
    current_time = 100.0

    class _FakeLive:
        def __init__(self, renderable: object, **kwargs: object) -> None:
            live_renderables.append(renderable)
            constructor_kwargs.update(kwargs)

        def start(self) -> None:
            lifecycle.append("start")

        def stop(self) -> None:
            lifecycle.append("stop")

    monkeypatch.setattr("llm_sandbox_evals.terminal.Live", _FakeLive)
    monkeypatch.setattr("llm_sandbox_evals.terminal.perf_counter", lambda: current_time)
    reporter = _reporter(human=True)
    reporter._state.started_at = 100.0
    reporter._console = Console(width=160, height=24, force_terminal=False, file=StringIO())

    with reporter:
        assert constructor_kwargs["vertical_overflow"] == "visible"
        assert constructor_kwargs["refresh_per_second"] == 24
        assert lifecycle == ["start"]

    assert lifecycle == ["start", "stop"]
    assert live_renderables

    first_frame = Console(width=160, force_terminal=False, record=True)
    first_frame.print(live_renderables[0])
    current_time = 102.4
    second_frame = Console(width=160, force_terminal=False, record=True)
    second_frame.print(live_renderables[0])

    assert "Elapsed 0.0s" in first_frame.export_text()
    assert "Elapsed 2.4s" in second_frame.export_text()


def test_machine_phase_events_produce_no_terminal_output() -> None:
    stream = StringIO()
    reporter = _reporter(human=False)
    reporter._console = Console(width=200, force_terminal=False, file=stream)
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="Turn on light"))
    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))

    assert stream.getvalue() == ""


def test_live_coalesces_duplicate_phase_redraws() -> None:
    reporter = _reporter(human=True)
    recorder = _LiveRecorder()
    reporter._live = cast(Live, recorder)
    cell = _cell()
    reporter.handle(MatrixProgressEvent("cell_started", cell=cell, request="light"))
    recorder.update_count = 0

    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))
    assert recorder.update_count == 1
    reporter.handle_phase(LanePhaseEvent(cell, "thinking"))
    assert recorder.update_count == 1
    reporter.handle_phase(LanePhaseEvent(cell, "responding"))
    assert recorder.update_count == 2
    reporter.handle_phase(LanePhaseEvent(cell, "responding"))
    assert recorder.update_count == 2
    reporter.handle_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))
    assert recorder.update_count == 3
    reporter.handle_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))
    assert recorder.update_count == 3
    reporter.handle_phase(LanePhaseEvent(cell, "running_tool", "get_history"))

    assert recorder.update_count == 4


def test_machine_events_emit_stable_kv_without_raw_payload() -> None:
    output_buffer = StringIO()
    stream = Console(width=200, force_terminal=False, file=output_buffer)
    reporter = _reporter(human=False)
    reporter._console = stream
    cell = _cell()
    trace = _trace(
        state="incomplete",
        action_reason=None,
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


def test_durable_final_emits_counts_and_artifact_path_once() -> None:
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


def test_left_ellipsis_returns_value_when_it_fits() -> None:
    assert _left_ellipsis("stub(high)", 22) == "stub(high)"


def test_left_ellipsis_truncates_head_and_keeps_suffix() -> None:
    value = "openai-chat:oc/model(xhigh)"
    result = _left_ellipsis(value, 12)
    # Overflow drops the head with a leading ellipsis while preserving the exact suffix.
    assert len(result) == 12
    assert result.startswith("…")
    assert result[1:] == value[-11:]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param(None, "tokens unavailable", id="unavailable"),
        pytest.param(299_499, "tokens 299k", id="rounds-down"),
        pytest.param(299_500, "tokens 300k", id="rounds-up"),
    ],
)
def test_token_total_uses_rounded_thousands(tokens: float | None, expected: str) -> None:
    assert _token_total(tokens) == expected


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

    # The cancellation hint appears once, in the live frame footer, not duplicated in the header.
    orientation = Console(width=160, force_terminal=False, record=True)
    orientation.print(interactive._orientation())
    orientation_text = orientation.export_text()
    assert "Escape" not in orientation_text
    assert "Ctrl+C" not in orientation_text

    # The transient Live frame carries the correct mechanism for the active stream mode.
    frame = Console(width=160, force_terminal=False, record=True)
    frame.print(interactive._render())
    frame_text = frame.export_text()
    assert expected_hint in frame_text
    assert absent_hint not in frame_text

    # Redirected (non-TTY) runs emit deterministic KV only and never print a human cancel hint.
    stream = StringIO()
    redirected._console = Console(width=200, force_terminal=False, file=stream)
    redirected.handle(MatrixProgressEvent("matrix_started", total=1))
    assert expected_hint not in stream.getvalue()
