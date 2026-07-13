from typing import Literal

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.presentation import (
    LanePhaseEvent,
    PresentationState,
    ReportPresentationModel,
    effective_cause,
    rate,
    result_counts,
    result_label,
)
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EvalDiagnostics,
    RequiredAction,
)
from pydantic_evals.reporting import EvaluationReport, ReportCase
import pytest


def _trace(
    *,
    state: Literal["correct", "incorrect", "incomplete"] = "correct",
    case_id: str = "case",
    candidate_id: str = "baseline",
    model_id: str = "stub",
    reasoning_effort: str | None = None,
    cap_exhausted: bool = False,
    failure: str | None = None,
    action_reason: str | None = "ok",
) -> CaseTrace:
    expected = (RequiredAction("light", "turn_on", ("light.bedroom",)),)
    return CaseTrace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        answer=None,
        required_actions=expected,
        outcome=CaseOutcome(state, action_reason),
        action_result=ActionResult(state == "correct", action_reason or "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(cap_exhausted=cap_exhausted, failure=failure, elapsed_seconds=1.0),
        reasoning_effort=reasoning_effort,
    )


def _cell(case_id: str, candidate_id: str = "baseline", model_id: str = "stub") -> MatrixCellRef:
    return MatrixCellRef(case_id, candidate_id, model_id, "home_minimal")


@pytest.mark.parametrize(
    ("trace_kwargs", "expected"),
    [
        pytest.param({"state": "incorrect", "cap_exhausted": True, "action_reason": "wrong_target"}, "cap_exhausted", id="cap-exhausted"),
        pytest.param({"state": "incomplete", "action_reason": None, "failure": "timeout"}, "timeout", id="incomplete-timeout"),
        pytest.param({"state": "incomplete", "action_reason": None, "failure": None}, "unknown", id="incomplete-no-failure"),
        pytest.param({"state": "incorrect", "action_reason": "wrong_target"}, "wrong_target", id="scored-reason"),
        pytest.param({"state": "correct", "action_reason": "ok"}, "ok", id="correct"),
    ],
)
def test_effective_cause_resolves_every_branch(trace_kwargs: dict[str, object], expected: str) -> None:
    trace = _trace(**trace_kwargs)  # type: ignore[arg-type]
    assert effective_cause(trace) == expected


def test_result_label_combines_state_and_cause_without_raw_payload() -> None:
    trace = _trace(state="incomplete", action_reason=None, failure="provider_error")
    label = result_label(trace)

    assert label == "incomplete·provider_error"
    assert effective_cause(trace) in label
    assert trace.outcome.state in label


def test_rate_is_zero_for_empty_denominator() -> None:
    assert rate(3, 0) == 0.0
    assert rate(3, 6) == 0.5


def test_result_counts_scored_vocabulary_excludes_completed() -> None:
    counts = result_counts([
        _trace(state="correct", case_id="a"),
        _trace(state="incorrect", case_id="b"),
        _trace(state="incomplete", case_id="c"),
    ])

    assert (counts.total, counts.correct, counts.incorrect, counts.incomplete) == (3, 1, 1, 1)
    assert counts.scored == 2
    assert counts.quality_rate == 0.5
    assert counts.coverage_rate == pytest.approx(2 / 3)


def test_presentation_state_projects_lifecycle_events() -> None:
    state = PresentationState()
    timeout_cell = _cell("timeout-case")
    correct_cell = _cell("correct-case")
    timeout_trace = _trace(state="incomplete", action_reason=None, failure="timeout")
    correct_trace = _trace(state="correct")

    state.ingest(MatrixProgressEvent("matrix_started", total=2), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("cell_started", cell=timeout_cell, request="r1"), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("tool_started", cell=timeout_cell, tool_name="execute_home_code"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=timeout_cell, trace=timeout_trace, completion_index=1, total=2),
        timeout=10.0,
        max_tool_calls=10,
    )
    state.ingest(MatrixProgressEvent("cell_started", cell=correct_cell, request="r2"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=correct_cell, trace=correct_trace, completion_index=2, total=2),
        timeout=10.0,
        max_tool_calls=10,
    )

    assert state.counts.scored == 1
    assert state.counts.incomplete == 1
    assert state.counts.quality_rate == 1.0
    assert state.counts.coverage_rate == 0.5
    # Operational issues group by the real cause, never action_mismatch.
    assert dict(state.operational_issues) == {"timeout": 1}
    assert not state.lanes
    # Lane phase events are accepted but render nothing (streaming deferred).
    state.ingest_phase(LanePhaseEvent(correct_cell, "thinking"))


def test_presentation_state_cap_exhausted_does_not_count_as_operational_issue() -> None:
    state = PresentationState()
    cell = _cell("cap-case")
    trace = _trace(state="incorrect", cap_exhausted=True, action_reason="wrong_target")

    state.ingest(MatrixProgressEvent("matrix_started", total=1), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("cell_started", cell=cell, request="r"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=1, total=1),
        timeout=10.0,
        max_tool_calls=10,
    )

    # Cap exhaustion is scored (not an operational issue) yet resolves to its own cause.
    assert state.counts.scored == 1
    assert state.counts.incomplete == 0
    assert dict(state.operational_issues) == {}


def test_presentation_state_uses_planned_total_for_coverage_while_run_active() -> None:
    # Fix #2: coverage reflects the planned matrix denominator while a run is in progress,
    # not the number of cells that have finished so far.
    state = PresentationState()
    cell = _cell("active-case")
    correct_trace = _trace(state="correct")

    state.ingest(MatrixProgressEvent("matrix_started", total=4), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("cell_started", cell=cell, request="r"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=cell, trace=correct_trace, completion_index=1, total=4),
        timeout=10.0,
        max_tool_calls=10,
    )

    counts = state.counts
    # The planned total (4) is the coverage denominator, not the one finished cell.
    assert counts.total == 4
    assert counts.scored == 1
    assert counts.coverage_rate == 0.25
    # Quality still uses the scored denominator (correct/scored), unaffected by planned total.
    assert counts.quality_rate == 1.0


def _report_case(trace: CaseTrace, cell: MatrixCellRef, *, metrics: dict[str, float | int] | None = None) -> ReportCase:
    return ReportCase(
        name=f"{cell.candidate_id}/{cell.model_id}/{cell.case_id}",
        inputs=cell,
        metadata={
            "run_id": "report-projection",
            "case_id": cell.case_id,
            "candidate_id": cell.candidate_id,
            "model_id": cell.model_id,
            "home": cell.home,
            "reasoning_effort": cell.reasoning_effort,
            "temperature": cell.temperature,
            "variant_label": "stub(default)",
        },
        expected_output=None,
        output=trace,
        metrics=metrics or {},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=None,
        total_duration=None,
    )


def test_report_presentation_model_shares_semantics_with_runtime_state() -> None:
    timeout_cell = _cell("timeout-case", model_id="luna")
    correct_cell = _cell("correct-case", model_id="luna")
    timeout_trace = _trace(state="incomplete", action_reason=None, failure="timeout", model_id="luna")
    correct_trace = _trace(state="correct", model_id="luna")
    report = EvaluationReport(
        name="report-projection",
        cases=[_report_case(timeout_trace, timeout_cell), _report_case(correct_trace, correct_cell)],
        experiment_metadata={"models": [{"model_id": "luna", "variant_label": "luna(default)"}]},
    )

    model = ReportPresentationModel.from_report(report)

    assert model.counts == result_counts([timeout_trace, correct_trace])
    assert dict(model.operational_issues) == {"timeout": 1}
    # The saved-report projection never mutates and carries the descriptor verbatim.
    assert model.descriptor["models"][0]["variant_label"] == "luna(default)"
    # result_label/effective_cause are identical across both projections for the same trace.
    state = PresentationState()
    state.ingest(MatrixProgressEvent("matrix_started", total=2), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("cell_started", cell=timeout_cell, request="r"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=timeout_cell, trace=timeout_trace, completion_index=1, total=2),
        timeout=10.0,
        max_tool_calls=10,
    )
    assert result_label(timeout_trace) == result_label(state.completed[0].trace)
    assert effective_cause(timeout_trace) == effective_cause(state.completed[0].trace)


def test_report_presentation_model_reads_metrics_with_usage_fallback() -> None:
    cell = _cell("metric-case")
    trace = _trace(state="correct")
    trace_with_usage = CaseTrace(
        case_id="metric-case",
        candidate_id="baseline",
        model_id="stub",
        answer=None,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        outcome=CaseOutcome("correct", "ok"),
        action_result=ActionResult(True, "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(elapsed_seconds=1.0, usage={"total_tokens": 42, "cost": 0.01}),
    )
    report = EvaluationReport(
        name="metrics-projection",
        cases=[
            _report_case(trace, cell, metrics={"tool_calls": 3, "total_tokens": 30}),
            _report_case(trace_with_usage, cell, metrics={}),
        ],
    )

    model = ReportPresentationModel.from_report(report)
    aggregate = model.aggregates[0]

    # Metrics take precedence over trace diagnostics; the trace usage fallback applies when metrics are absent.
    # First cell: tool_calls=3 from metrics. Second cell: metrics empty -> trace diagnostics tool_calls=0.
    assert aggregate.mean_calls == 1.5
    # total_tokens: first cell has none in metrics and no trace usage; second cell falls back to trace usage (42).
    assert aggregate.total_tokens == 72.0
    assert aggregate.total_cost == 0.01
