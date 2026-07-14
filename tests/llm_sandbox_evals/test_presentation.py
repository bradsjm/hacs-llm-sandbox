from typing import Literal, cast

from llm_sandbox_evals.experiment import MatrixCellRef, MatrixProgressEvent
from llm_sandbox_evals.phases import LanePhase
from llm_sandbox_evals.presentation import (
    LanePhaseEvent,
    OperationalIssueGroup,
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
    EndStateResult,
    EvalDiagnostics,
    ExecutionError,
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
    score_reason: str | None = "ok",
    scoring_mode: str | None = "actions",
    provider_error: str | None = None,
    execution_error: ExecutionError | None = None,
) -> CaseTrace:
    expected = (RequiredAction("light", "turn_on", ("light.bedroom",)),)
    return CaseTrace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        answer=None,
        required_actions=expected,
        desired_states=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome(state, scoring_mode, score_reason),
        action_result=ActionResult(state == "correct", score_reason or "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(cap_exhausted=cap_exhausted, failure=failure, elapsed_seconds=1.0),
        reasoning_effort=reasoning_effort,
        provider_error=provider_error,
        execution_error=execution_error,
    )


def _cell(
    case_id: str,
    candidate_id: str = "baseline",
    model_id: str = "stub",
    reasoning_effort: str | None = None,
) -> MatrixCellRef:
    return MatrixCellRef(case_id, candidate_id, model_id, "home_minimal", reasoning_effort)


@pytest.mark.parametrize(
    ("trace_kwargs", "expected"),
    [
        pytest.param(
            {"state": "incorrect", "cap_exhausted": True, "score_reason": "wrong_target"},
            "cap_exhausted",
            id="cap-exhausted",
        ),
        pytest.param(
            {"state": "incomplete", "score_reason": None, "failure": "timeout"}, "timeout", id="incomplete-timeout"
        ),
        pytest.param(
            {"state": "incomplete", "score_reason": None, "failure": None}, "unknown", id="incomplete-no-failure"
        ),
        pytest.param({"state": "incorrect", "score_reason": "wrong_target"}, "wrong_target", id="scored-reason"),
        pytest.param(
            {"state": "correct", "score_reason": "equivalent_target_partition"},
            "equivalent_target_partition",
            id="equivalent-target-partition",
        ),
        pytest.param({"state": "correct", "score_reason": "ok"}, "ok", id="correct"),
    ],
)
def test_effective_cause_resolves_every_branch(trace_kwargs: dict[str, object], expected: str) -> None:
    trace = _trace(**trace_kwargs)
    assert effective_cause(trace) == expected


def test_result_label_combines_state_and_cause_without_raw_payload() -> None:
    trace = _trace(state="incomplete", score_reason=None, failure="provider_error")
    label = result_label(trace)

    assert label == "incomplete·provider_error"
    assert effective_cause(trace) in label
    assert trace.outcome.state in label


def test_result_label_preserves_equivalent_target_partition_reason() -> None:
    trace = _trace(state="correct", score_reason="equivalent_target_partition")

    assert result_label(trace) == "correct·equivalent_target_partition"


def test_rate_is_zero_for_empty_denominator() -> None:
    assert rate(3, 0) == 0.0
    assert rate(3, 6) == 0.5


def test_result_counts_scored_vocabulary_excludes_completed() -> None:
    counts = result_counts(
        [
            _trace(state="correct", case_id="a"),
            _trace(state="incorrect", case_id="b"),
            _trace(state="incomplete", case_id="c"),
        ]
    )

    assert (counts.total, counts.correct, counts.incorrect, counts.incomplete) == (3, 1, 1, 1)
    assert counts.scored == 2
    assert counts.quality_rate == 0.5
    assert counts.coverage_rate == pytest.approx(2 / 3)


def test_presentation_state_projects_lifecycle_events() -> None:
    state = PresentationState()
    timeout_cell = _cell("timeout-case")
    correct_cell = _cell("correct-case")
    timeout_trace = _trace(state="incomplete", score_reason=None, failure="timeout")
    correct_trace = _trace(state="correct")

    state.ingest(MatrixProgressEvent("matrix_started", total=2), timeout=10.0, max_tool_calls=10)
    state.ingest(MatrixProgressEvent("cell_started", cell=timeout_cell, request="r1"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("tool_started", cell=timeout_cell, tool_name="execute_home_code"),
        timeout=10.0,
        max_tool_calls=10,
    )
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
    # A phase from a completed lane does not activate the live Activity column.
    state.ingest_phase(LanePhaseEvent(correct_cell, "thinking"))


def test_presentation_state_cap_exhausted_does_not_count_as_operational_issue() -> None:
    state = PresentationState()
    cell = _cell("cap-case")
    trace = _trace(
        state="incorrect", cap_exhausted=True, scoring_mode="cap_exhausted", score_reason="cap_exhausted"
    )

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


def test_presentation_state_ignores_thinking_for_unknown_and_finished_lanes() -> None:
    state = PresentationState()
    finished_cell = _cell("finished-case")

    state.ingest(MatrixProgressEvent("matrix_started", total=1), timeout=10.0, max_tool_calls=10)
    assert not state.ingest_phase(LanePhaseEvent(_cell("unknown-case"), "thinking"))
    state.ingest(MatrixProgressEvent("cell_started", cell=finished_cell, request="r"), timeout=10.0, max_tool_calls=10)
    state.ingest(
        MatrixProgressEvent("cell_finished", cell=finished_cell, trace=_trace(), completion_index=1, total=1),
        timeout=10.0,
        max_tool_calls=10,
    )
    assert not state.ingest_phase(LanePhaseEvent(finished_cell, "thinking"))

    assert not state.activity_enabled
    assert not state.lanes


def test_ingest_phase_reports_only_visible_active_lane_changes() -> None:
    state = PresentationState()
    cell = _cell("phase-change-case")

    state.ingest(MatrixProgressEvent("cell_started", cell=cell, request="r"), timeout=10.0, max_tool_calls=10)

    assert state.ingest_phase(LanePhaseEvent(cell, "queued"))
    assert not state.ingest_phase(LanePhaseEvent(cell, "queued"))
    assert state.ingest_phase(LanePhaseEvent(cell, "thinking"))
    assert state.activity_enabled
    assert not state.ingest_phase(LanePhaseEvent(cell, "thinking"))
    assert state.ingest_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))
    assert not state.ingest_phase(LanePhaseEvent(cell, "running_tool", "execute_home_code"))
    assert state.ingest_phase(LanePhaseEvent(cell, "running_tool", "get_history"))


def test_ingest_phase_coalesces_provider_tool_names_and_rejects_invalid_phases() -> None:
    state = PresentationState()
    cell = _cell("preparing-case")

    state.ingest(MatrixProgressEvent("cell_started", cell=cell, request="r"), timeout=10.0, max_tool_calls=10)

    assert state.ingest_phase(LanePhaseEvent(cell, "preparing_tool_call", "provider-name-one"))
    assert not state.ingest_phase(LanePhaseEvent(cell, "preparing_tool_call", "provider-name-two"))
    assert state.lanes[cell].tool_name is None
    assert not state.ingest_phase(LanePhaseEvent(cell, cast(LanePhase, "invalid_phase")))


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param("queued", id="queued"),
        pytest.param("awaiting_model", id="awaiting-model"),
        pytest.param("preparing_tool_call", id="preparing-tool-call"),
        pytest.param("running_tool", id="running-tool"),
        pytest.param("processing_tool_result", id="processing-tool-result"),
        pytest.param("responding", id="responding"),
        pytest.param("scoring", id="scoring"),
        pytest.param("finished", id="finished"),
    ],
)
def test_pre_reveal_phases_update_active_lane_without_enabling_activity(phase: LanePhase) -> None:
    state = PresentationState()
    cell = _cell("phase-case")

    state.ingest(MatrixProgressEvent("cell_started", cell=cell, request="r"), timeout=10.0, max_tool_calls=10)
    state.ingest_phase(LanePhaseEvent(cell, phase))

    assert state.lanes[cell].phase == phase
    assert not state.activity_enabled


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


def _report_case(
    trace: CaseTrace, cell: MatrixCellRef, *, metrics: dict[str, float | int] | None = None
) -> ReportCase:
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


def test_operational_issue_groups_preserve_exact_identity_across_runtime_and_report() -> None:
    quota_payload = '{"code":"token_quota","resource":"tokens"}'
    quota_error = ExecutionError(
        "ModelHTTPError",
        "Too many requests",
        status_code=429,
        provider_code="token_quota",
        provider_model="cerebras",
        provider_detail=quota_payload,
    )
    distinct_quota_error = ExecutionError(
        "ModelHTTPError",
        "Too many requests",
        status_code=429,
        provider_code="request_quota",
        provider_model="cerebras",
        provider_detail='{"code":"request_quota","resource":"requests"}',
    )
    distinct_message_error = ExecutionError(
        "ModelHTTPError",
        "Token quota exceeded",
        status_code=429,
        provider_code="token_quota",
        provider_model="cerebras",
        provider_detail=quota_payload,
    )
    distinct_provider_model_error = ExecutionError(
        "ModelHTTPError",
        "Too many requests",
        status_code=429,
        provider_code="token_quota",
        provider_model="cerebras-fallback",
        provider_detail=quota_payload,
    )
    legacy_detail = 'legacy response: {"status":429,"code":"token_quota"}'
    entries = (
        (
            _cell("legacy", "legacy-candidate", "cerebras"),
            _trace(
                state="incomplete",
                case_id="legacy",
                candidate_id="legacy-candidate",
                model_id="cerebras",
                score_reason=None,
                failure="provider_error",
                provider_error=legacy_detail,
            ),
        ),
        (
            _cell("distinct-payload", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="distinct-payload",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=distinct_quota_error,
            ),
        ),
        (
            _cell("case-a", "zeta", "cerebras"),
            _trace(
                state="incomplete",
                case_id="case-a",
                candidate_id="zeta",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=quota_error,
            ),
        ),
        (_cell("success", model_id="cerebras"), _trace(case_id="success", model_id="cerebras")),
        (
            _cell("timeout", "timeout-candidate", "cerebras"),
            _trace(
                state="incomplete",
                case_id="timeout",
                candidate_id="timeout-candidate",
                model_id="cerebras",
                score_reason=None,
                failure="timeout",
                execution_error=ExecutionError("TimeoutError", "Timed out after 10 seconds"),
            ),
        ),
        (
            _cell("variant", "baseline", "cerebras", "high"),
            _trace(
                state="incomplete",
                case_id="variant",
                model_id="cerebras",
                reasoning_effort="high",
                score_reason=None,
                failure="rate_limit",
                execution_error=quota_error,
            ),
        ),
        (
            _cell("cap-exhausted", model_id="cerebras"),
            _trace(
                state="incomplete",
                case_id="cap-exhausted",
                model_id="cerebras",
                cap_exhausted=True,
                score_reason=None,
                failure="cap_exhausted",
            ),
        ),
        (
            _cell("incorrect", model_id="cerebras"),
            _trace(state="incorrect", case_id="incorrect", model_id="cerebras", score_reason="wrong_target"),
        ),
        (
            _cell("distinct-message", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="distinct-message",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=distinct_message_error,
            ),
        ),
        (
            _cell("distinct-provider-model", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="distinct-provider-model",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=distinct_provider_model_error,
            ),
        ),
        (
            _cell("case-z", "alpha", "cerebras"),
            _trace(
                state="incomplete",
                case_id="case-z",
                candidate_id="alpha",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=quota_error,
            ),
        ),
    )
    expected_groups = (
        OperationalIssueGroup(
            2,
            "rate_limit",
            "cerebras(default)",
            ("alpha/case-z", "zeta/case-a"),
            "ModelHTTPError",
            429,
            "token_quota",
            "cerebras",
            "Too many requests",
            quota_payload,
        ),
        OperationalIssueGroup(
            1,
            "provider_error",
            "cerebras(default)",
            ("legacy-candidate/legacy",),
            "unknown",
            None,
            None,
            None,
            None,
            legacy_detail,
        ),
        OperationalIssueGroup(
            1,
            "rate_limit",
            "cerebras(default)",
            ("baseline/distinct-payload",),
            "ModelHTTPError",
            429,
            "request_quota",
            "cerebras",
            "Too many requests",
            '{"code":"request_quota","resource":"requests"}',
        ),
        OperationalIssueGroup(
            1,
            "rate_limit",
            "cerebras(default)",
            ("baseline/distinct-message",),
            "ModelHTTPError",
            429,
            "token_quota",
            "cerebras",
            "Token quota exceeded",
            quota_payload,
        ),
        OperationalIssueGroup(
            1,
            "rate_limit",
            "cerebras(default)",
            ("baseline/distinct-provider-model",),
            "ModelHTTPError",
            429,
            "token_quota",
            "cerebras-fallback",
            "Too many requests",
            quota_payload,
        ),
        OperationalIssueGroup(
            1,
            "rate_limit",
            "cerebras(high)",
            ("baseline/variant",),
            "ModelHTTPError",
            429,
            "token_quota",
            "cerebras",
            "Too many requests",
            quota_payload,
        ),
        OperationalIssueGroup(
            1,
            "timeout",
            "cerebras(default)",
            ("timeout-candidate/timeout",),
            "TimeoutError",
            None,
            None,
            None,
            "Timed out after 10 seconds",
            "Timed out after 10 seconds",
        ),
    )

    state = PresentationState()
    state.ingest(MatrixProgressEvent("matrix_started", total=len(entries)), timeout=10.0, max_tool_calls=10)
    for completion_index, (cell, trace) in enumerate(entries, start=1):
        state.ingest(
            MatrixProgressEvent("cell_started", cell=cell, request=cell.case_id), timeout=10.0, max_tool_calls=10
        )
        state.ingest(
            MatrixProgressEvent(
                "cell_finished",
                cell=cell,
                trace=trace,
                completion_index=completion_index,
                total=len(entries),
            ),
            timeout=10.0,
            max_tool_calls=10,
        )
    report = EvaluationReport(
        name="operational-issue-groups",
        cases=[_report_case(trace, cell) for cell, trace in entries],
    )
    report_model = ReportPresentationModel.from_report(report)

    assert len(state.operational_issue_groups) == 7
    assert state.operational_issue_groups == expected_groups
    assert report_model.operational_issue_groups == state.operational_issue_groups
    assert dict(state.operational_issues) == {"rate_limit": 6, "provider_error": 1, "timeout": 1}
    assert report_model.operational_issues == state.operational_issues


def test_operational_issue_groups_include_full_raw_provider_error_identity() -> None:
    structured_detail = "Structured provider detail"
    execution_error = ExecutionError(
        "ModelHTTPError",
        "Too many requests",
        status_code=429,
        provider_code="token_quota",
        provider_model="cerebras",
        provider_detail=structured_detail,
    )
    raw_alpha = (
        "ProviderRequestError: RAW_WRAPPER_ALPHA\n"
        "The above exception was the direct cause of the following exception:\n"
        "RateLimitError: TOKEN_QUOTA_CAUSE"
    )
    raw_beta = (
        "ProviderRequestError: RAW_WRAPPER_BETA\n"
        "The above exception was the direct cause of the following exception:\n"
        "RateLimitError: TOKEN_QUOTA_CAUSE"
    )
    shared_raw = "ProviderRequestError: SHARED_WRAPPER\nCaused by: SHARED_TOKEN_QUOTA_CAUSE"
    legacy_alpha = "LegacyProviderError: LEGACY_WRAPPER_ALPHA\nCaused by: LEGACY_CAUSE"
    legacy_beta = "LegacyProviderError: LEGACY_WRAPPER_BETA\nCaused by: LEGACY_CAUSE"
    entries = (
        (
            _cell("raw-beta", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="raw-beta",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                provider_error=raw_beta,
                execution_error=execution_error,
            ),
        ),
        (
            _cell("legacy-alpha", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="legacy-alpha",
                model_id="cerebras",
                score_reason=None,
                failure="provider_error",
                provider_error=legacy_alpha,
            ),
        ),
        (
            _cell("case-a", "zeta", "cerebras"),
            _trace(
                state="incomplete",
                case_id="case-a",
                candidate_id="zeta",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                provider_error=shared_raw,
                execution_error=execution_error,
            ),
        ),
        (
            _cell("fallback", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="fallback",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                execution_error=execution_error,
            ),
        ),
        (
            _cell("legacy-beta", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="legacy-beta",
                model_id="cerebras",
                score_reason=None,
                failure="provider_error",
                provider_error=legacy_beta,
            ),
        ),
        (
            _cell("raw-alpha", "baseline", "cerebras"),
            _trace(
                state="incomplete",
                case_id="raw-alpha",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                provider_error=raw_alpha,
                execution_error=execution_error,
            ),
        ),
        (
            _cell("case-z", "alpha", "cerebras"),
            _trace(
                state="incomplete",
                case_id="case-z",
                candidate_id="alpha",
                model_id="cerebras",
                score_reason=None,
                failure="rate_limit",
                provider_error=shared_raw,
                execution_error=execution_error,
            ),
        ),
    )
    expected_groups = (
        OperationalIssueGroup(
            count=2,
            cause="rate_limit",
            variant="cerebras(default)",
            cells=("alpha/case-z", "zeta/case-a"),
            exception_type="ModelHTTPError",
            status_code=429,
            provider_code="token_quota",
            provider_model="cerebras",
            message="Too many requests",
            detail=shared_raw,
        ),
        OperationalIssueGroup(
            count=1,
            cause="provider_error",
            variant="cerebras(default)",
            cells=("baseline/legacy-alpha",),
            exception_type="unknown",
            status_code=None,
            provider_code=None,
            provider_model=None,
            message=None,
            detail=legacy_alpha,
        ),
        OperationalIssueGroup(
            count=1,
            cause="provider_error",
            variant="cerebras(default)",
            cells=("baseline/legacy-beta",),
            exception_type="unknown",
            status_code=None,
            provider_code=None,
            provider_model=None,
            message=None,
            detail=legacy_beta,
        ),
        OperationalIssueGroup(
            count=1,
            cause="rate_limit",
            variant="cerebras(default)",
            cells=("baseline/raw-alpha",),
            exception_type="ModelHTTPError",
            status_code=429,
            provider_code="token_quota",
            provider_model="cerebras",
            message="Too many requests",
            detail=raw_alpha,
        ),
        OperationalIssueGroup(
            count=1,
            cause="rate_limit",
            variant="cerebras(default)",
            cells=("baseline/raw-beta",),
            exception_type="ModelHTTPError",
            status_code=429,
            provider_code="token_quota",
            provider_model="cerebras",
            message="Too many requests",
            detail=raw_beta,
        ),
        OperationalIssueGroup(
            count=1,
            cause="rate_limit",
            variant="cerebras(default)",
            cells=("baseline/fallback",),
            exception_type="ModelHTTPError",
            status_code=429,
            provider_code="token_quota",
            provider_model="cerebras",
            message="Too many requests",
            detail=structured_detail,
        ),
    )

    state = PresentationState()
    state.ingest(MatrixProgressEvent("matrix_started", total=len(entries)), timeout=10.0, max_tool_calls=10)
    for completion_index, (cell, trace) in enumerate(entries, start=1):
        state.ingest(
            MatrixProgressEvent("cell_finished", cell=cell, trace=trace, completion_index=completion_index),
            timeout=10.0,
            max_tool_calls=10,
        )
    report_model = ReportPresentationModel.from_report(
        EvaluationReport(
            name="raw-operational-issue-groups",
            cases=[_report_case(trace, cell) for cell, trace in entries],
        )
    )

    assert state.operational_issue_groups == expected_groups
    assert report_model.operational_issue_groups == expected_groups
    assert dict(state.operational_issues) == {"rate_limit": 5, "provider_error": 2}
    assert report_model.operational_issues == state.operational_issues


def test_report_presentation_model_shares_semantics_with_runtime_state() -> None:
    timeout_cell = _cell("timeout-case", model_id="luna")
    correct_cell = _cell("correct-case", model_id="luna")
    timeout_trace = _trace(state="incomplete", score_reason=None, failure="timeout", model_id="luna")
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
        desired_states=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome("correct", "actions", "ok"),
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
