from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import (
    LanePhaseEvent,
    MatrixCellRef,
    _record_trace_metrics,
    build_dataset,
    matrix_summary_lines,
    run_matrix,
)
from llm_sandbox_evals.presentation import ReportPresentationModel
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalCase,
    EvalDiagnostics,
    PromptCandidate,
    RequestVariant,
    RequiredAction,
)
from pydantic_evals.reporting.analyses import ScalarResult, TableResult
import pytest

from llm_sandbox_evals import reports


async def test_run_matrix_stub_persists_v7_action_trace_and_variant_identity(tmp_path: Path) -> None:
    config = EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=["direct_turn_off_utility_room_accent"],
        homes=None,
        runs_dir=tmp_path,
        reasoning_effort="low",
    )
    report = await run_matrix(config, run_id="stub-v7")
    reloaded = reports.load_report(reports.write_report_json(report, config, run_id="stub-v7-written"))
    trace = reloaded.cases[0].output

    assert trace.outcome.state == "correct"
    assert trace.outcome.score == 1.0
    assert trace.answer == "Done."
    assert trace.scoring_version == 9
    assert trace.reasoning_effort == "low"
    assert _scalar(reloaded.analyses, "Quality rate").value == 1.0
    # The run descriptor rides on native experiment_metadata and survives reload.
    models = reloaded.experiment_metadata["models"]
    assert models[0]["variant_label"] == "stub(low)"


async def test_run_matrix_emits_plain_text_lifecycle_response(tmp_path: Path) -> None:
    events = []
    report = await run_matrix(
        _config(tmp_path, cases=["direct_turn_off_utility_room_accent"]),
        run_id="lifecycle-v7",
        on_event=events.append,
    )

    assert [event.state for event in events] == [
        "matrix_started",
        "cell_started",
        "tool_started",
        "tool_finished",
        "response_received",
        "cell_finished",
    ]
    assert events[-2].response == "Done."
    assert events[-1].trace == report.cases[0].output


async def test_run_matrix_forwards_payload_free_phases_for_the_active_cell(tmp_path: Path) -> None:
    phases: list[LanePhaseEvent] = []
    case_id = "direct_turn_off_utility_room_accent"
    report = await run_matrix(
        _config(tmp_path, cases=[case_id]),
        run_id="phase-forwarding-v7",
        on_phase=phases.append,
    )

    expected_cell = MatrixCellRef(case_id, "canonical", "baseline", "stub", "home_full")

    assert report.cases[0].output.outcome.state == "correct"
    assert [event.cell for event in phases] == [expected_cell] * len(phases)
    assert [(event.phase, event.tool_name) for event in phases] == [
        ("queued", None),
        ("awaiting_model", None),
        ("running_tool", "execute_home_code"),
        ("processing_tool_result", "execute_home_code"),
        ("responding", None),
        ("responding", None),
        ("scoring", None),
        ("finished", None),
    ]
    assert tuple(field.name for field in fields(phases[0])) == ("cell", "phase", "tool_name")


async def test_report_uses_scored_vocabulary_and_excludes_completed(tmp_path: Path) -> None:
    config = _config(tmp_path, models=["model-a", "model-b"])
    candidates = [_candidate("baseline", api_prompt="long authored prompt"), _candidate("compact", api_prompt="short")]
    selected_cases = [_case("case-a"), _case("case-b")]
    dataset = build_dataset(config, candidates, selected_cases, "aggregation-v7")
    states = {
        ("baseline", "model-a", "case-a"): "correct",
        ("baseline", "model-a", "case-b"): "incorrect",
        ("baseline", "model-b", "case-a"): "incomplete",
        ("baseline", "model-b", "case-b"): "incomplete",
        ("compact", "model-a", "case-a"): "correct",
        ("compact", "model-a", "case-b"): "incorrect",
        ("compact", "model-b", "case-a"): "correct",
        ("compact", "model-b", "case-b"): "incorrect",
    }

    async def task(cell: MatrixCellRef) -> CaseTrace:
        return _trace(cell, states[(cell.candidate_id, cell.model_id, cell.case_id)])

    report = await dataset.evaluate(task, name="aggregation-v7", progress=False, retry_task=None)
    ranking = _table(report.analyses, "Candidate ranking")
    pairs = _table(report.analyses, "Candidate x model outcomes")

    # Quality rate = correct / scored; completed is gone from the vocabulary.
    assert _scalar(report.analyses, "Quality rate").value == pytest.approx(0.5)
    assert _scalar(report.analyses, "Scored cells").value == 6
    assert _scalar(report.analyses, "Coverage rate").value == pytest.approx(0.75)
    with pytest.raises(StopIteration):
        next(a for a in report.analyses if isinstance(a, ScalarResult) and "Completed" in a.title)
    # Ranking columns: Candidate, Correct, Incorrect, Incomplete, Scored, Quality rate, Coverage rate, ...
    assert "Completed" not in ranking.columns
    assert "Scored" in ranking.columns
    compact_ranking = next(row for row in ranking.rows if row[0] == "compact")
    assert compact_ranking[0] == "compact"
    baseline_ranking = next(row for row in ranking.rows if row[0] == "baseline")
    # baseline: correct=1, incorrect=1, incomplete=2, scored=2, quality=0.5, coverage=0.5
    assert baseline_ranking[1:7] == [1, 1, 2, 2, 0.5, 0.5]
    baseline_model_b = next(row for row in pairs.rows if row[0] == "baseline" and row[1] == "model-b(default)")
    # baseline/model-b: correct=0, incorrect=0, incomplete=2, scored=0, quality=None, coverage=0.0
    assert baseline_model_b[2:8] == [0, 0, 2, 0, None, 0.0]


async def test_matrix_summary_lines_emit_scored_vocabulary(tmp_path: Path) -> None:
    config = _config(tmp_path, models=["model-a"])
    candidates = [_candidate("baseline")]
    selected_cases = [_case("case-a"), _case("case-b")]
    dataset = build_dataset(config, candidates, selected_cases, "summary-v7")

    async def task(cell: MatrixCellRef) -> CaseTrace:
        return _trace(cell, "correct")

    report = await dataset.evaluate(task, name="summary-v7", progress=False, retry_task=None)
    lines = matrix_summary_lines(report)

    assert lines[0].startswith("quality_rate: ")
    assert lines[1].startswith("coverage_rate: ")
    assert lines[2].startswith("scored: ")
    assert not any("completed=" in line for line in lines)
    per_pair = [line for line in lines if line.startswith("baseline/")]
    assert per_pair
    assert "quality_rate=" in per_pair[0]
    assert "coverage_rate=" in per_pair[0]
    assert "scored=" in per_pair[0]


async def test_report_case_metrics_carry_tool_calls_for_stub_and_no_tokens(tmp_path: Path) -> None:
    config = _config(tmp_path, cases=["direct_turn_off_utility_room_accent"])
    report = await run_matrix(config, run_id="metrics-v7")

    metrics = report.cases[0].metrics
    # The stub emits tool activity but no provider usage, so tokens stay unavailable.
    assert metrics["tool_calls"] == 1
    assert metrics.get("total_tokens") is None
    assert metrics.get("cost") is None


async def test_native_metrics_omit_cost_and_presentation_uses_trace_cost_fallback(tmp_path: Path) -> None:
    config = _config(tmp_path, models=["model-a"])
    dataset = build_dataset(config, [_candidate("baseline")], [_case("case-a")], "metric-cost-fallback")

    async def task(cell: MatrixCellRef) -> CaseTrace:
        trace = replace(
            _trace(cell, "correct"),
            diagnostics=EvalDiagnostics(
                tool_calls=4,
                successful_tool_calls=3,
                failed_tool_calls=1,
                model_turns=2,
                elapsed_seconds=1.5,
                usage={"total_tokens": 21, "cost": 0.03},
            ),
        )
        _record_trace_metrics(trace)
        return trace

    report = await dataset.evaluate(task, name="metric-cost-fallback", progress=False, retry_task=None)
    metrics = report.cases[0].metrics

    # Native task metrics retain operational counts, elapsed time, and token usage.
    assert metrics["tool_calls"] == 4
    assert metrics["successful_tool_calls"] == 3
    assert metrics["failed_tool_calls"] == 1
    assert metrics["model_turns"] == 2
    assert metrics["elapsed_seconds"] == 1.5
    assert metrics["total_tokens"] == 21.0
    # Cost stays in the self-contained provider usage trace rather than a custom eval metric.
    assert "cost" not in metrics
    assert ReportPresentationModel.from_report(report).aggregates[0].total_cost == 0.03


def _config(runs_dir: Path, *, models: list[str] | None = None, cases: list[str] | None = None) -> EvalConfig:
    return EvalConfig(
        models=models or ["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=cases,
        homes=None,
        runs_dir=runs_dir,
        concurrency=1,
    )


def _candidate(candidate_id: str, *, api_prompt: str = "prompt") -> PromptCandidate:
    return PromptCandidate(candidate_id, api_prompt, "execute", "history", "statistics", "logbook", "automation")


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id,
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Turn on bedroom light"),),
        (RequiredAction("light", "turn_on", ("light.bedroom",)),),
    )


def _trace(cell: MatrixCellRef, state: str) -> CaseTrace:
    expected = (RequiredAction("light", "turn_on", ("light.bedroom",)),)
    action_reason = "ok" if state == "correct" else "action_mismatch"
    return CaseTrace(
        case_id=cell.case_id,
        candidate_id=cell.candidate_id,
        model_id=cell.model_id,
        request_variant_id=cell.request_variant_id,
        request_text="Turn on bedroom light",
        category="test",
        answer="Done.",
        required_actions=expected,
        desired_entities=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome(
            state,
            "actions" if state != "incomplete" else None,
            action_reason if state != "incomplete" else None,
        ),
        action_result=ActionResult(state == "correct", action_reason),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(elapsed_seconds=0.5),
    )


def _table(analyses: Sequence[object], title: str) -> TableResult:
    return next(analysis for analysis in analyses if isinstance(analysis, TableResult) and analysis.title == title)


def _scalar(analyses: Sequence[object], title: str) -> ScalarResult:
    return next(analysis for analysis in analyses if isinstance(analysis, ScalarResult) and analysis.title == title)
