from collections.abc import Sequence
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellRef, build_dataset, run_matrix
from llm_sandbox_evals.schema import (
    ActionAnswer,
    ActionLedger,
    BlockedOutcome,
    CaseOutcome,
    CaseTrace,
    EvalCase,
    EvalDiagnostics,
    Expected,
    PromptCandidate,
)
from pydantic_evals.reporting.analyses import ScalarResult, TableResult
import pytest

from llm_sandbox_evals import reports


async def test_run_matrix_stub_persists_v4_trace_and_binary_rate(tmp_path: Path) -> None:
    report = await run_matrix(_config(tmp_path, cases=["state_living_temperature"]), run_id="stub-v4")
    run_dir = reports.write_report_json(
        report, _config(tmp_path, cases=["state_living_temperature"]), run_id="stub-v4"
    )
    reloaded = reports.load_report(run_dir)
    trace = reloaded.cases[0].output

    assert trace.outcome.state in {"correct", "incorrect"}
    assert trace.outcome.score in {0.0, 1.0}
    assert trace.answer is not None
    assert isinstance(trace.diagnostics.tool_calls, int)
    assert _scalar(reloaded.analyses, "Overall correct rate").value in {0.0, 1.0}


async def test_run_matrix_emits_v4_lifecycle_events(tmp_path: Path) -> None:
    events = []
    report = await run_matrix(
        _config(tmp_path, cases=["state_living_temperature"]), run_id="lifecycle-v4", on_event=events.append
    )

    assert [event.state for event in events] == [
        "matrix_started",
        "cell_started",
        "tool_started",
        "tool_finished",
        "response_received",
        "cell_finished",
    ]
    assert events[-1].trace == report.cases[0].output


async def test_report_excludes_incomplete_from_correct_rate_and_keeps_coverage(tmp_path: Path) -> None:
    config = _config(tmp_path, models=["model-a", "model-b"])
    candidates = [_candidate("baseline", api_prompt="long authored prompt"), _candidate("compact", api_prompt="short")]
    selected_cases = [_case("case-a", "state"), _case("case-b", "state")]
    dataset = build_dataset(config, candidates, selected_cases, "aggregation-v4")
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

    report = await dataset.evaluate(task, name="aggregation-v4", progress=False, retry_task=None)
    ranking = _table(report.analyses, "Candidate ranking")
    pairs = _table(report.analyses, "Candidate x model outcomes")

    assert _scalar(report.analyses, "Overall correct rate").value == pytest.approx(0.5)
    assert _scalar(report.analyses, "Completed cells").value == 6
    assert ranking.rows[0][0] == "compact"
    assert ranking.rows[0][5] == pytest.approx(0.5)
    baseline_ranking = next(row for row in ranking.rows if row[0] == "baseline")
    assert baseline_ranking[1:7] == [1, 1, 2, 2, 0.5, 0.5]
    baseline_model_a = next(row for row in pairs.rows if row[:2] == ["baseline", "model-a"])
    assert baseline_model_a[2:8] == [1, 1, 0, 2, 0.5, 1.0]
    baseline_model_b = next(row for row in pairs.rows if row[:2] == ["baseline", "model-b"])
    assert baseline_model_b[2:8] == [0, 0, 2, 0, None, 0.0]
    assert baseline_model_b[14] is None


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


def _case(case_id: str, category: str) -> EvalCase:
    return EvalCase(
        case_id,
        category,
        "home_default",
        "exercise native aggregation",
        False,
        Expected(blocked_outcome=BlockedOutcome()),
    )


def _trace(cell: MatrixCellRef, state: str) -> CaseTrace:
    return CaseTrace(
        case_id=cell.case_id,
        category=cell.category,
        candidate_id=cell.candidate_id,
        model_id=cell.model_id,
        answer=ActionAnswer(answer=""),
        expected=Expected(blocked_outcome=BlockedOutcome()),
        outcome=CaseOutcome(state, state),
        conclusions=(),
        actions=(),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(),
        scoring_version=4,
    )


def _table(analyses: Sequence[object], title: str) -> TableResult:
    return next(analysis for analysis in analyses if isinstance(analysis, TableResult) and analysis.title == title)


def _scalar(analyses: Sequence[object], title: str) -> ScalarResult:
    return next(analysis for analysis in analyses if isinstance(analysis, ScalarResult) and analysis.title == title)
