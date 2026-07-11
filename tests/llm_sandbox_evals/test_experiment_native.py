from collections.abc import Sequence
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from freezegun.api import FrozenDateTimeFactory
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellRef, build_dataset, run_matrix
from llm_sandbox_evals.schema import (
    CaseContext,
    CaseTrace,
    CheckResult,
    EvalCase,
    Expected,
    PromptCandidate,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.reporting.analyses import ScalarResult, TableResult
import pytest

from llm_sandbox_evals import agent_runner, reports


async def test_run_matrix_stub_slice_writes_reloadable_report_json(tmp_path: Path) -> None:
    config = _config(tmp_path, cases=["state_living_temperature"])

    report = await run_matrix(config, run_id="stub-slice")
    run_dir = reports.write_report_json(
        report,
        config,
        run_id="stub-slice",
    )
    reloaded = reports.load_report(run_dir)

    assert len(report.cases) == 1
    assert len(reloaded.cases) == 1
    report_case = reloaded.cases[0]

    assert isinstance(report_case.inputs, MatrixCellRef)
    assert report_case.inputs == MatrixCellRef(
        case_id="state_living_temperature",
        candidate_id="baseline",
        model_id="stub",
        home="home_minimal",
        category="state_read",
    )
    assert report_case.metadata == {
        "run_id": "stub-slice",
        "case_id": "state_living_temperature",
        "candidate_id": "baseline",
        "model_id": "stub",
        "home": "home_minimal",
        "category": "state_read",
    }
    assert isinstance(report_case.output, CaseTrace)
    assert report_case.output.score == pytest.approx(1.0)
    assert report_case.scores["score"].value == pytest.approx(1.0)
    assert _scalar(reloaded.analyses, "Overall mean score").value == pytest.approx(1.0)
    assert _scalar(reloaded.analyses, "Incomplete cells").value == 0
    # The stub read the temperature entity, so one tool event with its return is persisted.
    assert len(report_case.output.tool_events) == 1
    assert report_case.output.tool_events[0].tool_name == "execute_home_code"


@pytest.mark.parametrize(
    ("case_id", "expected_domain", "expected_service"),
    [
        pytest.param("multi_history_then_living_fan", "fan", "set_percentage", id="history-action"),
        pytest.param("multi_logbook_then_living_light_off", "light", "turn_off", id="logbook-action"),
    ],
)
async def test_stub_completes_dependent_recorder_actions_in_one_execute_call(
    tmp_path: Path,
    freezer: FrozenDateTimeFactory,
    case_id: str,
    expected_domain: str,
    expected_service: str,
) -> None:
    freezer.move_to("2026-06-29T12:00:00+00:00")
    report = await run_matrix(_config(tmp_path, cases=[case_id]), run_id=f"stub-{case_id}")
    trace = report.cases[0].output

    assert isinstance(trace, CaseTrace)
    assert trace.score == pytest.approx(1.0)
    assert trace.tool_call_count == 1
    assert [event.tool_name for event in trace.tool_events] == ["execute_home_code"]
    assert trace.recorded_actions[0]["domain"] == expected_domain
    assert trace.recorded_actions[0]["service"] == expected_service


async def test_sandbox_outcome_reports_score_required_gate_and_model_error_label(tmp_path: Path) -> None:
    config = _config(tmp_path, cases=None)
    dataset = build_dataset(config, [_candidate("candidate-a")], [_case("case-a", "state_read")], "test-run")

    async def task(cell: MatrixCellRef) -> CaseTrace:
        return _trace(
            cell,
            score=0.25,
            checks=(
                CheckResult("tool_used", True, True, ""),
                CheckResult("domain_goal", False, True, "missing required outcome"),
                CheckResult("model_error", False, True, "provider failed"),
            ),
        )

    report = await dataset.evaluate(task, name="sandbox-outcome", progress=False, retry_task=None)
    report_case = report.cases[0]

    assert report_case.scores["score"].value == pytest.approx(0.25)
    assert report_case.scores["score"].reason == "failed: domain_goal, model_error"
    assert report_case.assertions["required_gates_passed"].value is False
    assert report_case.assertions["required_gates_passed"].reason == "domain_goal, model_error"
    assert report_case.labels["model_error"].value == "true"


async def test_candidate_matrix_report_aggregates_candidate_model_and_category_means(tmp_path: Path) -> None:
    config = _config(tmp_path, models=["model-a", "model-b"], cases=None)
    candidates = [_candidate("baseline", api_prompt="baseline prompt"), _candidate("compact", api_prompt="short")]
    selected_cases = [_case("case-state", "state_read"), _case("case-recorder", "recorder_read")]
    dataset = build_dataset(config, candidates, selected_cases, "test-run")
    outcomes = {
        ("baseline", "model-a", "case-state"): (0.8, 1),
        ("baseline", "model-a", "case-recorder"): (0.6, 3),
        ("baseline", "model-b", "case-state"): (0.4, 1),
        ("baseline", "model-b", "case-recorder"): (0.2, 1),
        ("compact", "model-a", "case-state"): (0.9, 1),
        ("compact", "model-a", "case-recorder"): (0.7, 1),
        ("compact", "model-b", "case-state"): (0.8, 2),
        ("compact", "model-b", "case-recorder"): (0.8, 2),
    }

    async def task(cell: MatrixCellRef) -> CaseTrace:
        score, tool_calls = outcomes[(cell.candidate_id, cell.model_id, cell.case_id)]
        return _trace(cell, score=score, tool_call_count=tool_calls)

    report = await dataset.evaluate(task, name="candidate-matrix", progress=False, retry_task=None)
    ranking = _table(report.analyses, "Candidate ranking")
    pair_means = _table(report.analyses, "Candidate x model means")
    overall = _scalar(report.analyses, "Overall mean score")

    assert ranking.columns == [
        "Candidate",
        "Mean",
        "MinModel",
        "ToolCalls",
        "PromptChars",
        "SizeRatio",
        "state_read",
        "recorder_read",
    ]
    assert ranking.rows[0][:4] == ["compact", 0.8, 0.8, 1.5]
    assert ranking.rows[1][:4] == ["baseline", 0.5, 0.3, 1.5]
    assert pair_means.rows == [
        ["baseline", "model-a", 0.7, 2.0],
        ["baseline", "model-b", 0.3, 1.0],
        ["compact", "model-a", 0.8, 1.0],
        ["compact", "model-b", 0.8, 2.0],
    ]
    assert overall.value == pytest.approx(0.65)


async def test_run_matrix_keeps_scoring_other_models_when_one_model_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_make_model = agent_runner.make_model

    def make_model(model_id: str) -> Model:
        if model_id == "bad-model":
            return FunctionModel(_raiseing_model, model_name="bad-model")
        return original_make_model(model_id)

    monkeypatch.setattr(
        agent_runner,
        "make_model",
        make_model,
    )
    config = _config(tmp_path, models=["bad-model", "stub"], cases=["state_living_temperature"])

    report = await run_matrix(config)
    traces_by_model = {case.output.model_id: case.output for case in report.cases}

    assert set(traces_by_model) == {"bad-model", "stub"}
    assert traces_by_model["bad-model"].score == 0.0
    assert [check.name for check in traces_by_model["bad-model"].checks] == ["model_error"]
    assert traces_by_model["stub"].score == pytest.approx(1.0)


async def _raiseing_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    raise RuntimeError("provider rejected model")


def _config(
    runs_dir: Path,
    *,
    models: list[str] | None = None,
    cases: list[str] | None = None,
) -> EvalConfig:
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
    return PromptCandidate(
        id=candidate_id,
        api_prompt=api_prompt,
        execute_home_code_description="execute",
        get_history_description="history",
        get_statistics_description="statistics",
        get_logbook_description="logbook",
        get_automation_description="automation",
    )


def _case(case_id: str, category: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        category=category,
        home="home_default",
        user_request="exercise native experiment aggregation",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(),
    )


def _trace(
    cell: MatrixCellRef,
    *,
    score: float,
    checks: tuple[CheckResult, ...] = (CheckResult("tool_used", True, True, ""),),
    tool_call_count: int = 1,
) -> CaseTrace:
    return CaseTrace(
        case_id=cell.case_id,
        category=cell.category,
        candidate_id=cell.candidate_id,
        model_id=cell.model_id,
        score=score,
        output="done",
        tool_call_count=tool_call_count,
        recorded_actions=(),
        checks=checks,
        error=None,
    )


def _table(analyses: Sequence[object], title: str) -> TableResult:
    return next(analysis for analysis in analyses if isinstance(analysis, TableResult) and analysis.title == title)


def _scalar(analyses: Sequence[object], title: str) -> ScalarResult:
    return next(analysis for analysis in analyses if isinstance(analysis, ScalarResult) and analysis.title == title)
