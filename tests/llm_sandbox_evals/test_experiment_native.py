import asyncio
from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import (
    LanePhaseEvent,
    MatrixCellRef,
    MatrixProgressEvent,
    _record_trace_metrics,
    build_dataset,
    build_run_descriptor,
    matrix_summary_lines,
    run_matrix,
)
from llm_sandbox_evals.presentation import PresentationState, ReportPresentationModel
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    AnswerPredicate,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalCase,
    EvalDiagnostics,
    ExpectedToolCall,
    PromptCandidate,
    RequestVariant,
    RequiredAction,
)
from pydantic_evals.reporting.analyses import ScalarResult, TableResult
import pytest

from llm_sandbox_evals import experiment, reports


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


def test_case_trace_contract_remains_scoring_v9_without_judge_fields() -> None:
    assert {field.name for field in fields(CaseTrace)} == {
        "case_id",
        "candidate_id",
        "model_id",
        "request_variant_id",
        "request_text",
        "answer",
        "required_actions",
        "desired_entities",
        "overlay_state_seeds",
        "recorded_invocations",
        "end_state_result",
        "outcome",
        "action_result",
        "action_ledger",
        "tool_events",
        "diagnostics",
        "oracle",
        "expected_tool_calls",
        "expected_answer",
        "tool_call_result",
        "answer_result",
        "reasoning_effort",
        "temperature",
        "scoring_version",
        "provider_error",
        "execution_error",
        "category",
        "tags",
        "conversation_id",
    }
    assert (
        _trace(MatrixCellRef("case", "canonical", "baseline", "stub", "home_minimal"), "correct").scoring_version == 9
    )


async def test_run_descriptor_and_experiment_metadata_include_judge_configuration(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        cases=["direct_turn_off_utility_room_accent"],
        judge_model="openai-chat:gpt-5.4",
    )
    selected_case = _case("direct_turn_off_utility_room_accent")
    descriptor = build_run_descriptor(config, "judge-descriptor", [selected_case])
    report = await run_matrix(config, descriptor=descriptor)

    assert descriptor.judge_model == "openai-chat:gpt-5.4"
    assert descriptor.judge_rubric_id
    assert descriptor.judge_rubric_version == 2
    assert report.experiment_metadata["judge_model"] == descriptor.judge_model
    assert report.experiment_metadata["judge_rubric_id"] == descriptor.judge_rubric_id
    assert report.experiment_metadata["judge_rubric_version"] == descriptor.judge_rubric_version


def _judge_effect_case() -> EvalCase:
    return EvalCase(
        "judge-effect",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Turn on the bedroom light."),),
        (),
        judge_code=True,
    )


def _judge_tool_calls_case() -> EvalCase:
    return EvalCase(
        "judge-tool-calls",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Check the bedroom light history."),),
        (),
        oracle="tool_calls",
        expected_tool_calls=(ExpectedToolCall("get_history"),),
        judge_code=True,
    )


def _judge_answer_case() -> EvalCase:
    return EvalCase(
        "judge-answer",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Is the bedroom light on?"),),
        (),
        oracle="answer",
        expected_answer=AnswerPredicate("boolean", value=True),
        judge_code=True,
    )


@pytest.mark.parametrize(
    "case_factory",
    [
        pytest.param(_judge_effect_case, id="effect"),
        pytest.param(_judge_tool_calls_case, id="tool-calls"),
        pytest.param(_judge_answer_case, id="answer"),
    ],
)
def test_dataset_marks_judge_opt_in_per_cell_for_every_oracle_type(
    case_factory: Callable[[], EvalCase], tmp_path: Path
) -> None:
    case = case_factory()
    config = _config(tmp_path, judge_model="openai-chat:gpt-5.4")
    [dataset_case] = build_dataset(config, [_candidate("baseline")], [case], "judge-cell").cases

    assert dataset_case.metadata["judge_code"] is True
    assert dataset_case.metadata["judge_enabled"] is True
    assert [type(evaluator).__name__ for evaluator in dataset_case.evaluators] == [
        "SandboxOutcome",
        "CodeQualityJudge",
    ]


@pytest.mark.parametrize(
    ("judge_model", "judge_code"),
    [
        pytest.param(None, False, id="no-model"),
        pytest.param("openai-chat:gpt-5.4", False, id="not-opted-in"),
    ],
)
def test_dataset_disables_judging_without_both_opt_in_conditions(
    judge_model: str | None, judge_code: bool, tmp_path: Path
) -> None:
    case = EvalCase(
        "judge-disabled",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Turn on the bedroom light."),),
        (),
        judge_code=judge_code,
    )
    config = _config(tmp_path, judge_model=judge_model)
    [dataset_case] = build_dataset(config, [_candidate("baseline")], [case], "judge-disabled").cases

    assert dataset_case.metadata["judge_code"] is judge_code
    assert dataset_case.metadata["judge_enabled"] is False
    assert [type(evaluator).__name__ for evaluator in dataset_case.evaluators] == ["SandboxOutcome"]


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


async def test_unjudged_run_keeps_the_existing_phase_order_and_timing(tmp_path: Path) -> None:
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
    assert all(event.phase != "judging" for event in phases)
    assert tuple(field.name for field in fields(phases[0])) == ("cell", "phase", "tool_name")


async def test_judged_cell_stays_active_until_the_judge_finishes_then_emits_one_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _synthetic_case("judged-cell")
    _select_synthetic_cases(monkeypatch, [case])
    events = []
    phases: list[LanePhaseEvent] = []
    state = PresentationState()
    lane_states_during_judging: list[bool] = []

    async def judge(*_args: object, **_kwargs: object) -> _JudgeResult:
        return _JudgeResult()

    def on_event(event: MatrixProgressEvent) -> None:
        events.append(event)
        state.ingest(event, timeout=75.0, max_tool_calls=10)

    def on_phase(event: LanePhaseEvent) -> None:
        phases.append(event)
        state.ingest_phase(event)
        if event.phase == "judging":
            lane_states_during_judging.append(event.cell in state.lanes)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge)
    report = await run_matrix(
        _config(tmp_path, judge_model="openai-chat:gpt-5.4"),
        run_id="judged-lifecycle",
        on_event=on_event,
        on_phase=on_phase,
    )

    terminal_phases = [event.phase for event in phases if event.phase in {"scoring", "judging", "finished"}]
    completed = [event for event in events if event.state == "cell_finished"]

    assert [event.phase for event in phases].index("scoring") < [event.phase for event in phases].index("judging")
    assert terminal_phases == ["scoring", "judging", "finished"]
    assert lane_states_during_judging == [True]
    assert len(completed) == 1
    assert completed[0].trace == report.cases[0].output
    assert not state.lanes


async def test_judge_provider_failure_is_a_native_evaluator_failure_after_the_trace_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = _synthetic_case("judge-provider-failure")
    _select_synthetic_cases(monkeypatch, [case])
    events = []
    state = PresentationState()

    async def judge(*_args: object, **_kwargs: object) -> _JudgeResult:
        raise RuntimeError("judge provider unavailable")

    def on_event(event: MatrixProgressEvent) -> None:
        events.append(event)
        state.ingest(event, timeout=75.0, max_tool_calls=10)

    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge)
    report = await run_matrix(
        _config(tmp_path, judge_model="openai-chat:gpt-5.4"),
        run_id="judge-provider-failure",
        on_event=on_event,
    )

    completed = [event for event in events if event.state == "cell_finished"]

    assert len(report.cases) == 1
    assert report.cases[0].output.case_id == case.id
    assert report.cases[0].output.outcome.state == "correct"
    assert [(failure.name, failure.error_type) for failure in report.cases[0].evaluator_failures] == [
        ("code_quality_judge", "RuntimeError")
    ]
    assert len(completed) == 1
    assert completed[0].trace == report.cases[0].output
    assert not state.lanes


async def test_concurrent_judged_cells_complete_in_judge_return_order_with_unique_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _synthetic_case("first-judged", "Turn off the Utility Room accent light.")
    second = _synthetic_case("second-judged", "Turn on the Utility Room ceiling light.")
    _select_synthetic_cases(monkeypatch, [first, second])
    first_judge_started = asyncio.Event()
    release_first_judge = asyncio.Event()

    async def judge(request: dict[str, object], *_args: object, **_kwargs: object) -> _JudgeResult:
        if request["request_text"] == first.requests[0].text:
            first_judge_started.set()
            await release_first_judge.wait()
        else:
            await first_judge_started.wait()
            release_first_judge.set()
        return _JudgeResult()

    events = []
    monkeypatch.setattr("llm_sandbox_evals.code_judge.judge_input_output", judge)
    await run_matrix(
        _config(tmp_path, concurrency=2, judge_model="openai-chat:gpt-5.4"),
        run_id="reverse-judge-completion",
        on_event=events.append,
    )

    completed = [event for event in events if event.state == "cell_finished"]

    assert [(event.cell.case_id, event.completion_index) for event in completed if event.cell is not None] == [
        (second.id, 1),
        (first.id, 2),
    ]
    assert len({event.cell for event in completed}) == 2


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


def _config(
    runs_dir: Path,
    *,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    judge_model: str | None = None,
    concurrency: int = 1,
) -> EvalConfig:
    return EvalConfig(
        models=models or ["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=cases,
        homes=None,
        runs_dir=runs_dir,
        concurrency=concurrency,
        judge_model=judge_model,
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


def _synthetic_case(case_id: str, request: str = "Turn off the Utility Room accent light.") -> EvalCase:
    action = (
        RequiredAction("light", "turn_on", ("light.utility_room_ceiling",))
        if request == "Turn on the Utility Room ceiling light."
        else RequiredAction("light", "turn_off", ("light.utility_room_accent",))
    )
    return EvalCase(
        case_id,
        "home_full",
        "test",
        (RequestVariant("canonical", request),),
        (action,),
        judge_code=True,
    )


def _select_synthetic_cases(monkeypatch: pytest.MonkeyPatch, cases: list[EvalCase]) -> None:
    def select_cases(_case_ids: list[str] | None, _homes: list[str] | None) -> list[EvalCase]:
        return cases

    monkeypatch.setattr(experiment, "_select_cases", select_cases)


class _JudgeResult:
    def __init__(self) -> None:
        self.reason = "clear"
        self.pass_ = True
        self.score = 1.0


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
