from os import close
from pathlib import Path
from tempfile import mkstemp
from typing import Self

from llm_sandbox_evals.experiment import MatrixCellRef
from llm_sandbox_evals.markdown_report import render_markdown, write_markdown
from llm_sandbox_evals.presentation import ReportPresentationModel
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalDiagnostics,
)
from pydantic_evals.evaluators import EvaluationResult, EvaluatorFailure
from pydantic_evals.evaluators.spec import EvaluatorSpec
from pydantic_evals.reporting import EvaluationReport, ReportCase
import pytest

from llm_sandbox_evals import markdown_report

_JUDGE_SOURCE = EvaluatorSpec(name="code_quality_judge", arguments={"model": "judge"})


def test_markdown_omits_code_judge_section_without_a_requested_judge() -> None:
    markdown = render_markdown(_model(_report(_report_case("not-requested"))))

    assert "## Code Judge" not in markdown


@pytest.mark.parametrize(
    "line_break",
    [
        pytest.param("\n", id="lf"),
        pytest.param("\r\n", id="crlf"),
        pytest.param("\r", id="cr"),
    ],
)
def test_markdown_renders_safe_code_judge_states_in_deterministic_cell_order(line_break: str) -> None:
    judge_reason = f"clear|code{line_break}review"
    failure_type = f"Safe{line_break}JudgeError"
    available = _report_case("case-a")
    available.metadata["judge_enabled"] = True
    available.scores["code_quality_score"] = EvaluationResult(
        name="code_quality_score",
        value=0.75,
        reason=judge_reason,
        source=_JUDGE_SOURCE,
    )
    available.assertions["code_quality_pass"] = EvaluationResult(
        name="code_quality_pass",
        value=True,
        reason=judge_reason,
        source=_JUDGE_SOURCE,
    )
    failed = _report_case("case-b")
    failed.metadata["judge_enabled"] = True
    failed.evaluator_failures.append(
        EvaluatorFailure(
            name="code_quality_judge",
            error_message="RAW_JUDGE_PROVIDER_MESSAGE",
            error_stacktrace="RAW_JUDGE_STACKTRACE",
            source=_JUDGE_SOURCE,
            error_type=failure_type,
        )
    )
    unavailable = _report_case("case-c")
    unavailable.metadata["judge_enabled"] = True
    not_requested = _report_case("case-d")

    markdown = render_markdown(
        _model(EvaluationReport(name="markdown-judge", cases=[not_requested, failed, available, unavailable]))
    )
    section = markdown.split("## Code Judge", maxsplit=1)[1]
    available_row = _judge_row(section, "case-a")
    failed_row = _judge_row(section, "case-b")
    unavailable_row = _judge_row(section, "case-c")
    not_requested_row = _judge_row(section, "case-d")

    assert "available" in available_row
    assert "0.75" in available_row
    assert "true" in available_row.lower()
    assert "clear\\|code review" in available_row
    assert "failed" in failed_row
    assert "Safe JudgeError" in failed_row
    assert "unavailable" in unavailable_row
    assert "not_requested" in not_requested_row
    assert "RAW_JUDGE_PROVIDER_MESSAGE" not in section
    assert "RAW_JUDGE_STACKTRACE" not in section
    assert judge_reason not in section
    assert failure_type not in section
    assert [section.index(row) for row in (available_row, failed_row, unavailable_row, not_requested_row)] == sorted(
        section.index(row) for row in (available_row, failed_row, unavailable_row, not_requested_row)
    )


def test_markdown_deterministic_sections_do_not_depend_on_code_judge_results() -> None:
    available = _report_case("same-case")
    available.metadata["judge_enabled"] = True
    available.scores["code_quality_score"] = EvaluationResult(
        name="code_quality_score", value=1.0, reason="clear", source=_JUDGE_SOURCE
    )
    available.assertions["code_quality_pass"] = EvaluationResult(
        name="code_quality_pass", value=True, reason="clear", source=_JUDGE_SOURCE
    )
    failed = _report_case("same-case")
    failed.metadata["judge_enabled"] = True
    failed.evaluator_failures.append(
        EvaluatorFailure(
            name="code_quality_judge",
            error_message="RAW_JUDGE_PROVIDER_MESSAGE",
            error_stacktrace="RAW_JUDGE_STACKTRACE",
            source=_JUDGE_SOURCE,
            error_type="SafeJudgeError",
        )
    )

    available_markdown = render_markdown(_model(_report(available)))
    failed_markdown = render_markdown(_model(_report(failed)))
    available_before_judge = available_markdown.split("## Code Judge", maxsplit=1)[0]
    failed_before_judge = failed_markdown.split("## Code Judge", maxsplit=1)[0]

    assert "## Method" in available_before_judge
    assert available_before_judge.encode() == failed_before_judge.encode()


def test_markdown_writer_preserves_existing_report_and_cleans_tempfile_when_writing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"
    report_path.write_text("previous complete report", encoding="utf-8")
    monkeypatch.setattr(markdown_report, "NamedTemporaryFile", _failing_named_temporary_file)

    with pytest.raises(OSError, match="write failed"):
        write_markdown(tmp_path, _model(_report(_report_case("write-failure"))))

    assert report_path.read_text(encoding="utf-8") == "previous complete report"
    assert not list(tmp_path.glob(".report.md.*"))


def _model(report: EvaluationReport) -> ReportPresentationModel:
    return ReportPresentationModel.from_report(report)


def _report(*cases: ReportCase) -> EvaluationReport:
    return EvaluationReport(name="markdown-judge", cases=list(cases))


def _report_case(case_id: str) -> ReportCase:
    trace = CaseTrace(
        case_id=case_id,
        candidate_id="baseline",
        model_id="stub",
        request_variant_id="canonical",
        request_text="Turn on a light.",
        category="test",
        answer="Done.",
        required_actions=(),
        desired_entities=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome("correct", "actions", "ok"),
        action_result=ActionResult(True, "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(),
    )
    cell = MatrixCellRef(case_id, "canonical", "baseline", "stub", "home_minimal", None, None)
    return ReportCase(
        name=f"baseline/stub/{case_id}",
        inputs=cell,
        metadata={
            "case_id": case_id,
            "request_variant_id": "canonical",
            "candidate_id": "baseline",
            "model_id": "stub",
            "home": "home_minimal",
            "reasoning_effort": None,
            "temperature": None,
        },
        expected_output=None,
        output=trace,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.0,
        total_duration=0.0,
    )


def _judge_row(section: str, case_id: str) -> str:
    return next(line for line in section.splitlines() if line.startswith("|") and case_id in line)


class _FailingTemporaryFile:
    """Real temporary path whose content write models a filesystem failure."""

    def __init__(self, directory: Path, prefix: str) -> None:
        descriptor, path = mkstemp(dir=directory, prefix=prefix, text=True)
        close(descriptor)
        self._path = Path(path)

    @property
    def name(self) -> str:
        return str(self._path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        return None

    def write(self, _content: str) -> int:
        raise OSError("write failed")


def _failing_named_temporary_file(
    _mode: str,
    *,
    encoding: str | None,
    dir: Path | None,
    prefix: str | None,
    delete: bool,
) -> _FailingTemporaryFile:
    assert encoding == "utf-8"
    assert isinstance(dir, Path)
    assert isinstance(prefix, str)
    assert delete is False
    return _FailingTemporaryFile(dir, prefix)
