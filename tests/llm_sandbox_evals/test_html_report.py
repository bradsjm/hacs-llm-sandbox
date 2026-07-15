from collections.abc import Callable
import json
from os import close
from pathlib import Path
from tempfile import mkstemp
from typing import Self, TypedDict, cast

from llm_sandbox_evals.experiment import MatrixCellRef
from llm_sandbox_evals.html_report import render_html, write_html
from llm_sandbox_evals.reports import _REPORT_ADAPTER
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    EndStateResult,
    EvalDiagnostics,
    RequiredAction,
)
from pydantic_evals.evaluators import EvaluationResult, EvaluatorFailure
from pydantic_evals.evaluators.spec import EvaluatorSpec
from pydantic_evals.reporting import EvaluationReport, ReportCase
import pytest


def _trace(
    *,
    case_id: str = "action-fail",
    state: str = "incorrect",
    score_reason: str | None = "wrong_target",
    scoring_mode: str | None = "actions",
    failure: str | None = None,
    answer: str | None = None,
    reasoning_effort: str | None = "high",
    usage: dict[str, object] | None = None,
    tool_calls: int = 1,
    action_ledger: ActionLedger | None = None,
    request_variant_id: str = "canonical",
) -> CaseTrace:
    return CaseTrace(
        case_id=case_id,
        candidate_id="baseline",
        model_id="stub",
        request_variant_id=request_variant_id,
        request_text="Turn on bedroom light",
        category="test",
        answer=answer,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        desired_entities=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome(state, scoring_mode, score_reason),
        action_result=ActionResult(state == "correct", score_reason or "ok"),
        action_ledger=action_ledger or ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(
            tool_calls=tool_calls,
            elapsed_seconds=0.5,
            usage=usage,
            failure=failure,
        ),
        reasoning_effort=reasoning_effort,
        temperature=0.7,
    )


def _case(trace: CaseTrace) -> ReportCase:
    cell = MatrixCellRef(
        trace.case_id,
        trace.request_variant_id,
        "baseline",
        "stub",
        "home_minimal",
        trace.reasoning_effort,
        trace.temperature,
    )
    return ReportCase(
        name=f"baseline/stub/{trace.case_id}/{trace.request_variant_id}",
        inputs=cell,
        metadata={
            "run_id": "html-report",
            "case_id": trace.case_id,
            "request_variant_id": trace.request_variant_id,
            "candidate_id": "baseline",
            "model_id": "stub",
            "home": "home_minimal",
            "reasoning_effort": trace.reasoning_effort,
            "temperature": trace.temperature,
            "variant_label": "stub(high)",
        },
        expected_output=None,
        output=trace,
        metrics={"tool_calls": trace.diagnostics.tool_calls},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.0,
        total_duration=0.0,
    )


def _report() -> EvaluationReport:
    wrong_target_action = {
        "domain": "light",
        "service": "turn_on",
        "target": {"entity_id": ["light.kitchen"]},
        "service_data": {},
        "status": "success",
    }
    fail = _trace(
        state="incorrect",
        score_reason="wrong_target",
        answer="Done. </script><script>alert(1)</script>",
        usage={"total_tokens": 42, "cost": 0.001},
        action_ledger=ActionLedger(successful=(wrong_target_action,)),
    )
    ok = _trace(case_id="action-ok", state="correct", score_reason="ok", answer="Turned on.", usage=None)
    incomplete = _trace(
        case_id="action-timeout",
        state="incomplete",
        score_reason=None,
        failure="timeout",
        tool_calls=0,
        usage=None,
    )
    robust_canonical = _trace(
        case_id="brightness-robustness",
        state="correct",
        score_reason="ok",
    )
    robust_paraphrase_ok = _trace(
        case_id="brightness-robustness",
        state="correct",
        score_reason="ok",
        request_variant_id="paraphrase_currently",
    )
    robust_paraphrase_fail = _trace(
        case_id="brightness-robustness",
        state="incorrect",
        score_reason="wrong_target",
        request_variant_id="paraphrase_half",
    )
    return EvaluationReport(
        name="html-report",
        cases=[
            _case(fail),
            _case(ok),
            _case(incomplete),
            _case(robust_canonical),
            _case(robust_paraphrase_ok),
            _case(robust_paraphrase_fail),
        ],
        experiment_metadata={
            "models": [
                {"model_id": "stub", "reasoning_effort": "high", "temperature": 0.7, "variant_label": "stub(high)"}
            ],
        },
    )


def _report_with_judge_states() -> EvaluationReport:
    """Build native report records covering each public advisory judge state."""
    report = _report()
    source = EvaluatorSpec(name="code_quality_judge", arguments={"model": "judge"})
    available = next(report_case for report_case in report.cases if report_case.output.case_id == "action-ok")
    available.metadata["judge_enabled"] = True
    available.scores["code_quality_score"] = EvaluationResult(
        name="code_quality_score",
        value=0.75,
        reason="judge-reason </script><script>reason_breakout()</script>",
        source=source,
    )
    available.assertions["code_quality_pass"] = EvaluationResult(
        name="code_quality_pass",
        value=True,
        reason="judge-reason </script><script>reason_breakout()</script>",
        source=source,
    )
    failed = next(report_case for report_case in report.cases if report_case.output.case_id == "action-timeout")
    failed.metadata["judge_enabled"] = True
    failed.evaluator_failures.append(
        EvaluatorFailure(
            name="code_quality_judge",
            error_message="RAW_JUDGE_ERROR_MESSAGE",
            error_stacktrace="RAW_JUDGE_STACKTRACE",
            source=source,
            error_type="JudgeFailure</script><script>failure_breakout()</script>",
        )
    )
    unavailable = next(
        report_case
        for report_case in report.cases
        if report_case.output.case_id == "brightness-robustness"
        and report_case.output.request_variant_id == "canonical"
    )
    unavailable.metadata["judge_enabled"] = True
    return report


class _EmbeddedReport(TypedDict):
    counts: dict[str, int]
    cells: list[dict[str, object]]
    aggregates: list[dict[str, object]]
    paraphrase_counts: dict[str, object]
    task_robustness: list[dict[str, object]]
    category_aggregates: list[dict[str, object]]


def _embedded_report(html_text: str) -> _EmbeddedReport:
    marker = '<script type="application/json" id="report-data">'
    start = html_text.index(marker) + len(marker)
    end = html_text.index("</script>", start)
    # render_html neutralizes "</" as "<\/" so the payload never closes its host script early.
    return cast(_EmbeddedReport, json.loads(html_text[start:end].replace("<\\/", "</")))


@pytest.mark.parametrize(
    ("report_factory", "judge_requested"),
    [
        pytest.param(_report, False, id="no-requested-judges"),
        pytest.param(_report_with_judge_states, True, id="requested-judge"),
    ],
)
def test_code_judge_section_is_present_only_when_a_cell_requested_judging(
    report_factory: Callable[[], EvaluationReport], judge_requested: bool
) -> None:
    html_text = render_html(report_factory(), run_id="20260713-100000-000000")

    # The section marker is a stable DOM contract, not its display prose or layout.
    assert ('id="code-judge"' in html_text) is judge_requested


def test_code_judge_payload_projects_mixed_states_without_exposing_failure_details() -> None:
    html_text = render_html(_report_with_judge_states(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)
    cells = {(cell["case_id"], cell["request_variant_id"]): cell for cell in payload["cells"]}

    assert cells[("action-fail", "canonical")]["judge"]["status"] == "not_requested"
    available = cells[("action-ok", "canonical")]["judge"]
    assert (available["status"], available["score"], available["passed"], available["reason"]) == (
        "available",
        0.75,
        True,
        "judge-reason </script><script>reason_breakout()</script>",
    )
    failed = cells[("action-timeout", "canonical")]["judge"]
    assert (failed["status"], failed["score"], failed["passed"], failed["reason"]) == (
        "failed",
        None,
        None,
        None,
    )
    assert failed["failure"] == {
        "error_type": "JudgeFailure</script><script>failure_breakout()</script>",
        "message": None,
    }
    assert cells[("brightness-robustness", "canonical")]["judge"]["status"] == "unavailable"
    # Raw evaluator details never become report data, including the advisory failure payload.
    assert "RAW_JUDGE_ERROR_MESSAGE" not in html_text
    assert "RAW_JUDGE_STACKTRACE" not in html_text


def test_code_judge_data_cannot_break_out_of_the_embedded_report_data_script() -> None:
    html_text = render_html(_report_with_judge_states(), run_id="20260713-100000-000000")

    assert "judge-reason <\\/script><script>reason_breakout()<\\/script>" in html_text
    assert "JudgeFailure<\\/script><script>failure_breakout()<\\/script>" in html_text
    assert "judge-reason </script><script>reason_breakout()</script>" not in html_text
    assert "JudgeFailure</script><script>failure_breakout()</script>" not in html_text


def test_code_judge_values_do_not_change_deterministic_report_data() -> None:
    baseline_html = render_html(_report(), run_id="20260713-100000-000000")
    judged_html = render_html(_report_with_judge_states(), run_id="20260713-100000-000000")
    baseline = _embedded_report(baseline_html)
    judged = _embedded_report(judged_html)

    # Aggregates and charts continue to consume the deterministic scoring projection.
    for key in ("counts", "paraphrase_counts", "task_robustness", "aggregates", "category_aggregates"):
        assert baseline[key] == judged[key]
    deterministic_cell_fields = (
        "case_id",
        "request_variant_id",
        "category",
        "candidate_id",
        "model_id",
        "variant",
        "result",
        "cause",
        "state",
        "scoring_mode",
        "score_reason",
        "diagnostics",
        "metrics",
    )
    assert [{field: cell[field] for field in deterministic_cell_fields} for cell in baseline["cells"]] == [
        {field: cell[field] for field in deterministic_cell_fields} for cell in judged["cells"]
    ]
    # The existing CSV score remains tied to deterministic cell outcome, never advisory judge data.
    assert "c.state==='correct'?1:0" in judged_html


def test_hero_exposes_quality_coverage_variant_config() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")

    for element_id in ("quality", "coverage", "incomplete", "total", "candidate-variants", "variant-config"):
        assert f'id="{element_id}"' in html_text
    # Variant configuration line surfaces the resolved model identity.
    assert "stub(high)" in html_text
    # No legacy completed/pass-fail scalar cards remain.
    assert "Completed" not in html_text
    assert 'id="pass-count"' not in html_text


def test_comparison_and_charts_split_quality_and_operational_failures() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")

    assert 'id="comparison"' in html_text
    assert 'id="heatmap"' in html_text
    # Charts separate quality (scored cells) from operational failures.
    assert 'id="quality-chart"' in html_text
    assert 'id="failure-chart"' in html_text


def test_incomplete_cell_renders_operational_cause_not_action_mismatch() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    incomplete_cell = next(cell for cell in payload["cells"] if cell["state"] == "incomplete")
    # The cause is the operational failure, never a forced action_mismatch.
    assert incomplete_cell["cause"] == "timeout"
    assert incomplete_cell["score_reason"] is None
    assert incomplete_cell["result"] == "incomplete·timeout"
    # No incomplete cell reads as a scored action reason anywhere in the page.
    assert "incomplete·action_mismatch" not in html_text


def test_payload_carries_variant_usage_and_raw_action_evidence() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    fail_cell = next(cell for cell in payload["cells"] if cell["case_id"] == "action-fail")
    assert fail_cell["variant"] == "stub(high)"
    # Per-cell metrics and the trace usage fallback both reach the renderer.
    assert fail_cell["metrics"]["tool_calls"] == 1
    assert fail_cell["diagnostics"]["usage"]["total_tokens"] == 42
    assert fail_cell["score_reason"] == "wrong_target"
    assert fail_cell["cause"] == "wrong_target"
    assert fail_cell["result"] == "incorrect·wrong_target"
    assert fail_cell["action_ledger"] == {
        "successful": [
            {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": ["light.kitchen"]},
                "service_data": {},
                "status": "success",
            }
        ],
        "rejected": [],
    }
    # Aggregates carry candidate and variant identity.
    assert payload["aggregates"][0]["candidate"] == "baseline"
    assert payload["aggregates"][0]["variant"] == "stub(high)"


def test_payload_carries_request_variant_and_paraphrase_robustness() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    assert all("request_variant_id" in cell for cell in payload["cells"])
    assert payload["paraphrase_counts"]["correct"] == 1
    assert payload["paraphrase_counts"]["scored"] == 2
    assert payload["paraphrase_counts"]["quality_rate"] == 0.5
    assert payload["paraphrase_counts"]["quality_interval"]
    robust = next(value for value in payload["task_robustness"] if value["case_id"] == "brightness-robustness")
    assert robust["correct_variants"] == 2
    assert robust["total_variants"] == 3
    assert robust["all_passed"] is False


def test_html_renders_paraphrase_card_robustness_and_request_variant_column() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")

    assert 'id="paraphrase-quality"' in html_text
    assert 'id="paraphrase-quality-ci"' in html_text
    assert 'id="task-robustness"' in html_text
    assert "<h2>Task robustness</h2>" in html_text
    assert "By category (all request variants)" in html_text
    assert "<th>Request variant</th>" in html_text
    assert "<th>Model variant</th>" in html_text
    assert "'request_variant_id'" in html_text


def test_payload_carries_all_eval_diagnostics_fields() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    fail_cell = next(cell for cell in payload["cells"] if cell["case_id"] == "action-fail")
    diagnostics = cast(dict[str, object], fail_cell["diagnostics"])
    # The raw payload is projected from the slots dataclass via asdict, retaining every field.
    expected_fields = {
        "tool_calls",
        "successful_tool_calls",
        "failed_tool_calls",
        "execute_repairs",
        "model_turns",
        "parallel_batches",
        "max_batch_size",
        "elapsed_seconds",
        "cap_exhausted",
        "usage",
        "failure",
    }
    assert expected_fields.issubset(diagnostics.keys())


def test_report_neutralizes_script_breakout_in_answer() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")

    # The answer's "</script>" cannot close the data island; it is escaped to "<\\/script>".
    assert "Done. <\\/script><script>alert(1)<\\/script>" in html_text
    assert "Done. </script><script>alert(1)</script>" not in html_text


def test_invalid_report_failure_page_does_not_claim_valid_or_rerenderable(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    # A legacy/invalid report cannot be loaded, so the failure page must not offer a re-render.
    (run_dir / "report.json").write_text(json.dumps({"scoring_version": 5, "cases": []}), encoding="utf-8")

    page = write_html(run_dir).read_text(encoding="utf-8")

    # The invalid-report page surfaces the load failure without claiming validity or a recovery command.
    assert "could not be loaded" in page
    assert "python -m llm_sandbox_evals report" not in page
    assert "still valid" not in page


def test_valid_report_render_failure_recovery_command_includes_runs_dir_and_is_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import html as html_module
    import re
    import shlex

    from llm_sandbox_evals.reports import load_report

    from llm_sandbox_evals import cli as eval_cli

    # A custom runs-dir parent with a space proves the command safely represents the path.
    runs_parent = tmp_path / "my runs dir"
    run_dir = runs_parent / "render-fail-run"
    run_dir.mkdir(parents=True)
    _write_valid_report(run_dir)

    import llm_sandbox_evals.html_report as report_module

    monkeypatch.setattr(report_module, "render_html", _raise_render_error)

    page = write_html(run_dir).read_text(encoding="utf-8")

    assert "Report render failed" in page
    assert "still valid" in page
    # Extract the recovery command from the <pre> block and html-unescape it.
    command = html_module.unescape(re.search(r"<pre>(.*?)</pre>", page, re.DOTALL).group(1))
    # The command round-trips through shlex into the CLI parser with the correct paths.
    parsed = eval_cli._build_parser().parse_args(shlex.split(command)[3:])
    assert parsed.command == "report"
    assert parsed.run_id == run_dir.name
    assert Path(parsed.runs_dir) == runs_parent
    # The parsed paths resolve to a loadable report — the command is usable as intended.
    assert load_report(Path(parsed.runs_dir) / parsed.run_id).cases


def test_html_writer_cleans_temporary_file_when_content_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "html-write-failure"
    run_dir.mkdir()
    _write_valid_report(run_dir)

    import llm_sandbox_evals.html_report as report_module

    monkeypatch.setattr(report_module, "NamedTemporaryFile", _failing_named_temporary_file)

    with pytest.raises(OSError, match="write failed"):
        write_html(run_dir)

    # A failed write exposes neither a target page nor a stranded atomic-write tempfile.
    assert not (run_dir / "report.html").exists()
    assert not list(run_dir.glob(".report.html.*"))


def _raise_render_error(*_args: object, **_kwargs: object) -> str:
    raise RuntimeError("render exploded")


class _FailingTemporaryFile:
    """A real temporary path whose content write simulates a filesystem failure."""

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


def _write_valid_report(run_dir: Path) -> None:
    """Persist a valid scoring-v9 report.json that load_report accepts."""
    payload = json.loads(_REPORT_ADAPTER.dump_json(_report()))
    payload["scoring_version"] = 9
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")
