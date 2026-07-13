import json
from os import close
from pathlib import Path
from tempfile import mkstemp
from typing import Self

from llm_sandbox_evals.experiment import MatrixCellRef
from llm_sandbox_evals.html_report import render_html, write_html
from llm_sandbox_evals.reports import _REPORT_ADAPTER
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
    case_id: str = "action-fail",
    state: str = "incorrect",
    action_reason: str | None = "wrong_target",
    failure: str | None = None,
    answer: str | None = None,
    reasoning_effort: str | None = "high",
    usage: dict[str, object] | None = None,
    tool_calls: int = 1,
) -> CaseTrace:
    return CaseTrace(
        case_id=case_id,
        candidate_id="baseline",
        model_id="stub",
        answer=answer,
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        outcome=CaseOutcome(state, action_reason),
        action_result=ActionResult(state == "correct", action_reason or "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(
            tool_calls=tool_calls,
            elapsed_seconds=0.5,
            usage=usage,
            failure=failure,
        ),
        reasoning_effort=reasoning_effort,
        temperature=0.7,
        user_request="Turn on bedroom light",
    )


def _case(trace: CaseTrace) -> ReportCase:
    cell = MatrixCellRef(trace.case_id, "baseline", "stub", "home_minimal", trace.reasoning_effort, trace.temperature)
    return ReportCase(
        name=f"baseline/stub/{trace.case_id}",
        inputs=cell,
        metadata={
            "run_id": "html-report",
            "case_id": trace.case_id,
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
    fail = _trace(
        state="incorrect",
        action_reason="wrong_target",
        answer="Done. </script><script>alert(1)</script>",
        usage={"total_tokens": 42, "cost": 0.001},
    )
    ok = _trace(case_id="action-ok", state="correct", action_reason="ok", answer="Turned on.", usage=None)
    incomplete = _trace(
        case_id="action-timeout",
        state="incomplete",
        action_reason=None,
        failure="timeout",
        tool_calls=0,
        usage=None,
    )
    return EvaluationReport(
        name="html-report",
        cases=[_case(fail), _case(ok), _case(incomplete)],
        experiment_metadata={
            "models": [{"model_id": "stub", "reasoning_effort": "high", "temperature": 0.7, "variant_label": "stub(high)"}],
        },
    )


def _embedded_report(html_text: str) -> dict[str, object]:
    marker = '<script type="application/json" id="report-data">'
    start = html_text.index(marker) + len(marker)
    end = html_text.index("</script>", start)
    # render_html neutralizes "</" as "<\/" so the payload never closes its host script early.
    return json.loads(html_text[start:end].replace("<\\/", "</"))


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
    assert incomplete_cell["action_reason"] is None
    assert incomplete_cell["result"] == "incomplete·timeout"
    # No incomplete cell reads as a scored action reason anywhere in the page.
    assert "incomplete·action_mismatch" not in html_text


def test_payload_carries_variant_and_usage_metrics() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    fail_cell = next(cell for cell in payload["cells"] if cell["case_id"] == "action-fail")
    assert fail_cell["variant"] == "stub(high)"
    # Per-cell metrics and the trace usage fallback both reach the renderer.
    assert fail_cell["metrics"]["tool_calls"] == 1
    assert fail_cell["diagnostics"]["usage"]["total_tokens"] == 42
    # Aggregates carry candidate and variant identity.
    assert payload["aggregates"][0]["candidate"] == "baseline"
    assert payload["aggregates"][0]["variant"] == "stub(high)"


def test_payload_carries_all_eval_diagnostics_fields() -> None:
    html_text = render_html(_report(), run_id="20260713-100000-000000")
    payload = _embedded_report(html_text)

    fail_cell = next(cell for cell in payload["cells"] if cell["case_id"] == "action-fail")
    diagnostics = fail_cell["diagnostics"]
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
    """Persist a valid scoring-v6 report.json that load_report accepts."""
    payload = json.loads(_REPORT_ADAPTER.dump_json(_report()))
    payload["scoring_version"] = 6
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")
