import json
from pathlib import Path

from llm_sandbox_evals.html_report import render_html, write_html
import pytest


def test_render_html_embeds_data_island() -> None:
    rendered = render_html(_report(_case("a", "state_read")))

    assert rendered.startswith("<!doctype html>")
    assert '<script type="application/json" id="report-data">' in rendered
    assert "__REPORT_DATA__" not in rendered


@pytest.mark.parametrize(
    "dangerous",
    [
        "</script><img src=x>",
        "safe prefix </script><script>alert(1)</script>",
        "</SCRIPT><img src=x onerror=alert(1)>",
        "</ScRiPt>",
    ],
    ids=["img-breakout", "script-breakout", "uppercase-breakout", "mixedcase-breakout"],
)
def test_render_html_escapes_script_breakout(dangerous: str) -> None:
    rendered = render_html(_report(_case("a", "state_read", output=dangerous)))

    # The inlined JSON island must contain no `</` delimiter that an HTML parser
    # could read as a closing tag, regardless of tag-name casing in the payload.
    island = rendered.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert "</" not in island


def test_write_html_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260709-120102-123456"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(_json_report(_report(_case("case-a", "state_read"))), encoding="utf-8")

    report_html = write_html(run_dir)
    rendered = report_html.read_text(encoding="utf-8")

    assert report_html == run_dir / "report.html"
    assert report_html.exists()
    assert rendered.startswith("<!doctype html>")
    assert "20260709-120102-123456" in rendered
    assert "case-a" in rendered


def test_write_html_derives_run_id_and_created_at(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260709-120102-123456"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(_json_report(_report(_case("case-a", "state_read"))), encoding="utf-8")

    rendered = write_html(run_dir).read_text(encoding="utf-8")

    assert "20260709-120102-123456" in rendered
    assert "2026-07-09T12:01:02.123456+00:00" in rendered


def test_render_html_tolerates_minimal_empty_report() -> None:
    rendered = render_html({"name": "matrix", "cases": [], "analyses": []})

    assert rendered.startswith("<!doctype html>")


def test_render_html_keeps_empty_cells_document_renderable() -> None:
    rendered = render_html(_report(analyses=[]))

    assert rendered.startswith("<!doctype html>")
    assert '<section id="analyses"' in rendered


def _case(
    case_id: str,
    category: str,
    *,
    candidate: str = "baseline",
    model: str = "stub",
    score: float = 1.0,
    checks: list[dict[str, object]] | None = None,
    tool_events: list[dict[str, object]] | None = None,
    recorded_actions: list[dict[str, object]] | None = None,
    error: str | None = None,
    passed: bool = True,
    output: str = "done",
) -> dict[str, object]:
    if checks is None:
        checks = [{"name": "meaningful_oracle", "passed": passed, "required": True, "feedback": "ok"}]
    inputs: dict[str, object] = {
        "case_id": case_id,
        "candidate_id": candidate,
        "model_id": model,
        "home": "home_minimal",
        "category": category,
    }
    first_fail = next((check["name"] for check in checks if check["required"] and not check["passed"]), "none")
    return {
        "name": f"{candidate}/{model}/{case_id}",
        "inputs": inputs,
        "metadata": inputs,
        "output": {
            "case_id": case_id,
            "category": category,
            "candidate_id": candidate,
            "model_id": model,
            "score": score,
            "output": output,
            "tool_call_count": len(tool_events or []),
            "recorded_actions": recorded_actions or [],
            "checks": checks,
            "error": error,
            "tool_events": tool_events or [],
        },
        "scores": {
            "score": {
                "name": "score",
                "value": score,
                "reason": "passed" if passed else "failed",
                "source": "evaluator",
                "evaluator_version": "1",
            }
        },
        "labels": {
            "model_error": "false",
            "outcome": "passed" if passed else "failed",
            "failure_kind": first_fail,
            "error_type": "none" if error is None else "error",
        },
        "assertions": {"required_gates_passed": {"value": passed, "reason": "ok", "source": "evaluator"}},
        "task_duration": 0.1,
        "total_duration": 0.1,
    }


def _report(
    *cases: dict[str, object], analyses: list[dict[str, object]] | None = None, name: str = "matrix"
) -> dict[str, object]:
    return {
        "name": name,
        "cases": list(cases),
        "failures": [],
        "analyses": analyses
        if analyses is not None
        else [
            {"type": "scalar", "title": "Overall mean score", "description": "Mean", "value": 1.0, "unit": ""},
            {"type": "scalar", "title": "Incomplete cells", "description": "Incomplete", "value": 0, "unit": ""},
        ],
        "report_evaluator_failures": [],
        "experiment_metadata": None,
        "trace_id": "trace",
        "span_id": "span",
    }


def _json_report(report: dict[str, object]) -> str:
    return json.dumps(report)
