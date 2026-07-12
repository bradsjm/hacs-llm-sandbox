import json
from pathlib import Path
from typing import cast

from llm_sandbox_evals.html_report import render_html, write_html
import pytest


def test_render_html_embeds_v2_trace_data_island() -> None:
    report = _report(_case())
    rendered = render_html(report, run_id="v2-run", created_at="2026-07-12T00:00:00+00:00")

    assert rendered.startswith("<!doctype html>")
    island = _data_island(rendered)
    assert island["cases"][0]["output"]["outcome"]["state"] == "correct"
    assert island["cases"][0]["output"]["expected"]["conclusions"][0]["assertion"] == "equals"
    assert island["cases"][0]["output"]["conclusions"][0]["semantic_status"] == "matched"
    assert island["cases"][0]["output"]["conclusions"][0]["grounding_status"] == "grounded"
    assert island["cases"][0]["output"]["action_ledger"]["successful"][0]["domain"] == "light"
    assert island["cases"][0]["output"]["tool_events"][0]["output"]["execution"]["status"] == "ok"
    assert island["cases"][0]["output"]["diagnostics"]["tool_calls"] == 2
    assert island["cases"][0]["output"]["answer"]["answer"] == "The light is on."


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
def test_render_html_escapes_v2_answer_script_breakout(dangerous: str) -> None:
    report = _report(_case(answer_text=dangerous))
    rendered = render_html(report)

    assert "</" not in rendered.split('id="report-data">', 1)[1].split("</script>", 1)[0]


def test_render_html_keeps_v2_detail_projection_and_raw_trace_fields() -> None:
    rendered = render_html(_report(_case()))

    for marker in (
        "trace.expected",
        "trace.conclusions",
        "trace.action_ledger",
        "trace.diagnostics",
        "trace.answer",
        "Raw case JSON",
    ):
        assert marker in rendered
    for value in ("grounded", "light", "turn_on", "sensor.living_temp", "normal", "provider-token"):
        assert value in rendered
    assert "execute_home_code" in rendered
    assert "get_logbook" in rendered
    markers = (
        "Authored expected conclusions and effects",
        "Submitted claims and grounding",
        "Action ledgers and results",
        "Chronological tool evidence",
        "Diagnostics",
        "Unrestricted final answer",
        "Raw case JSON",
    )
    assert [rendered.index(marker) for marker in markers] == sorted(rendered.index(marker) for marker in markers)
    assert '.map((event, index) => toolCard(event.tool_name === "execute_home_code")(event, index))' in rendered


def test_write_html_round_trip_preserves_v2_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260709-120102-123456"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(json.dumps(_report(_case())), encoding="utf-8")

    report_html = write_html(run_dir)
    rendered = report_html.read_text(encoding="utf-8")

    assert report_html == run_dir / "report.html"
    assert report_html.exists()
    assert "20260709-120102-123456" in rendered
    assert "provider-token" in rendered


def test_render_html_tolerates_minimal_empty_report() -> None:
    rendered = render_html({"name": "matrix", "cases": [], "analyses": []})

    assert rendered.startswith("<!doctype html>")


def _case(*, answer_text: str = "The light is on.") -> dict[str, object]:
    value_claim = {
        "kind": "value",
        "subject_kind": "entity",
        "subject_id": "sensor.living_temp",
        "field": "state",
        "attribute_name": None,
        "value": "on",
    }
    expected = {
        "conclusions": [{"claim": value_claim, "assertion": "equals", "tolerance": None}],
        "actions": [
            {"domain": "light", "service": "turn_on", "target_entity_ids": ["light.living"], "service_data": None}
        ],
        "blocked_outcome": None,
    }
    return {
        "name": "baseline/stub/case-v2",
        "inputs": {
            "case_id": "case-v2",
            "candidate_id": "baseline",
            "model_id": "stub",
            "home": "home_minimal",
            "category": "action",
        },
        "metadata": {
            "case_id": "case-v2",
            "candidate_id": "baseline",
            "model_id": "stub",
            "home": "home_minimal",
            "category": "action",
        },
        "output": {
            "case_id": "case-v2",
            "category": "state",
            "candidate_id": "baseline",
            "model_id": "stub",
            "answer": {"answer": answer_text, "claims": [value_claim]},
            "expected": expected,
            "outcome": {"state": "correct", "reason": "ok", "score": 1.0},
            "conclusions": [
                {
                    "expected": expected["conclusions"][0],
                    "answer_claim": value_claim,
                    "semantic_status": "matched",
                    "grounding_status": "grounded",
                    "reason": "ok",
                }
            ],
            "actions": [{"status": "allowed", "passed": True, "mismatches": []}],
            "action_ledger": {
                "successful": [{"domain": "light", "service": "turn_on", "target": {"entity_id": "light.living"}}],
                "rejected": [],
            },
            "tool_events": [
                {
                    "tool_name": "execute_home_code",
                    "args": {"code": "return hass.states['sensor.living_temp']"},
                    "output": {
                        "execution": {"status": "ok"},
                        "output": {"entity_id": "sensor.living_temp", "state": "on"},
                        "note": "provider-token",
                    },
                    "call_index": 0,
                    "turn_index": 0,
                    "batch_index": 0,
                    "batch_size": 1,
                },
                {
                    "tool_name": "get_logbook",
                    "args": {"entity_ids": ["light.living"]},
                    "output": {"entries": []},
                    "call_index": 1,
                    "turn_index": 1,
                    "batch_index": 0,
                    "batch_size": 1,
                },
            ],
            "diagnostics": {
                "tool_calls": 2,
                "successful_tool_calls": 1,
                "failed_tool_calls": 1,
                "execute_repairs": 1,
                "model_turns": 2,
                "parallel_batches": 0,
                "max_batch_size": 1,
                "elapsed_seconds": 1.25,
                "cap_exhausted": False,
                "usage": {
                    "requests": 2,
                    "request_tokens": 10,
                    "response_tokens": 20,
                    "total_tokens": 30,
                    "cost": None,
                },
                "failure": "normal",
            },
            "provider_error": None,
            "user_request": "What is the living room temperature?",
        },
        "scores": {
            "score": {"name": "score", "value": 1.0, "reason": "ok", "source": "evaluator", "evaluator_version": "2"}
        },
        "labels": {"outcome": "correct", "incomplete": False, "failure_classification": "normal"},
        "assertions": {},
        "task_duration": 1.25,
        "total_duration": 1.25,
    }


def _report(case: dict[str, object]) -> dict[str, object]:
    return {
        "name": "matrix",
        "cases": [case],
        "failures": [],
        "analyses": [
            {
                "type": "scalar",
                "title": "Overall correct rate",
                "description": "Correct rate",
                "value": 1.0,
                "unit": "",
            }
        ],
        "report_evaluator_failures": [],
        "experiment_metadata": None,
        "trace_id": "trace",
        "span_id": "span",
    }


def _data_island(rendered: str) -> dict[str, object]:
    island = rendered.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    decoded = json.loads(island)
    if not isinstance(decoded, dict):
        raise AssertionError("report data island must contain a JSON object")
    return cast(dict[str, object], decoded)
