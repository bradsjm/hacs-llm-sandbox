import json

from llm_sandbox_evals.html_report import render_html


def _v5_report() -> dict[str, object]:
    """A minimal two-cell scoring-v5 report exercising the action-only trace shape."""
    return {
        "scoring_version": 5,
        "analyses": [{"type": "scalar", "title": "Overall correct rate", "value": 0.5, "unit": ""}],
        "cases": [
            {
                "name": "baseline/stub/action-fail",
                "inputs": {
                    "case_id": "action-fail",
                    "candidate_id": "baseline",
                    "model_id": "stub",
                    "home": "home_minimal",
                },
                "scores": {"score": {"name": "score", "value": 0.0, "reason": "wrong_target"}},
                "output": {
                    "case_id": "action-fail",
                    "candidate_id": "baseline",
                    "model_id": "stub",
                    "answer": "Done. </script><script>alert(1)</script>",
                    "required_actions": [
                        {
                            "domain": "light",
                            "service": "turn_on",
                            "target_entity_ids": ["light.bedroom"],
                            "service_data": None,
                        }
                    ],
                    "outcome": {"state": "incorrect", "reason": "wrong_target", "score": 0.0},
                    "action_result": {
                        "passed": False,
                        "reason": "wrong_target",
                        "comparisons": [
                            {
                                "expected": {
                                    "domain": "light",
                                    "service": "turn_on",
                                    "target_entity_ids": ["light.bedroom"],
                                    "service_data": {"brightness": 100},
                                },
                                "actual": {
                                    "domain": "light",
                                    "service": "turn_on",
                                    "target_entity_ids": ["light.living"],
                                    "service_data": {"brightness": 100},
                                },
                                "service_matches": True,
                                "target_matches": False,
                                "service_data_matches": True,
                                "matched": False,
                            }
                        ],
                        "unexpected_actions": [
                            {
                                "domain": "switch",
                                "service": "turn_off",
                                "target_entity_ids": ["switch.garage"],
                                "service_data": {},
                            }
                        ],
                    },
                    "action_ledger": {
                        "successful": [
                            {
                                "domain": "light",
                                "service": "turn_on",
                                "target": {"entity_id": ["light.living"]},
                                "status": "ok",
                            }
                        ],
                        "rejected": [
                            {
                                "domain": "switch",
                                "service": "turn_off",
                                "target": {"entity_id": ["switch.garage"]},
                                "status": "rejected",
                            }
                        ],
                    },
                    "tool_events": [
                        {
                            "tool_name": "execute_home_code",
                            "args": {"code": "hass.services.call('light', 'turn_on')"},
                            "output": {"ok": True},
                            "call_index": 0,
                        }
                    ],
                    "diagnostics": {
                        "tool_calls": 1,
                        "failed_tool_calls": 0,
                        "model_turns": 2,
                        "elapsed_seconds": 0.5,
                        "usage": {"total_tokens": 42, "cost": 0.001},
                    },
                    "scoring_version": 5,
                    "provider_error": None,
                    "user_request": "Turn on bedroom light",
                    "conversation_id": "conv-1",
                },
            },
            {
                "name": "baseline/stub/action-ok",
                "inputs": {
                    "case_id": "action-ok",
                    "candidate_id": "baseline",
                    "model_id": "stub",
                    "home": "home_minimal",
                },
                "scores": {"score": {"name": "score", "value": 1.0, "reason": "ok"}},
                "output": {
                    "case_id": "action-ok",
                    "candidate_id": "baseline",
                    "model_id": "stub",
                    "answer": "Turned on the bedroom light.",
                    "required_actions": [
                        {
                            "domain": "light",
                            "service": "turn_on",
                            "target_entity_ids": ["light.bedroom"],
                            "service_data": None,
                        }
                    ],
                    "outcome": {"state": "correct", "reason": "ok", "score": 1.0},
                    "action_result": {"passed": True, "reason": "ok", "comparisons": [], "unexpected_actions": []},
                    "action_ledger": {"successful": [], "rejected": []},
                    "tool_events": [],
                    "diagnostics": {
                        "tool_calls": 1,
                        "failed_tool_calls": 0,
                        "model_turns": 2,
                        "elapsed_seconds": 0.3,
                        "usage": None,
                    },
                    "scoring_version": 5,
                    "provider_error": None,
                    "user_request": "Turn on bedroom light",
                    "conversation_id": "conv-2",
                },
            },
        ],
    }


def _embedded_report(html: str) -> dict[str, object]:
    """Recover the JSON payload the client renderer consumes from the report shell."""
    marker = '<script type="application/json" id="report-data">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    # render_html neutralizes "</" as "<\\/" so the payload never closes its host
    # <script> early; reverse that to parse the original data.
    payload: dict[str, object] = json.loads(html[start:end].replace("<\\/", "</"))
    return payload


def test_report_shell_exposes_kpis_table_matrix_and_filters() -> None:
    html = render_html(_v5_report(), run_id="test-run", created_at="2026-07-12T00:00:00+00:00")

    # Hero KPI cards.
    for kpi in ("overall-correct-rate", "pass-count", "fail-count", "incomplete-count", "case-count"):
        assert f'id="{kpi}"' in html
    # Table/matrix toggle and both view containers.
    assert 'id="view-toggle"' in html
    assert 'data-view="matrix"' in html
    assert 'data-view="table"' in html
    assert 'id="results-grid"' in html
    assert 'id="results-matrix"' in html
    # Filter controls that survive the action-only contract.
    for control in ("candidate-filter", "model-filter", "status-filter", "quick-filter", "group-case"):
        assert f'id="{control}"' in html
    # Offline-degradation and third-party assets with SRI + fallback behavior.
    assert "<noscript>" in html
    assert "ag-grid-community@32.3.3" in html
    assert 'integrity="sha384-' in html


def test_report_binds_only_v5_action_fields_and_drops_read_panels() -> None:
    html = render_html(_v5_report(), run_id="test-run", created_at="2026-07-12T00:00:00+00:00")

    # Charts and filters rebind the removed read-era category onto the stored reason.
    assert "Outcome by reason" in html
    assert "Tool calls by reason" in html
    assert "Outcome by category" not in html
    assert 'id="category-filter"' not in html
    # No read-era conclusions / grounding / authored-expectation panels remain.
    assert "conclusions" not in html
    assert "grounding" not in html
    assert "Submitted answer" not in html
    assert "Authored expectation" not in html
    # Structured action assessment replaces the removed claims/grounding inspector.
    assert "Action assessment" in html
    assert "Expected service / target / data" in html
    assert "Actual service / target / data" in html


def test_embedded_payload_preserves_action_comparisons_ledger_and_diagnostics() -> None:
    html = render_html(_v5_report(), run_id="test-run", created_at="2026-07-12T00:00:00+00:00")
    payload = _embedded_report(html)

    output = payload["cases"][0]["output"]
    result = output["action_result"]
    comparison = result["comparisons"][0]
    # Per-dimension comparison the inspector renders as match indicators.
    assert comparison["service_matches"] is True
    assert comparison["target_matches"] is False
    assert comparison["service_data_matches"] is True
    assert comparison["matched"] is False
    assert comparison["expected"]["target_entity_ids"] == ["light.bedroom"]
    assert comparison["actual"]["target_entity_ids"] == ["light.living"]
    # Unexpected effect and rejected ledger entry are both carried for diagnostics.
    assert result["unexpected_actions"][0]["target_entity_ids"] == ["switch.garage"]
    assert output["action_ledger"]["successful"][0]["service"] == "turn_on"
    assert output["action_ledger"]["rejected"][0]["status"] == "rejected"
    # Chronological tool evidence and diagnostics/usage remain intact.
    assert output["tool_events"][0]["tool_name"] == "execute_home_code"
    assert output["diagnostics"]["usage"]["total_tokens"] == 42
    # Reason taxonomy is stored and served verbatim.
    assert output["outcome"]["reason"] == "wrong_target"


def test_report_neutralizes_script_breakout_in_answer() -> None:
    html = render_html(_v5_report(), run_id="test-run", created_at="2026-07-12T00:00:00+00:00")

    # The answer's "</script>" cannot close the data island; it is escaped to "<\\/script>".
    assert "Done. <\\/script><script>alert(1)<\\/script>" in html
    assert "Done. </script><script>alert(1)</script>" not in html
