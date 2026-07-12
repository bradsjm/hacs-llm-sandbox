from llm_sandbox_evals.html_report import render_html


def test_action_report_renders_structured_assessment_and_escapes_prose() -> None:
    report = {
        "cases": [
            {
                "name": "action-case",
                "inputs": {"case_id": "action-case", "candidate_id": "baseline", "model_id": "stub"},
                "output": {
                    "user_request": "Turn on bedroom light",
                    "answer": "Done. </script><script>alert(1)</script>",
                    "expected_actions": [],
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
                    "action_ledger": {"successful": [], "rejected": []},
                    "tool_events": [],
                    "diagnostics": {},
                    "outcome": {"state": "incorrect", "reason": "wrong_target", "score": 0.0},
                },
            }
        ]
    }

    html = render_html(report, run_id="test-run", created_at="2026-07-12T00:00:00+00:00")

    assert "Action assessment" in html
    assert "Expected service / target / data" in html
    assert "Actual service / target / data" in html
    assert "Service match" in html
    assert "Target match" in html
    assert "Data match" in html
    assert "light.bedroom" in html
    assert "light.living" in html
    assert "switch.garage" in html
    assert "Unexpected actions" in html
    assert 'class="match">yes' in html
    assert 'class="mismatch">no' in html
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Outcome by category" not in html
