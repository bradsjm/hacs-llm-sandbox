import pytest
from llm_sandbox_evals.harness import _recorded_actions_from_tool_events
from llm_sandbox_evals.schema import (
    BlockedOutcome,
    CaseContext,
    EvalCase,
    Expected,
    ExpectedAction,
    ToolEvent,
    ToolResultCheck,
)
from llm_sandbox_evals.scoring import check_case, is_incomplete, score_case


def test_structured_action_outcome_scores_with_tool_call_efficiency() -> None:
    checks = check_case(
        _case(
            Expected(
                actions=(ExpectedAction("light", "turn_off", ("light.living",)),),
            )
        ),
        "The living room temperature is 23.4 °C.",
        _actions(_action("light", "turn_off", "light.living")),
        2,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name == {
        "meaningful_oracle": True,
        "answer_evidence_present": True,
        "execution_ok": True,
        "actions_match": True,
        "tool_calls_within_max": True,
        "tool_call_efficiency": False,
    }
    assert score_case(checks) == pytest.approx(0.9444444444)


def test_successful_outcome_score_decreases_as_tool_calls_increase() -> None:
    expected = Expected(actions=(ExpectedAction("light", "turn_off", ("light.living",)),))
    recorded_actions = _actions(_action("light", "turn_off", "light.living"))

    one_call_checks = check_case(_case(expected), "", recorded_actions, 1, ())
    five_call_checks = check_case(_case(expected), "", recorded_actions, 5, ())

    assert score_case(one_call_checks) == pytest.approx(1.0)
    assert score_case(five_call_checks) == pytest.approx(0.7777777778)


def test_symbolic_answer_evidence_uses_applicable_boundaries() -> None:
    checks = check_case(
        _case(Expected(answer_values=("°C",), max_tool_calls=1)),
        "The living room temperature is 23.4 °C.",
        (),
        0,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["answer_evidence_present"] is True


def test_answer_evidence_failure_is_diagnostic_only() -> None:
    checks = check_case(
        _case(Expected(answer_values=("23.4",), provenance_values=("sensor.living_temp",))),
        "No matching fact here.",
        (),
        2,
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "result = states.get('sensor.living_temp').state"},
                output={"execution": {"status": "ok"}, "output": {"entity_id": "sensor.living_temp", "state": "23.4"}},
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["answer_evidence_present"] is False
    assert passed_by_name["provenance_evidence_present"] is True
    assert passed_by_name["tool_calls_within_max"] is True
    assert score_case(checks) == pytest.approx(0.9444444444)


def test_execute_tool_payload_can_score_without_answer_text_match() -> None:
    tool_events = (
        ToolEvent(
            tool_name="execute_home_code",
            args={"code": "result = states.get('sensor.living_temp')"},
            output={"execution": {"status": "ok"}, "output": {"entity_id": "sensor.living_temp", "state": "23.4"}},
        ),
    )
    checks = check_case(
        _case(
            Expected(
                answer_values=("sensor.living_temp",),
                tool_result_checks=(
                    ToolResultCheck(
                        tool_name="execute_home_code",
                        entity_ids=("sensor.living_temp",),
                        entry_values=("23.4",),
                    ),
                ),
            )
        ),
        "I checked the value.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["tool_result_check_0"] is True
    assert passed_by_name["answer_evidence_present"] is False
    assert passed_by_name["execution_ok"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_execute_tool_payload_evidence_can_span_successful_calls() -> None:
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(
                        tool_name="execute_home_code",
                        entity_ids=("light.living",),
                        entry_values=("label_evening", "on"),
                    ),
                ),
            )
        ),
        "One evening light is on.",
        (),
        2,
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "result = label_registry.async_get_label_by_name('evening')"},
                output={"execution": {"status": "ok"}, "output": {"label_id": "label_evening"}},
            ),
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "result = hass.states.get('light.living')"},
                output={"execution": {"status": "ok"}, "output": {"entity_id": "light.living", "state": "on"}},
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["tool_result_check_0"] is True
    assert score_case(checks) == pytest.approx(0.9444444444)


def test_execute_structured_checks_ignore_printed_lines_and_envelope_metadata() -> None:
    """Only an execute envelope's top-level output can satisfy structured evidence."""
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(
                        tool_name="execute_home_code",
                        entity_ids=("sensor.living_temp",),
                        entry_values=("23.4",),
                    ),
                ),
            )
        ),
        "I checked it.",
        (),
        1,
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "print('sensor.living_temp is 23.4')"},
                output={
                    "execution": {"status": "ok", "note": "sensor.living_temp"},
                    "output": None,
                    "printed": ["sensor.living_temp is 23.4"],
                    "notes": ["sensor.living_temp"],
                },
            ),
        ),
    )

    check = next(check for check in checks if check.name == "tool_result_check_0")
    assert check.passed is False
    assert "empty_output" in check.feedback
    assert "missing_entry_entity:sensor.living_temp" in check.feedback
    assert "missing_entry_value:23.4" in check.feedback
    assert score_case(checks) == 0.0


def test_provenance_evidence_reads_tool_payloads() -> None:
    tool_events = (
        ToolEvent(
            tool_name="execute_home_code",
            args={"code": "result = states.get('sensor.living_temp')"},
            output={"execution": {"status": "ok"}, "output": {"entity_id": "sensor.living_temp"}},
        ),
    )
    checks = check_case(
        _case(Expected(provenance_values=("sensor.living_temp",))),
        "The living room temperature is normal.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["provenance_evidence_present"] is True
    assert score_case(checks) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("tool_event", "tool_check"),
    [
        pytest.param(
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["sensor.living_temp"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "entities": {"sensor.living_temp": {"rows": [["2026-06-29T11:00:00+00:00", "23.4"]]}},
                },
            ),
            ToolResultCheck(tool_name="get_history", entity_ids=("sensor.living_temp",)),
            id="history-entity-rows",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["sensor.living_temp"], "aggregate": "state_counts", "group_by": ["entity_id"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "rows": [{"entity_id": "sensor.living_temp", "state_counts": {"23.4": 1}}],
                },
            ),
            ToolResultCheck(
                tool_name="get_history",
                entity_ids=("sensor.living_temp",),
                entry_values=("23.4",),
            ),
            id="history-analytics-top-level-rows",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["light.living"], "aggregate": {"mode": "time_in_state"}},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "rows": [{"time_in_state": {"off": 43172.0, "on": 43228.0}, "unit": "seconds"}],
                },
            ),
            ToolResultCheck(
                tool_name="get_history",
                entity_ids=("light.living",),
                entry_values=("43228",),
            ),
            id="history-analytics-entity-provenance-from-args",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_statistics",
                args={"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour"},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "period": "hour",
                    "statistics": {
                        "sensor.bedroom_humidity": {
                            "fields": ["mean"],
                            "rows": [["2026-06-29T11:00:00+00:00", {"mean": 63.0}]],
                        }
                    },
                },
            ),
            ToolResultCheck(
                tool_name="get_statistics",
                statistic_ids=("sensor.bedroom_humidity",),
                fields=("mean",),
                period="hour",
            ),
            id="statistics-field-period-rows",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_logbook",
                args={"entity_ids": ["light.living"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "entries": [
                        {"entity_id": "light.living", "when": "2026-06-29T11:00:00+00:00", "message": "turned on"}
                    ],
                },
            ),
            ToolResultCheck(tool_name="get_logbook", entity_ids=("light.living",), entry_values=("turned on",)),
            id="logbook-entry-facts",
        ),
        pytest.param(
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "result = hass.states.get('sensor.living_temp')"},
                output={
                    "execution": {"status": "ok"},
                    "output": {
                        "entity_id": "sensor.living_temp",
                        "state": "23.4",
                        "attributes": {"unit_of_measurement": "°C"},
                    },
                },
            ),
            ToolResultCheck(
                tool_name="execute_home_code",
                entity_ids=("sensor.living_temp",),
                entry_values=("23.4", "°C"),
            ),
            id="execute-home-code-structured-output",
        ),
    ],
)
def test_structured_recorder_evidence_passes(tool_event: ToolEvent, tool_check: ToolResultCheck) -> None:
    checks = check_case(
        _case(Expected(tool_result_checks=(tool_check,))),
        "I checked the recorder data.",
        (),
        1,
        (tool_event,),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["tool_result_check_0"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_history_per_entity_entry_values_do_not_cross_match() -> None:
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(
                        tool_name="get_history",
                        entity_ids=("sensor.living_temp", "sensor.bedroom_humidity"),
                        entry_values_by_entity={
                            "sensor.living_temp": ("25.2",),
                            "sensor.bedroom_humidity": ("64",),
                        },
                        min_results=1,
                    ),
                )
            )
        ),
        "I checked both sensors.",
        (),
        1,
        (
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["sensor.living_temp", "sensor.bedroom_humidity"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "entities": {
                        "sensor.living_temp": {"rows": [["2026-06-29T12:00:00+00:00", "25.2"]]},
                        "sensor.bedroom_humidity": {"rows": [["2026-06-29T12:00:00+00:00", "64"]]},
                    },
                },
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["tool_result_check_0"] is True
    assert score_case(checks) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("tool_event", "tool_check"),
    [
        pytest.param(
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["sensor.living_temp"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "entities": {"sensor.living_temp": {"rows": [["2026-06-29T11:00:00+00:00", "23.4"]]}},
                },
            ),
            ToolResultCheck(tool_name="get_history", entity_ids=("sensor.living_temp",), entry_values=("19.0",)),
            id="history-wrong-entry-value",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_history",
                args={"entity_ids": ["sensor.living_temp"], "aggregate": "state_counts", "group_by": ["entity_id"]},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "rows": [{"entity_id": "sensor.living_temp", "state_counts": {"23.4": 1}}],
                },
            ),
            ToolResultCheck(
                tool_name="get_history",
                entity_ids=("sensor.living_temp",),
                entry_values=("19.0",),
            ),
            id="history-analytics-wrong-entry-value",
        ),
        pytest.param(
            ToolEvent(
                tool_name="get_statistics",
                args={"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour"},
                output={
                    "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "period": "hour",
                    "statistics": {
                        "sensor.bedroom_humidity": {
                            "fields": ["mean"],
                            "rows": [["2026-06-29T11:00:00+00:00", {"mean": 63.0}]],
                        }
                    },
                },
            ),
            ToolResultCheck(
                tool_name="get_statistics",
                statistic_ids=("sensor.bedroom_humidity",),
                fields=("mean",),
                period="hour",
                entry_values=("79.25",),
            ),
            id="statistics-wrong-entry-value",
        ),
    ],
)
def test_history_and_statistics_entry_values_must_match(tool_event: ToolEvent, tool_check: ToolResultCheck) -> None:
    checks = check_case(
        _case(Expected(tool_result_checks=(tool_check,))),
        "I checked the recorder data.",
        (),
        1,
        (tool_event,),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["tool_result_check_0"] is False
    assert score_case(checks) == 0.0


@pytest.mark.parametrize(
    ("rows", "expected_passed", "expected_feedback", "expected_score"),
    [
        pytest.param([], True, "tool=get_statistics", 1.0, id="empty-statistics-rows"),
        pytest.param(
            [["2026-06-29T11:00:00+00:00", {"mean": 63.0}]],
            False,
            "unexpected_results:sensor.bedroom_humidity",
            0.0,
            id="unexpected-statistics-rows",
        ),
    ],
)
def test_empty_statistics_check_rejects_unexpected_rows(
    rows: list[object], expected_passed: bool, expected_feedback: str, expected_score: float
) -> None:
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(
                        tool_name="get_statistics",
                        statistic_ids=("sensor.bedroom_humidity",),
                        fields=("mean",),
                        period="hour",
                        min_results=0,
                    ),
                ),
                max_tool_calls=1,
            )
        ),
        "There were none.",
        (),
        1,
        (
            ToolEvent(
                tool_name="get_statistics",
                args={"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour"},
                output={
                    "window": {"start": "2026-06-29T00:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "period": "hour",
                    "statistics": {"sensor.bedroom_humidity": {"fields": ["mean"], "rows": rows}},
                },
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    feedback_by_name = {check.name: check.feedback for check in checks}
    assert passed_by_name["tool_result_check_0"] is expected_passed
    assert expected_feedback in feedback_by_name["tool_result_check_0"]
    assert score_case(checks) == pytest.approx(expected_score)


@pytest.mark.parametrize(
    ("tool_args", "entries", "expected_passed", "expected_feedback", "expected_score"),
    [
        pytest.param(
            {"entity_ids": ["light.living"]}, [], True, "tool=get_logbook", 1.0, id="matching-direct-entity-args"
        ),
        pytest.param(
            {"entity_ids": ["light.kitchen"]},
            [],
            False,
            "missing_query_entity:light.living",
            0.0,
            id="wrong-direct-entity-args",
        ),
        pytest.param(
            {"area_ids": ["area_living"]},
            [],
            False,
            "unverified_query_scope",
            0.0,
            id="selector-scoped-empty-rejected",
        ),
        pytest.param({}, [], False, "unverified_query_scope", 0.0, id="unscoped-empty-rejected"),
        pytest.param(
            {"entity_ids": ["light.living"]},
            [{"entity_id": "light.living", "when": "2026-06-29T11:00:00+00:00", "message": "turned on"}],
            False,
            "unexpected_results",
            0.0,
            id="unexpected-entries-rejected",
        ),
    ],
)
def test_empty_logbook_check_validates_query_args(
    tool_args: dict[str, object],
    entries: list[dict[str, object]],
    expected_passed: bool,
    expected_feedback: str,
    expected_score: float,
) -> None:
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(tool_name="get_logbook", entity_ids=("light.living",), min_results=0),
                ),
                max_tool_calls=1,
            )
        ),
        "There were none.",
        (),
        1,
        (
            ToolEvent(
                tool_name="get_logbook",
                args=tool_args,
                output={
                    "window": {"start": "2026-06-29T00:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
                    "entries": entries,
                },
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    feedback_by_name = {check.name: check.feedback for check in checks}
    assert passed_by_name["tool_result_check_0"] is expected_passed
    assert expected_feedback in feedback_by_name["tool_result_check_0"]
    assert score_case(checks) == pytest.approx(expected_score)


def test_execution_ok_fails_when_tool_event_is_error_envelope() -> None:
    tool_events = (
        ToolEvent(
            tool_name="execute_home_code",
            args={"code": "boom"},
            output={"execution": {"status": "code_error", "message": "NameError: boom"}},
        ),
    )
    checks = check_case(
        _case(
            Expected(
                tool_result_checks=(
                    ToolResultCheck(tool_name="execute_home_code", entity_ids=("sensor.living_temp",)),
                ),
                max_tool_calls=1,
            )
        ),
        "The value is 23.4.",
        (),
        1,
        tool_events,
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["answer_evidence_present"] is True
    assert passed_by_name["execution_ok"] is False
    assert score_case(checks) == 0.0


def test_empty_expected_actions_rejects_unexpected_recorded_action() -> None:
    checks = check_case(
        _case(Expected(provenance_values=("sensor.living_temp",))),
        "ok",
        _actions(_action("light", "turn_on", "light.living")),
        1,
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "result = hass.states.get('sensor.living_temp')"},
                output={"execution": {"status": "ok"}, "output": {"entity_id": "sensor.living_temp"}},
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_action_target_superset_fails() -> None:
    checks = check_case(
        _case(Expected(actions=(ExpectedAction("light", "turn_on", ("light.living",)),))),
        "",
        _actions(_action("light", "turn_on", ["light.living", "light.kitchen"])),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_split_exact_action_targets_pass() -> None:
    checks = check_case(
        _case(Expected(actions=(ExpectedAction("light", "turn_on", ("light.living", "light.kitchen")),))),
        "",
        _actions(
            _action("light", "turn_on", "light.living"),
            _action("light", "turn_on", "light.kitchen"),
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_overlapping_split_action_target_fails() -> None:
    checks = check_case(
        _case(Expected(actions=(ExpectedAction("light", "turn_on", ("light.living", "light.kitchen")),))),
        "",
        _actions(
            _action("light", "turn_on", ["light.living", "light.kitchen"]),
            _action("light", "turn_on", "light.living"),
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_action_service_data_is_part_of_exact_effect() -> None:
    checks = check_case(
        _case(
            Expected(
                actions=(
                    ExpectedAction(
                        "climate",
                        "set_temperature",
                        ("climate.living",),
                        service_data={"temperature": 21},
                    ),
                )
            )
        ),
        "",
        _actions(_action("climate", "set_temperature", "climate.living", service_data={"temperature": 19})),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_action_service_data_numeric_int_and_float_match() -> None:
    checks = check_case(
        _case(
            Expected(
                actions=(
                    ExpectedAction(
                        "climate",
                        "set_temperature",
                        ("climate.living",),
                        service_data={"temperature": 20, "hvac_mode": "heat", "enabled": True},
                    ),
                )
            )
        ),
        "",
        _actions(
            _action(
                "climate",
                "set_temperature",
                "climate.living",
                service_data={"temperature": 20.0, "hvac_mode": "heat", "enabled": True},
            )
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_recorded_actions_are_enriched_with_invoker_service_data() -> None:
    recorded_actions = _recorded_actions_from_tool_events(
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "await hass.services.async_call(...)"},
                output={
                    "execution": {"status": "ok"},
                    "actions": [
                        {
                            "service": "fan.set_percentage",
                            "target": {"entity_id": ["fan.living_fan"]},
                            "status": "ok",
                        }
                    ],
                },
            ),
        ),
        (
            {
                "domain": "fan",
                "service": "set_percentage",
                "target": {"entity_id": ["fan.living_fan"]},
                "service_data": {"percentage": 50},
            },
        ),
    )

    assert recorded_actions == (
        {
            "domain": "fan",
            "service": "set_percentage",
            "target": {"entity_id": ["fan.living_fan"]},
            "status": "ok",
            "service_data": {"percentage": 50},
        },
    )

    checks = check_case(
        _case(
            Expected(
                actions=(
                    ExpectedAction(
                        "fan",
                        "set_percentage",
                        ("fan.living_fan",),
                        service_data={"percentage": 50},
                    ),
                )
            )
        ),
        "",
        recorded_actions,
        1,
        (),
    )
    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is True


def test_duplicate_action_fails() -> None:
    checks = check_case(
        _case(Expected(actions=(ExpectedAction("light", "turn_on", ("light.living",)),))),
        "",
        _actions(
            _action("light", "turn_on", "light.living"),
            _action("light", "turn_on", "light.living"),
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is False
    assert score_case(checks) == 0.0


def test_intermediate_errored_action_does_not_fail_allowed_action_case() -> None:
    checks = check_case(
        _case(Expected(actions=(ExpectedAction("light", "turn_on", ("light.living",)),))),
        "",
        _actions(
            _action("light", "turn_on", "light.living", status="error", error_key="actions_disabled"),
            _action("light", "turn_on", "light.living"),
        ),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["actions_match"] is True
    assert score_case(checks) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("output", "acknowledgement_values"),
    [
        pytest.param("I can't turn that on from here.", ("cannot", "can't", "unable"), id="cant"),
        pytest.param("I'm unable to turn that on from here.", ("cannot", "can't", "unable"), id="unable"),
    ],
)
def test_blocked_ux_passes_for_acknowledgement_alternatives(
    output: str, acknowledgement_values: tuple[str, ...]
) -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=acknowledgement_values,
                )
            )
        ),
        output,
        _actions(_action("light", "turn_on", "light.living", status="error", error_key="actions_disabled")),
        1,
        (
            ToolEvent(
                tool_name="execute_home_code",
                args={"code": "await hass.services.async_call(...)"},
                output={
                    "execution": {"status": "ok"},
                    "actions": [
                        {
                            "domain": "light",
                            "service": "turn_on",
                            "target": {"entity_id": "light.living"},
                            "status": "error",
                            "error": {"key": "actions_disabled"},
                        }
                    ],
                },
            ),
        ),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["execution_ok"] is True
    assert passed_by_name["blocked_outcome"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_blocked_ux_does_not_require_acknowledgement_text() -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=("cannot", "can't", "unable"),
                )
            )
        ),
        "That action is not available here.",
        _actions(_action("light", "turn_on", "light.living", status="error", error_key="actions_disabled")),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["blocked_outcome"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_blocked_ux_fails_success_claim_or_excessive_retries() -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=("can't",),
                )
            )
        ),
        "I can't do that, but it was successfully turned on.",
        _actions(
            _action("light", "turn_on", "light.living", status="error", error_key="actions_disabled"),
            _action("light", "turn_on", "light.living", status="error", error_key="actions_disabled"),
        ),
        2,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["blocked_outcome"] is False
    assert score_case(checks) == 0.0


def test_blocked_ux_passes_without_attempt_when_answer_acknowledges_inability() -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=("can't",),
                )
            )
        ),
        "I can't turn that on from here.",
        (),
        0,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["blocked_outcome"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_blocked_ux_allows_inability_discussion_of_state_words() -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=("can't",),
                )
            )
        ),
        "I can't tell whether it is closed from here.",
        (),
        0,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["blocked_outcome"] is True
    assert score_case(checks) == pytest.approx(1.0)


def test_blocked_ux_fails_successful_action_without_blocked_attempt() -> None:
    checks = check_case(
        _case(
            Expected(
                blocked_outcome=BlockedOutcome(
                    max_attempts=1,
                    error_keys=("actions_disabled",),
                    acknowledgement_values=("can't",),
                )
            )
        ),
        "I can't turn that on from here.",
        _actions(_action("light", "turn_on", "light.living")),
        1,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["blocked_outcome"] is False
    assert score_case(checks) == 0.0


def test_empty_or_weak_oracle_fails() -> None:
    checks = check_case(
        _case(Expected(answer_excludes=("secret",))),
        "No secret here.",
        (),
        0,
        (),
    )

    passed_by_name = {check.name: check.passed for check in checks}
    assert passed_by_name["meaningful_oracle"] is False
    assert score_case(checks) == 0.0


def test_is_incomplete_only_flags_model_error() -> None:
    from llm_sandbox_evals.schema import CheckResult

    assert is_incomplete([CheckResult("model_error", False, True, "provider down")]) is True
    # Branch boundary: tool_calls_exceeded is a genuine model limit, not incomplete.
    assert is_incomplete([CheckResult("tool_calls_exceeded", False, True, "loop")]) is False
    assert is_incomplete([CheckResult("answer_evidence_present", True, True, "")]) is False


def _case(expected: Expected) -> EvalCase:
    return EvalCase(
        id="scoring-unit",
        category="unit",
        home="home_default",
        user_request="score this outcome",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=expected,
    )


def _actions(*actions: dict[str, object]) -> tuple[dict[str, object], ...]:
    return actions


def _action(
    domain: str,
    service: str,
    entity_id: str | list[str],
    *,
    status: str = "ok",
    error_key: str | None = None,
    service_data: dict[str, object] | None = None,
) -> dict[str, object]:
    action: dict[str, object] = {
        "domain": domain,
        "service": service,
        "target": {"entity_id": entity_id},
        "status": status,
    }
    if error_key is not None:
        action["error"] = {"key": error_key}
    if service_data is not None:
        action["service_data"] = service_data
    return action
