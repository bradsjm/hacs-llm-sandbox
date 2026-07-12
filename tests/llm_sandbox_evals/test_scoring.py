from llm_sandbox_evals.schema import (
    AggregateClaim,
    BlockedOutcome,
    CaseContext,
    CollectionClaim,
    EvalAnswer,
    EvalCase,
    EventClaim,
    Expected,
    ExpectedAction,
    ExpectedConclusion,
    NoDataClaim,
    RelationClaim,
    ToolEvent,
    ValueClaim,
)
from llm_sandbox_evals.scoring import evaluate_case
from pydantic import ValidationError
import pytest


def test_claim_contract_rejects_open_ended_and_invalid_shapes() -> None:
    with pytest.raises(ValidationError):
        EvalAnswer.model_validate(
            {
                "answer": "x",
                "claims": [
                    {"kind": "value", "subject_kind": "entity", "subject_id": "x", "field": "unknown", "value": "y"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        ValueClaim(subject_kind="entity", subject_id="x", field="state", attribute_name="wrong", value="on")
    with pytest.raises(ValidationError):
        CollectionClaim(collection="entity_ids", filter_kind="all", filter_value="x", items=["x"])
    with pytest.raises(ValidationError):
        AggregateClaim(source="states", operator="count", subject_ids=["x", "x"], input_field="none", value=2)
    with pytest.raises(ValidationError):
        ExpectedConclusion(
            claim=CollectionClaim(collection="entity_ids", filter_kind="all", items=["x"]), assertion="equals"
        )
    with pytest.raises(ValidationError):
        EvalAnswer.model_validate(
            {
                "answer": "x",
                "claims": [
                    {
                        "kind": "value",
                        "subject_kind": "entity",
                        "subject_id": "x",
                        "field": "state",
                        "predicate": "on",
                        "value": "on",
                    }
                ],
            }
        )


def test_expected_rejects_empty_and_mixed_blocked_oracles() -> None:
    with pytest.raises(ValueError, match="must declare"):
        Expected()
    with pytest.raises(ValueError, match="cannot also"):
        Expected(actions=(ExpectedAction("light", "turn_on"),), blocked_outcome=BlockedOutcome())


def test_failed_execute_then_successful_grounded_call_scores_correct() -> None:
    expected = Expected(
        conclusions=(
            ExpectedConclusion(
                claim=ValueClaim(subject_kind="entity", subject_id="light.living", field="state", value="on"),
                assertion="equals",
            ),
        )
    )
    answer = EvalAnswer(
        answer="on", claims=[ValueClaim(subject_kind="entity", subject_id="light.living", field="state", value="on")]
    )
    outcome, conclusions, _ = evaluate_case(
        _case(expected),
        answer,
        (
            ToolEvent("execute_home_code", {}, {"execution": {"status": "code_error"}}),
            ToolEvent(
                "execute_home_code",
                {},
                {"execution": {"status": "ok"}, "output": {"entity_id": "light.living", "state": "on"}},
            ),
        ),
    )
    assert outcome.state == "correct"
    assert conclusions[0].grounding_status == "grounded"


def test_parallel_direct_results_union_without_path_or_final_event_dependency() -> None:
    claims = [
        ValueClaim(subject_kind="entity", subject_id="sensor.a", field="state", value="20"),
        ValueClaim(subject_kind="entity", subject_id="sensor.b", field="state", value="21"),
    ]
    expected = Expected(conclusions=tuple(ExpectedConclusion(claim=claim, assertion="equals") for claim in claims))
    events = (
        ToolEvent(
            "execute_home_code",
            {},
            {"execution": {"status": "ok"}, "output": {"entity_id": "sensor.a", "state": "20"}},
            batch_index=0,
            batch_size=2,
        ),
        ToolEvent(
            "execute_home_code",
            {},
            {"execution": {"status": "ok"}, "output": {"entity_id": "sensor.b", "state": "21"}},
            batch_index=0,
            batch_size=2,
        ),
        ToolEvent("get_history", {}, {"status": "error"}),
    )
    outcome, _, _ = evaluate_case(_case(expected), EvalAnswer(answer="", claims=claims), events)
    assert outcome.state == "correct"


def test_wrong_entity_value_cannot_ground_and_extra_claim_fails() -> None:
    expected_claim = ValueClaim(subject_kind="entity", subject_id="sensor.a", field="state", value="20")
    wrong = ValueClaim(subject_kind="entity", subject_id="sensor.b", field="state", value="20")
    expected = Expected(conclusions=(ExpectedConclusion(claim=expected_claim, assertion="equals"),))
    outcome, _, _ = evaluate_case(
        _case(expected),
        EvalAnswer(answer="", claims=[expected_claim, wrong]),
        (
            ToolEvent(
                "execute_home_code",
                {},
                {"execution": {"status": "ok"}, "output": {"entity_id": "sensor.a", "state": "20"}},
            ),
        ),
    )
    assert outcome.state == "incorrect"


def test_aggregate_recomputed_and_tolerance_is_explicit() -> None:
    claim = AggregateClaim(
        source="states",
        operator="mean",
        subject_ids=["sensor.a", "sensor.b"],
        input_field="state",
        input_value="state",
        value=21.0,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="approximate", tolerance=0.01),))
    answer_claim = claim.model_copy(update={"value": 20.5})
    events = tuple(
        ToolEvent(
            "execute_home_code", {}, {"execution": {"status": "ok"}, "output": {"entity_id": entity, "state": state}}
        )
        for entity, state in (("sensor.a", 20), ("sensor.b", 21))
    )
    outcome, _, _ = evaluate_case(_case(expected), EvalAnswer(answer="", claims=[answer_claim]), events)
    assert outcome.state == "incorrect"


def test_empty_logbook_requires_returned_exact_scope() -> None:
    claim = NoDataClaim(source="logbook", scope_entity_ids=["light.a", "light.b"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    good = ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a", "light.b"]}, "entries": []})
    bad = ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a"]}, "entries": []})
    assert evaluate_case(_case(expected), EvalAnswer(answer="", claims=[claim]), (good,))[0].state == "correct"
    assert evaluate_case(_case(expected), EvalAnswer(answer="", claims=[claim]), (bad,))[0].state == "incorrect"


def test_empty_statistics_rejects_rows_for_the_matching_statistic_id() -> None:
    claim = NoDataClaim(source="statistics", scope_entity_ids=["sensor.a"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    event = ToolEvent(
        "get_statistics",
        {},
        {"statistics": {"sensor.a": {"fields": ["mean"], "rows": [["2026-01-01T00:00:00+00:00", {"mean": 5.0}]]}}},
    )
    assert evaluate_case(_case(expected), EvalAnswer(answer="", claims=[claim]), (event,))[0].state == "incorrect"


@pytest.mark.parametrize(
    ("recorded", "passed"),
    [
        (
            (
                {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.a"}, "status": "ok"},
                {"domain": "light", "service": "turn_on", "target": {"entity_id": "light.b"}, "status": "ok"},
            ),
            True,
        ),
        (
            (
                {
                    "domain": "light",
                    "service": "turn_on",
                    "target": {"entity_id": ["light.a", "light.a"]},
                    "status": "ok",
                },
            ),
            False,
        ),
        (
            (
                {
                    "domain": "light",
                    "service": "turn_on",
                    "target": {"entity_id": ["light.a", "light.b", "light.c"]},
                    "status": "ok",
                },
            ),
            False,
        ),
    ],
)
def test_action_ledger_accepts_disjoint_splits_and_rejects_duplicates_or_supersets(
    recorded: tuple[dict[str, object], ...], passed: bool
) -> None:
    expected = Expected(actions=(ExpectedAction("light", "turn_on", ("light.a", "light.b")),))
    outcome, _, actions = evaluate_case(_case(expected), EvalAnswer(answer=""), (), recorded)
    assert outcome.state == ("correct" if passed else "incorrect")
    assert actions[0].passed is passed


@pytest.mark.parametrize(
    ("operator", "input_value", "value", "rows"),
    [
        (
            "count",
            "state",
            3,
            [
                ["2026-01-01T00:00:00+00:00", "on"],
                ["2026-01-01T01:00:00+00:00", "off"],
                ["2026-01-01T02:00:00+00:00", "on"],
            ],
        ),
        ("mean", "state", 2, [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]),
        ("minimum", "state", 1, [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]),
        ("maximum", "state", 3, [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]),
        ("sum", "state", 4, [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]),
        ("duration_seconds", "on", 3600, [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T01:00:00+00:00", "off"]]),
        ("time_in_state", "on", 3600, [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T01:00:00+00:00", "off"]]),
        (
            "first_seen",
            "state",
            "2026-01-01T00:00:00+00:00",
            [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]],
        ),
        (
            "last_seen",
            "state",
            "2026-01-01T01:00:00+00:00",
            [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]],
        ),
    ],
)
def test_history_aggregate_operators_recompute_numeric_strings_and_timestamps(
    operator: str, input_value: object, value: object, rows: list[list[object]]
) -> None:
    claim = AggregateClaim(
        source="history",
        operator=operator,
        subject_ids=["sensor.a"],
        input_field="state",
        input_value=input_value,
        value=value,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    answer = EvalAnswer(answer="", claims=[claim])
    event = ToolEvent("get_history", {}, {"entities": {"sensor.a": {"rows": rows}}})
    assert evaluate_case(_case(expected), answer, (event,))[0].state == "correct"


def test_aggregate_recomputed_value_must_match_submitted_claim() -> None:
    claim = AggregateClaim(
        source="history", operator="mean", subject_ids=["sensor.a"], input_field="state", input_value="state", value=2
    )
    wrong = claim.model_copy(update={"value": 99})
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]}}},
    )
    assert evaluate_case(_case(expected), EvalAnswer(answer="", claims=[wrong]), (event,))[0].state == "incorrect"


def test_convert_aggregate_uses_declared_units() -> None:
    claim = AggregateClaim(
        source="states",
        operator="convert",
        subject_ids=["sensor.a"],
        input_field="state",
        input_value="°F",
        value=0.0,
        unit="°C",
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="approximate", tolerance=0.01),))
    event = ToolEvent(
        "execute_home_code", {}, {"execution": {"status": "ok"}, "output": {"entity_id": "sensor.a", "state": "32"}}
    )
    assert evaluate_case(_case(expected), EvalAnswer(answer="", claims=[claim]), (event,))[0].state == "correct"


@pytest.mark.parametrize(
    ("claim", "output"),
    [
        (
            EventClaim(source="history", entity_id="sensor.a", event_kind="state_transition", value="on", when="t"),
            {"entities": {"sensor.a": {"rows": [["t", "on"]]}}},
        ),
        (
            EventClaim(
                source="logbook", entity_id="light.a", event_kind="logbook_message", value="turned on", when="t"
            ),
            {"entries": [{"entity_id": "light.a", "when": "t", "message": "turned on"}]},
        ),
        (
            EventClaim(
                source="automation_run",
                entity_id="automation.a",
                event_kind="automation_run",
                value="triggered",
                when="t",
            ),
            {"automations": [{"entity_id": "automation.a", "runs": [{"when": "t", "message": "triggered"}]}]},
        ),
    ],
)
def test_event_claims_use_their_source_field_not_a_generic_value(claim: EventClaim, output: dict[str, object]) -> None:
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    tool = (
        "get_history"
        if claim.source == "history"
        else "get_logbook"
        if claim.source == "logbook"
        else "get_automation"
    )
    assert (
        evaluate_case(_case(expected), EvalAnswer(answer="", claims=[claim]), (ToolEvent(tool, {}, output),))[0].state
        == "correct"
    )


def test_execute_records_preserve_registry_system_and_service_associations() -> None:
    claims = [
        ValueClaim(subject_kind="device", subject_id="device.a", field="manufacturer", value="Acme"),
        ValueClaim(subject_kind="area", subject_id="area.a", field="name", value="Kitchen"),
        ValueClaim(subject_kind="repair", subject_id="repair.a", field="status", value=True),
        ValueClaim(subject_kind="notification", subject_id="note.a", field="message", value="Battery low"),
        RelationClaim(
            subject_kind="entity",
            subject_id="light.a",
            relation="entity_device",
            object_kind="device",
            object_id="device.a",
        ),
        RelationClaim(
            subject_kind="device",
            subject_id="device.a",
            relation="device_area",
            object_kind="area",
            object_id="area.a",
        ),
        RelationClaim(
            subject_kind="entity",
            subject_id="climate.a",
            relation="entity_service",
            object_kind="service",
            object_id="climate.set_temperature",
        ),
    ]
    output = {
        "device": {"id": "device.a", "manufacturer": "Acme", "area_id": "area.a"},
        "area": {"area_id": "area.a", "name": "Kitchen"},
        "repair": {"issue_id": "repair.a", "active": True},
        "notification": {"notification_id": "note.a", "message": "Battery low"},
        "entity": {"entity_id": "light.a", "device_id": "device.a"},
        "service": {"entity_id": "climate.a", "services": ["climate.set_temperature"]},
    }
    expected = Expected(conclusions=tuple(ExpectedConclusion(claim=claim, assertion="equals") for claim in claims))
    assert (
        evaluate_case(
            _case(expected),
            EvalAnswer(answer="", claims=claims),
            (ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": list(output.values())}),),
        )[0].state
        == "correct"
    )


def test_collection_filter_keeps_all_facts_for_each_subject() -> None:
    claim = CollectionClaim(
        collection="entity_ids", filter_kind="area", filter_value="area.a", items=["light.a", "switch.a"]
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="exact_items"),))
    output = [
        {"entity_id": "light.a", "area_id": "area.a", "state": "on"},
        {"entity_id": "switch.a", "area_id": "area.a", "state": "off"},
        {"entity_id": "light.b", "area_id": "area.b", "state": "on"},
    ]
    assert (
        evaluate_case(
            _case(expected),
            EvalAnswer(answer="", claims=[claim]),
            (ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output}),),
        )[0].state
        == "correct"
    )


def test_blocked_action_requires_every_authored_error_key() -> None:
    expected = Expected(
        blocked_outcome=BlockedOutcome(
            error_keys=("actions_disabled", "service_not_found"),
            actions=(ExpectedAction("light", "turn_on", ("light.a",)),),
        )
    )
    action = {
        "domain": "light",
        "service": "turn_on",
        "target": {"entity_id": "light.a"},
        "status": "error",
        "error": {"key": "actions_disabled"},
    }
    outcome, _, results = evaluate_case(_case(expected), EvalAnswer(answer=""), (), (action,))
    assert outcome.state == "incorrect"
    assert "missing_error_key" in results[0].mismatches


def test_blocked_action_rejects_wrong_entity_target() -> None:
    expected = Expected(
        blocked_outcome=BlockedOutcome(
            error_keys=("actions_disabled",),
            actions=(ExpectedAction("light", "turn_on", ("light.a",)),),
        )
    )
    action = {
        "domain": "light",
        "service": "turn_on",
        "target": {"entity_id": "light.b"},
        "status": "error",
        "error": {"key": "actions_disabled"},
    }
    outcome, _, results = evaluate_case(_case(expected), EvalAnswer(answer=""), (), (action,))
    assert outcome.state == "incorrect"
    assert any(mismatch.startswith("target:") for mismatch in results[0].mismatches)


def test_extra_aggregate_claim_must_match_its_recomputed_value() -> None:
    expected_claim = AggregateClaim(
        source="history", operator="mean", subject_ids=["sensor.a"], input_field="state", input_value="state", value=2
    )
    extra_claim = expected_claim.model_copy(update={"value": 99})
    expected = Expected(conclusions=(ExpectedConclusion(claim=expected_claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]}}},
    )
    answer = EvalAnswer(answer="", claims=[expected_claim, extra_claim])
    assert evaluate_case(_case(expected), answer, (event,))[0].state == "incorrect"


def _case(expected: Expected) -> EvalCase:
    return EvalCase("scoring", "unit", "home_default", "request", False, expected, llm_context=CaseContext())
