from llm_sandbox_evals.schema import (
    ActionAnswer,
    AggregateAnswer,
    AggregateExpectation,
    EntityAnswer,
    EntityCollectionAnswer,
    EntityCollectionExpectation,
    EntityExpectation,
    EntityRelationAnswer,
    EntityRelationExpectation,
    EvalCase,
    Expected,
    ExpectedAction,
    NoDataAnswer,
    NoDataExpectation,
    ToolEvent,
)
from llm_sandbox_evals.scoring import evaluate_case
from llm_sandbox_evals.scoring.evidence import normalize_events
import pytest


def _case(expected: Expected) -> EvalCase:
    return EvalCase("case", "state", "home_default", "request", False, expected)


def _execute(output: object, *, status: str = "ok") -> ToolEvent:
    return ToolEvent("execute_home_code", {}, {"execution": {"status": status}, "output": output})


@pytest.mark.parametrize(
    ("answer", "expected_reason", "grounded"),
    [
        pytest.param(EntityAnswer(answer="ignored", entity_id="sensor.temp", value="21"), "ok", True, id="grounded"),
        pytest.param(
            EntityAnswer(answer="ignored", entity_id="sensor.other", value="21"),
            "answer_mismatch",
            False,
            id="wrong-entity",
        ),
        pytest.param(
            EntityAnswer(answer="ignored", entity_id="sensor.temp", value="22"),
            "answer_mismatch",
            False,
            id="wrong-value",
        ),
    ],
)
def test_entity_answer_scores_required_fields_only(answer: EntityAnswer, expected_reason: str, grounded: bool) -> None:
    expected = Expected(
        expectation=EntityExpectation(source="states", entity_id="sensor.temp", input_field="state", value="21")
    )

    outcome, results, _ = evaluate_case(
        _case(expected), answer, (_execute({"entity_id": "sensor.temp", "state": "21"}),)
    )

    assert outcome.reason == expected_reason
    assert results[0].grounded is grounded


def test_correct_entity_value_without_successful_evidence_fails() -> None:
    expected = Expected(
        expectation=EntityExpectation(source="states", entity_id="sensor.temp", input_field="state", value="21")
    )
    answer = EntityAnswer(answer="harmless contextual prose", entity_id="sensor.temp", value="21")

    outcome, results, _ = evaluate_case(
        _case(expected), answer, (_execute({"entity_id": "sensor.temp", "state": "21"}, status="error"),)
    )

    assert outcome.reason == "evidence_missing"
    assert results[0].matched
    assert not results[0].grounded


def test_history_attribute_requires_the_authored_attribute_name() -> None:
    expected = Expected(
        expectation=EntityExpectation(
            source="history",
            entity_id="sensor.temp",
            input_field="attribute",
            input_value="unit_of_measurement",
            value="°C",
        )
    )
    answer = EntityAnswer(answer="", entity_id="sensor.temp", value="°C")
    evidence = (
        ToolEvent(
            "get_history",
            {},
            {"entities": {"sensor.temp": {"rows": [["2026-06-29T12:00:00+00:00", "21", {"note": "°C"}]]}}},
        ),
    )

    outcome, results, _ = evaluate_case(_case(expected), answer, evidence)

    assert outcome.reason == "evidence_missing"
    assert results[0].matched
    assert not results[0].grounded


@pytest.mark.parametrize(
    ("value", "state"),
    [pytest.param(20.5, "correct", id="boundary"), pytest.param(20.5001, "incorrect", id="outside")],
)
def test_entity_tolerance_boundary(value: float, state: str) -> None:
    expected = Expected(
        expectation=EntityExpectation(
            source="states", entity_id="sensor.temp", input_field="state", value=20.0, tolerance=0.5
        )
    )
    answer = EntityAnswer(answer="", entity_id="sensor.temp", value=value)

    outcome, _, _ = evaluate_case(_case(expected), answer, (_execute({"entity_id": "sensor.temp", "state": value}),))

    assert outcome.state == state


def test_collection_requires_exact_set_and_only_returned_ids() -> None:
    expected = Expected(expectation=EntityCollectionExpectation(entity_ids=["light.a", "light.b"]))
    evidence = (_execute([{"entity_id": "light.a", "state": "on"}, {"entity_id": "light.b", "state": "off"}]),)

    exact = evaluate_case(
        _case(expected), EntityCollectionAnswer(answer="", entity_ids=["light.b", "light.a"]), evidence
    )[0]
    contains = evaluate_case(
        _case(expected), EntityCollectionAnswer(answer="", entity_ids=["light.a", "light.b", "light.c"]), evidence
    )[0]

    assert exact.state == "correct"
    assert contains.reason == "answer_mismatch"


def test_aggregate_requires_evidence_from_every_subject() -> None:
    expected = Expected(
        expectation=AggregateExpectation(
            source="states",
            operator="mean",
            subject_ids=["sensor.a", "sensor.b"],
            input_field="state",
            input_value="°C",
            value=20.0,
            unit="°C",
        )
    )
    answer = AggregateAnswer(answer="", value=20.0)
    one_subject = (_execute({"entity_id": "sensor.a", "state": "20", "attributes": {"unit_of_measurement": "°C"}}),)

    outcome, results, _ = evaluate_case(_case(expected), answer, one_subject)

    assert outcome.reason == "evidence_missing"
    assert not results[0].grounded


def test_relation_pair_must_match_successful_relation_evidence() -> None:
    expected = Expected(
        expectation=EntityRelationExpectation(relation="entity_area", entity_id="fan.living", related_id="area_living")
    )
    evidence = (_execute({"entity_id": "fan.living", "state": "off", "area_id": "area_living"}),)

    outcome, _, _ = evaluate_case(
        _case(expected), EntityRelationAnswer(answer="", entity_id="fan.living", related_id="area_living"), evidence
    )

    assert outcome.state == "correct"


def test_no_data_scope_and_empty_rows_must_share_one_envelope() -> None:
    expected = Expected(expectation=NoDataExpectation(source="statistics", scope_entity_ids=["sensor.office_power"]))
    answer = NoDataAnswer(answer="", no_data=True)
    split = (
        ToolEvent(
            "get_statistics", {}, {"statistics": {"sensor.office_power": {"rows": [["2026-01-01", {"mean": 1}]]}}}
        ),
        ToolEvent("get_statistics", {}, {"statistics": {"sensor.other": {"rows": []}}}),
    )
    empty = (ToolEvent("get_statistics", {}, {"statistics": {"sensor.office_power": {"rows": []}}}),)

    assert evaluate_case(_case(expected), answer, split)[0].reason == "evidence_missing"
    assert evaluate_case(_case(expected), answer, empty)[0].state == "correct"


def test_action_answer_has_no_scored_fields_and_uses_ledger() -> None:
    expected = Expected(actions=(ExpectedAction("light", "turn_off", ("light.living",)),))
    action = {"domain": "light", "service": "turn_off", "target": {"entity_id": ["light.living"]}}

    first = evaluate_case(_case(expected), ActionAnswer(answer="one"), (), (action,))[0]
    second = evaluate_case(_case(expected), ActionAnswer(answer="different"), (), (action,))[0]

    assert first.state == second.state == "correct"


def test_execute_normalizer_recurses_actual_nested_failed_envelopes() -> None:
    events = (
        _execute(
            {
                "area": "Living Room",
                "fans": [
                    {
                        "entity_id": "fan.living_fan",
                        "name": "Living Room Fan",
                        "state": "off",
                        "last_changed": "2026-06-29T10:00:00+00:00",
                    }
                ],
            }
        ),
        _execute({"lights": [{"entity_id": "light.deck_sconce", "state": "on"}]}),
        _execute({"entities": [{"entity_id": "cover.office_blinds", "state": "open"}]}),
    )

    evidence = normalize_events(events)
    grounded_ids = {fact.get("subject_id") for fact in evidence.for_kind("value")}

    assert grounded_ids >= {"fan.living_fan", "light.deck_sconce", "cover.office_blinds"}


def test_execute_normalizer_parses_json_string_attributes_once() -> None:
    evidence = normalize_events(
        (_execute({"entity_id": "sensor.temp", "state": "21", "attributes": '{"unit_of_measurement": "°C"}'}),)
    )
    units = [fact for fact in evidence.for_kind("value") if fact.get("attribute_name") == "unit_of_measurement"]

    assert len(units) == 1
    assert units[0].get("value") == "°C"
