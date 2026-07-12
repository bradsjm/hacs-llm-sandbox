from typing import Literal

from llm_sandbox_evals.schema import (
    ActionAnswer,
    AggregateClaim,
    BlockedOutcome,
    CaseContext,
    CollectionClaim,
    EvalCase,
    EventClaim,
    Expected,
    ExpectedAction,
    ExpectedConclusion,
    Finding,
    ListAnswer,
    NoDataClaim,
    ReadAnswer,
    RelationClaim,
    ToolEvent,
    ValueClaim,
    select_answer_shape,
)
from llm_sandbox_evals.scoring import evaluate_case
import pytest


@pytest.mark.parametrize(
    ("expected", "shape"),
    [
        pytest.param(
            Expected(
                conclusions=(
                    ExpectedConclusion(
                        claim=CollectionClaim(collection="entity_ids", filter_kind="all", items=["light.a"]),
                        assertion="exact_items",
                    ),
                )
            ),
            ListAnswer,
            id="collection",
        ),
        pytest.param(
            Expected(
                conclusions=(
                    ExpectedConclusion(
                        claim=ValueClaim(subject_kind="entity", subject_id="sensor.a", field="state", value="on"),
                        assertion="equals",
                    ),
                )
            ),
            ReadAnswer,
            id="read",
        ),
        pytest.param(Expected(blocked_outcome=BlockedOutcome()), ActionAnswer, id="action"),
    ],
)
def test_select_answer_shape_maps_conclusions_to_flat_shapes(
    expected: Expected, shape: type[ActionAnswer | ReadAnswer | ListAnswer]
) -> None:
    assert select_answer_shape(expected) is shape


def test_expected_rejects_empty_and_mixed_blocked_oracles() -> None:
    with pytest.raises(ValueError, match="must declare"):
        Expected()
    with pytest.raises(ValueError, match="cannot also"):
        Expected(actions=(ExpectedAction("light", "turn_on"),), blocked_outcome=BlockedOutcome())


def test_read_only_case_has_no_synthetic_action_result() -> None:
    claim = ValueClaim(subject_kind="entity", subject_id="light.living", field="state", value="on")
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "execute_home_code",
        {},
        {"execution": {"status": "ok"}, "output": {"entity_id": "light.living", "state": "on"}},
    )

    outcome, _, actions = evaluate_case(_case(expected), _read_answer("light.living", "on"), (event,))

    assert outcome.state == "correct"
    assert actions == ()


def test_read_only_unexpected_successful_effect_is_incorrect_without_action_result() -> None:
    claim = ValueClaim(subject_kind="entity", subject_id="light.living", field="state", value="on")
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "execute_home_code",
        {},
        {"execution": {"status": "ok"}, "output": {"entity_id": "light.living", "state": "on"}},
    )
    recorded = ({"domain": "light", "service": "turn_off", "status": "ok"},)

    outcome, _, actions = evaluate_case(_case(expected), _read_answer("light.living", "on"), (event,), recorded)

    assert (outcome.state, outcome.reason) == ("incorrect", "unexpected_effect")
    assert actions == ()


def test_failed_execute_then_successful_grounded_call_scores_correct() -> None:
    expected = Expected(
        conclusions=(
            ExpectedConclusion(
                claim=ValueClaim(subject_kind="entity", subject_id="light.living", field="state", value="on"),
                assertion="equals",
            ),
        )
    )
    answer = ReadAnswer(answer="on", findings=[Finding(subject="light.living", value="on")])
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
    answer = ReadAnswer(answer="", findings=[Finding(subject=claim.subject_id, value=claim.value) for claim in claims])
    outcome, _, _ = evaluate_case(_case(expected), answer, events)
    assert outcome.state == "correct"


def test_wrong_entity_value_cannot_ground_and_extra_claim_fails() -> None:
    expected_claim = ValueClaim(subject_kind="entity", subject_id="sensor.a", field="state", value="20")
    wrong = ValueClaim(subject_kind="entity", subject_id="sensor.b", field="state", value="20")
    expected = Expected(conclusions=(ExpectedConclusion(claim=expected_claim, assertion="equals"),))
    outcome, _, _ = evaluate_case(
        _case(expected),
        ReadAnswer(
            answer="",
            findings=[
                Finding(subject=expected_claim.subject_id, value="20"),
                Finding(subject=wrong.subject_id, value="20"),
            ],
        ),
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
    submitted_value = 20.5
    events = tuple(
        ToolEvent(
            "execute_home_code", {}, {"execution": {"status": "ok"}, "output": {"entity_id": entity, "state": state}}
        )
        for entity, state in (("sensor.a", 20), ("sensor.b", 21))
    )
    outcome, _, _ = evaluate_case(_case(expected), _read_answer("sensor.a", submitted_value), events)
    assert outcome.state == "incorrect"


@pytest.mark.parametrize(
    ("submitted_value", "expected_state"),
    [pytest.param(20.495, "correct", id="within-tolerance"), pytest.param(20.48, "incorrect", id="outside-tolerance")],
)
def test_authored_approximate_claim_uses_tolerance_for_grounding_and_global_check(
    submitted_value: float, expected_state: str
) -> None:
    expected_claim = AggregateClaim(
        source="history",
        operator="mean",
        subject_ids=["sensor.a"],
        input_field="state",
        input_value="state",
        value=20.5,
    )
    expected = Expected(
        conclusions=(ExpectedConclusion(claim=expected_claim, assertion="approximate", tolerance=0.01),)
    )
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "20.5"]]}}},
    )

    outcome, _, _ = evaluate_case(_case(expected), _read_answer("sensor.a", submitted_value), (event,))

    assert outcome.state == expected_state


def test_extra_unselected_approximate_claim_remains_exactly_grounded() -> None:
    expected_claim = AggregateClaim(
        source="history",
        operator="mean",
        subject_ids=["sensor.a"],
        input_field="state",
        input_value="state",
        value=20.5,
    )
    expected = Expected(
        conclusions=(ExpectedConclusion(claim=expected_claim, assertion="approximate", tolerance=0.01),)
    )
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "20.5"]]}}},
    )

    answer = ReadAnswer(
        answer="",
        findings=[Finding(subject="sensor.a", value=20.495), Finding(subject="sensor.a", value=20.496)],
    )
    outcome, _, _ = evaluate_case(_case(expected), answer, (event,))

    assert outcome.state == "incorrect"


def test_empty_logbook_requires_returned_exact_scope() -> None:
    claim = NoDataClaim(source="logbook", scope_entity_ids=["light.a", "light.b"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    good = ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a", "light.b"]}, "entries": []}, call_index=1)
    bad = ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a"]}, "entries": []}, call_index=1)
    answer = ReadAnswer(answer="", findings=[])
    assert evaluate_case(_case(expected), answer, (good,))[0].state == "correct"
    assert evaluate_case(_case(expected), answer, (bad,))[0].state == "incorrect"


def test_empty_logbook_does_not_union_fragmented_scopes() -> None:
    claim = NoDataClaim(source="logbook", scope_entity_ids=["light.a", "light.b"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    events = (
        ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a"]}, "entries": []}, call_index=1),
        ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.b"]}, "entries": []}, call_index=2),
    )
    assert evaluate_case(_case(expected), ReadAnswer(answer="", findings=[]), events)[0].state == "incorrect"


def test_empty_logbook_ignores_unrelated_nonempty_event() -> None:
    claim = NoDataClaim(source="logbook", scope_entity_ids=["light.a"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    events = (
        ToolEvent("get_logbook", {}, {"scope": {"entity_ids": ["light.a"]}, "entries": []}, call_index=1),
        ToolEvent(
            "get_logbook",
            {},
            {"scope": {"entity_ids": ["light.other"]}, "entries": [{"entity_id": "light.other"}]},
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), ReadAnswer(answer="", findings=[]), events)[0].state == "correct"


def test_empty_statistics_rejects_rows_for_the_matching_statistic_id() -> None:
    claim = NoDataClaim(source="statistics", scope_entity_ids=["sensor.a"])
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="empty"),))
    event = ToolEvent(
        "get_statistics",
        {},
        {"statistics": {"sensor.a": {"fields": ["mean"], "rows": [["2026-01-01T00:00:00+00:00", {"mean": 5.0}]]}}},
    )
    assert evaluate_case(_case(expected), ReadAnswer(answer="", findings=[]), (event,))[0].state == "incorrect"


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
    outcome, _, actions = evaluate_case(_case(expected), ActionAnswer(answer="", success=True), (), recorded)
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
    answer = _aggregate_answer(claim)
    event = ToolEvent(
        "get_history",
        {},
        {
            "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T01:00:00+00:00"},
            "entities": {"sensor.a": {"rows": rows}},
        },
    )
    assert evaluate_case(_case(expected), answer, (event,))[0].state == "correct"


@pytest.mark.parametrize(
    ("claim", "tool_name", "output"),
    [
        pytest.param(
            AggregateClaim(
                source="history",
                operator="maximum",
                subject_ids=["sensor.a", "sensor.b"],
                input_field="state",
                input_value="state",
                value=3,
            ),
            "get_history",
            {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "3"]]}}},
            id="history",
        ),
        pytest.param(
            AggregateClaim(
                source="statistics",
                operator="mean",
                subject_ids=["sensor.a", "sensor.b"],
                input_field="mean",
                input_value=3,
                value=3,
            ),
            "get_statistics",
            {"statistics": {"sensor.a": {"fields": ["mean"], "rows": [["2026-01-01T00:00:00+00:00", {"mean": 3.0}]]}}},
            id="statistics",
        ),
        pytest.param(
            AggregateClaim(
                source="logbook",
                operator="count",
                subject_ids=["light.a", "light.b"],
                input_field="event_message",
                input_value="turned on",
                value=1,
            ),
            "get_logbook",
            {"entries": [{"entity_id": "light.a", "when": "2026-01-01T00:00:00+00:00", "message": "turned on"}]},
            id="logbook",
        ),
    ],
)
def test_recorder_aggregates_require_qualifying_source_from_every_subject(
    claim: AggregateClaim, tool_name: str, output: dict[str, object]
) -> None:
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))

    outcome, conclusions, _ = evaluate_case(
        _case(expected), _aggregate_answer(claim), (ToolEvent(tool_name, {}, output),)
    )

    assert outcome.state == "incorrect"
    assert conclusions[0].grounding_status == "ungrounded"


def test_aggregate_recomputed_value_must_match_submitted_claim() -> None:
    claim = AggregateClaim(
        source="history", operator="mean", subject_ids=["sensor.a"], input_field="state", input_value="state", value=2
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]}}},
    )
    assert evaluate_case(_case(expected), _read_answer("sensor.a", 99), (event,))[0].state == "incorrect"


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
        "execute_home_code",
        {},
        {
            "execution": {"status": "ok"},
            "output": {
                "entity_id": "sensor.a",
                "state": "32",
                "attributes": {"unit_of_measurement": "°F"},
            },
        },
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), (event,))[0].state == "correct"


def test_history_duration_operators_use_selected_state_intervals() -> None:
    rows = [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T01:00:00+00:00", "off"]]
    for operator in ("duration_seconds", "time_in_state"):
        claim = AggregateClaim(
            source="history",
            operator=operator,
            subject_ids=["light.a"],
            input_field="state",
            input_value="on",
            value=3600,
        )
        expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
        event = ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T01:00:00+00:00"},
                "entities": {"light.a": {"rows": rows}},
            },
        )
        outcome, _, _ = evaluate_case(_case(expected), _aggregate_answer(claim), (event,))
        assert outcome.state == "correct"


@pytest.mark.parametrize(
    ("output", "expected_value", "expected_state"),
    [
        pytest.param([{"entity_id": "light.a", "duration_seconds": 10}], 10, "incorrect", id="missing-subject"),
        pytest.param(
            [
                {"entity_id": "light.a", "duration_seconds": 10},
                {"entity_id": "light.b", "duration_seconds": 20},
            ],
            30,
            "correct",
            id="complete-evidence",
        ),
    ],
)
def test_state_duration_requires_explicit_evidence_from_every_subject(
    output: list[dict[str, object]], expected_value: int, expected_state: str
) -> None:
    claim = AggregateClaim(
        source="states",
        operator="duration_seconds",
        subject_ids=["light.a", "light.b"],
        input_field="none",
        value=expected_value,
        unit="seconds",
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output})

    outcome, _, _ = evaluate_case(_case(expected), _aggregate_answer(claim), (event,))

    assert outcome.state == expected_state


@pytest.mark.parametrize(
    ("window", "passed"),
    [
        (
            {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
            True,
        ),
        ({}, False),
        ({"start": "2026-01-01T00:00:00+00:00", "end": "not-a-timestamp"}, False),
        (
            {"start": "2026-01-01T04:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
            False,
        ),
    ],
)
def test_history_duration_requires_valid_window_endpoint(window: dict[str, object], passed: bool) -> None:
    claim = AggregateClaim(
        source="history",
        operator="time_in_state",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=7200,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {"window": window, "entities": {"light.a": {"rows": [["2026-01-01T01:00:00+00:00", "on"]]}}},
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), (event,))[0].state == (
        "correct" if passed else "incorrect"
    )


def test_history_duration_sums_interleaved_entities_independently() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a", "light.b"],
        input_field="state",
        input_value="on",
        value=14400,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {
            "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T04:00:00+00:00"},
            "entities": {
                "light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T02:00:00+00:00", "off"]]},
                "light.b": {"rows": [["2026-01-01T01:00:00+00:00", "on"], ["2026-01-01T03:00:00+00:00", "off"]]},
            },
        },
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), (event,))[0].state == "correct"


def test_history_duration_fails_when_subject_has_no_rows() -> None:
    claim = AggregateClaim(
        source="history",
        operator="time_in_state",
        subject_ids=["light.a", "light.b"],
        input_field="state",
        input_value="on",
        value=3600,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {
            "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T02:00:00+00:00"},
            "entities": {"light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"]]}, "light.b": {"rows": []}},
        },
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), (event,))[0].state == "incorrect"


def test_history_duration_ignores_unrelated_differing_window_endpoint() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=3600,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T01:00:00+00:00"},
                "entities": {"light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"]]}},
            },
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T02:00:00+00:00"},
                "entities": {"light.other": {"rows": [["2026-01-01T00:00:00+00:00", "off"]]}},
            },
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), events)[0].state == "correct"


def test_history_duration_combines_per_subject_calls_with_different_endpoints() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a", "light.b"],
        input_field="state",
        input_value="on",
        value=21600,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T02:00:00+00:00"},
                "entities": {"light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"]]}},
            },
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T05:00:00+00:00"},
                "entities": {"light.b": {"rows": [["2026-01-01T01:00:00+00:00", "on"]]}},
            },
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), events)[0].state == "correct"


def test_history_duration_cannot_borrow_endpoint_from_another_event() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=3600,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {"window": {}, "entities": {"light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"]]}}},
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T01:00:00+00:00"},
                "entities": {"light.other": {"rows": []}},
            },
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), events)[0].state == "incorrect"


def test_history_duration_unions_same_endpoint_pages() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=7200,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
                "entities": {
                    "light.a": {
                        "rows": [
                            ["2026-01-01T00:00:00+00:00", "on"],
                            ["2026-01-01T01:00:00+00:00", "off"],
                        ]
                    }
                },
            },
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
                "entities": {
                    "light.a": {
                        "rows": [
                            ["2026-01-01T01:00:00+00:00", "off"],
                            ["2026-01-01T02:00:00+00:00", "on"],
                        ]
                    }
                },
            },
            call_index=2,
        ),
    )
    answer = _aggregate_answer(claim)
    assert evaluate_case(_case(expected), answer, events[:1])[0].state == "incorrect"
    assert evaluate_case(_case(expected), answer, events)[0].state == "correct"


def test_history_duration_does_not_merge_same_end_different_start_windows() -> None:
    claim = AggregateClaim(
        source="history",
        operator="duration_seconds",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=7200,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {
                "window": {
                    "start": "2026-01-01T00:00:00+00:00",
                    "end": "2026-01-01T03:00:00+00:00",
                },
                "entities": {
                    "light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T01:00:00+00:00", "off"]]}
                },
            },
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {
                    "start": "2026-01-01T01:00:00+00:00",
                    "end": "2026-01-01T03:00:00+00:00",
                },
                "entities": {
                    "light.a": {"rows": [["2026-01-01T01:00:00+00:00", "off"], ["2026-01-01T02:00:00+00:00", "on"]]}
                },
            },
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), events)[0].state == "incorrect"


def test_history_duration_deduplicates_overlapping_same_endpoint_rows() -> None:
    claim = AggregateClaim(
        source="history",
        operator="time_in_state",
        subject_ids=["light.a"],
        input_field="state",
        input_value="on",
        value=3600,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    events = (
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
                "entities": {
                    "light.a": {
                        "rows": [
                            ["2026-01-01T00:00:00+00:00", "on"],
                            ["2026-01-01T01:00:00+00:00", "off"],
                        ]
                    }
                },
            },
            call_index=1,
        ),
        ToolEvent(
            "get_history",
            {},
            {
                "window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T03:00:00+00:00"},
                "entities": {
                    "light.a": {
                        "rows": [
                            ["2026-01-01T00:00:00+00:00", "on"],
                            ["2026-01-01T02:00:00+00:00", "off"],
                        ]
                    }
                },
            },
            call_index=2,
        ),
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), events)[0].state == "correct"


def test_history_count_respects_selected_state_qualifier() -> None:
    claim = AggregateClaim(
        source="history", operator="count", subject_ids=["light.a"], input_field="state", input_value="off", value=2
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {
            "entities": {
                "light.a": {"rows": [["2026-01-01T00:00:00+00:00", "on"], ["2026-01-01T01:00:00+00:00", "off"]]}
            }
        },
    )
    assert evaluate_case(_case(expected), _aggregate_answer(claim), (event,))[0].state == "incorrect"


@pytest.mark.parametrize(
    ("units", "passed"),
    [
        (("°C", "°C"), True),
        (("°C", "°F"), False),
    ],
)
def test_numeric_state_aggregates_require_each_source_unit(units: tuple[str, str], passed: bool) -> None:
    claim = AggregateClaim(
        source="states",
        operator="mean",
        subject_ids=["sensor.a", "sensor.b"],
        input_field="state",
        input_value="°C",
        value=21,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    output = [
        {"entity_id": "sensor.a", "state": "20", "attributes": {"unit_of_measurement": units[0]}},
        {"entity_id": "sensor.b", "state": "22", "attributes": {"unit_of_measurement": units[1]}},
    ]
    outcome, _, _ = evaluate_case(
        _case(expected),
        _aggregate_answer(claim),
        (ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output}),),
    )
    assert outcome.state == ("correct" if passed else "incorrect")


def test_state_count_rejects_missing_declared_subject_even_when_count_is_zero() -> None:
    claim = AggregateClaim(
        source="states",
        operator="count",
        subject_ids=["sensor.a", "sensor.b"],
        input_field="state",
        input_value="on",
        value=0,
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent(
        "execute_home_code",
        {},
        {
            "execution": {"status": "ok"},
            "output": {"entity_id": "sensor.a", "state": "off"},
        },
    )
    outcome, _, _ = evaluate_case(_case(expected), _aggregate_answer(claim), (event,))
    assert outcome.state == "incorrect"


@pytest.mark.parametrize(
    ("missing_index", "missing_field"),
    [(0, "area_id"), (2, "floor_id")],
    ids=["missing-entity-area", "missing-area-floor"],
)
def test_floor_collection_requires_transitive_entity_area_floor_links(missing_index: int, missing_field: str) -> None:
    claim = CollectionClaim(
        collection="entity_ids", filter_kind="floor", filter_value="floor_upstairs", items=["light.a", "light.b"]
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="exact_items"),))
    output: list[dict[str, object]] = [
        {"entity_id": "light.a", "area_id": "area.a", "state": "on"},
        {"entity_id": "light.b", "area_id": "area.b", "state": "off"},
        {"area_id": "area.a", "floor_id": "floor_upstairs", "name": "A"},
        {"area_id": "area.b", "floor_id": "floor_upstairs", "name": "B"},
    ]
    output[missing_index].pop(missing_field)
    event = ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output})
    assert evaluate_case(_case(expected), ListAnswer(answer="", items=claim.items), (event,))[0].state == "incorrect"


def test_floor_collection_uses_area_floor_join() -> None:
    claim = CollectionClaim(
        collection="entity_ids", filter_kind="floor", filter_value="floor_upstairs", items=["light.a", "light.b"]
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="exact_items"),))
    output = [
        {"entity_id": "light.a", "area_id": "area.a", "state": "on"},
        {"entity_id": "light.b", "area_id": "area.b", "state": "off"},
        {"area_id": "area.a", "floor_id": "floor_upstairs", "name": "A"},
        {"area_id": "area.b", "floor_id": "floor_upstairs", "name": "B"},
    ]
    event = ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output})
    assert evaluate_case(_case(expected), ListAnswer(answer="", items=claim.items), (event,))[0].state == "correct"


@pytest.mark.parametrize(
    ("assertion", "items", "expected_state"),
    [
        pytest.param("exact_items", ["light.a", "light.b"], "correct", id="exact-pass"),
        pytest.param("exact_items", ["light.a", "light.b", "light.c"], "incorrect", id="exact-fail"),
        pytest.param("contains_items", ["light.a", "light.b", "light.c"], "correct", id="contains-pass"),
        pytest.param("contains_items", ["light.a"], "incorrect", id="contains-fail"),
    ],
)
def test_list_items_match_collection_oracle_exact_and_contains(
    assertion: Literal["exact_items", "contains_items"], items: list[str], expected_state: str
) -> None:
    claim = CollectionClaim(
        collection="entity_ids", filter_kind="domain", filter_value="light", items=["light.a", "light.b"]
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion=assertion),))
    output = [{"entity_id": entity_id, "state": "on"} for entity_id in ("light.a", "light.b", "light.c")]
    event = ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": output})

    outcome, _, _ = evaluate_case(_case(expected), ListAnswer(answer="", items=items), (event,))

    assert outcome.state == expected_state


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
        evaluate_case(
            _case(expected),
            _read_answer(claim.entity_id, claim.value, when=claim.when),
            (ToolEvent(tool, {}, output),),
        )[0].state
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
    findings = [
        Finding(subject="device.a", value="Acme"),
        Finding(subject="area.a", value="Kitchen"),
        Finding(subject="repair.a", value=True),
        Finding(subject="note.a", value="Battery low"),
        Finding(subject="light.a", value="device.a"),
        Finding(subject="device.a", value="area.a"),
        Finding(subject="climate.a", value="climate.set_temperature"),
    ]
    assert (
        evaluate_case(
            _case(expected),
            ReadAnswer(answer="", findings=findings),
            (ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": list(output.values())}),),
        )[0].state
        == "correct"
    )


@pytest.mark.parametrize(
    ("claim", "record"),
    [
        pytest.param(
            ValueClaim(subject_kind="device", subject_id="device.a", field="name", value="Entity Name"),
            {"entity_id": "light.a", "device_id": "device.a", "name": "Entity Name", "state": "on"},
            id="entity-join-is-not-device-record",
        ),
        pytest.param(
            ValueClaim(subject_kind="area", subject_id="area.a", field="name", value="Entity Name"),
            {"entity_id": "light.a", "area_id": "area.a", "name": "Entity Name", "state": "on"},
            id="entity-join-is-not-area-record",
        ),
    ],
)
def test_execute_registry_join_keys_do_not_change_record_subject_kind(
    claim: ValueClaim, record: dict[str, object]
) -> None:
    expected = Expected(conclusions=(ExpectedConclusion(claim=claim, assertion="equals"),))
    event = ToolEvent("execute_home_code", {}, {"execution": {"status": "ok"}, "output": record})

    outcome, conclusions, _ = evaluate_case(_case(expected), _read_answer(claim.subject_id, claim.value), (event,))

    assert outcome.state == "incorrect"
    assert conclusions[0].grounding_status == "ungrounded"


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
            ListAnswer(answer="", items=claim.items),
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
    outcome, _, results = evaluate_case(_case(expected), ActionAnswer(answer="", success=True), (), (action,))
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
    outcome, _, results = evaluate_case(_case(expected), ActionAnswer(answer="", success=True), (), (action,))
    assert outcome.state == "incorrect"
    assert any(mismatch.startswith("target:") for mismatch in results[0].mismatches)


@pytest.mark.parametrize("blocked", [False, True], ids=["allowed", "blocked"])
@pytest.mark.parametrize(
    ("expected_targets", "actual_target"),
    [
        pytest.param((), "light.a", id="expected-targetless-actual-targeted"),
        pytest.param(("light.a",), None, id="expected-targeted-actual-targetless"),
    ],
)
def test_action_target_union_rejects_targetless_targeted_conflicts(
    blocked: bool, expected_targets: tuple[str, ...], actual_target: str | None
) -> None:
    expected_action = ExpectedAction("light", "turn_on", expected_targets)
    expected = (
        Expected(blocked_outcome=BlockedOutcome(error_keys=("actions_disabled",), actions=(expected_action,)))
        if blocked
        else Expected(actions=(expected_action,))
    )
    action: dict[str, object] = {
        "domain": "light",
        "service": "turn_on",
        "status": "error" if blocked else "ok",
    }
    if actual_target is not None:
        action["target"] = {"entity_id": actual_target}
    if blocked:
        action["error"] = {"key": "actions_disabled"}

    outcome, _, results = evaluate_case(_case(expected), ActionAnswer(answer="", success=True), (), (action,))

    assert outcome.state == "incorrect"
    assert any(mismatch.startswith("target:") for mismatch in results[0].mismatches)


@pytest.mark.parametrize("blocked", [False, True], ids=["allowed", "blocked"])
def test_action_ledger_rejects_duplicate_targets_within_one_recorded_action(blocked: bool) -> None:
    expected_action = ExpectedAction("light", "turn_on", ("light.a",))
    expected = (
        Expected(blocked_outcome=BlockedOutcome(error_keys=("actions_disabled",), actions=(expected_action,)))
        if blocked
        else Expected(actions=(expected_action,))
    )
    action: dict[str, object] = {
        "domain": "light",
        "service": "turn_on",
        "target": {"entity_id": ["light.a", "light.a"]},
        "status": "error" if blocked else "ok",
    }
    if blocked:
        action["error"] = {"key": "actions_disabled"}

    outcome, _, results = evaluate_case(_case(expected), ActionAnswer(answer="", success=True), (), (action,))

    assert outcome.state == "incorrect"
    assert any(mismatch.startswith("duplicate:") for mismatch in results[0].mismatches)


def test_extra_aggregate_claim_must_match_its_recomputed_value() -> None:
    expected_claim = AggregateClaim(
        source="history", operator="mean", subject_ids=["sensor.a"], input_field="state", input_value="state", value=2
    )
    expected = Expected(conclusions=(ExpectedConclusion(claim=expected_claim, assertion="equals"),))
    event = ToolEvent(
        "get_history",
        {},
        {"entities": {"sensor.a": {"rows": [["2026-01-01T00:00:00+00:00", "1"], ["2026-01-01T01:00:00+00:00", "3"]]}}},
    )
    answer = ReadAnswer(
        answer="",
        findings=[Finding(subject="sensor.a", value=2), Finding(subject="sensor.a", value=99)],
    )
    assert evaluate_case(_case(expected), answer, (event,))[0].state == "incorrect"


@pytest.mark.parametrize("success", [False, True], ids=["diagnostic-failure", "diagnostic-success"])
def test_action_answer_success_is_never_scored(success: bool) -> None:
    expected = Expected(actions=(ExpectedAction("light", "turn_on", ("light.a",)),))
    recorded = ({"domain": "light", "service": "turn_on", "target": {"entity_id": "light.a"}, "status": "ok"},)

    outcome, _, _ = evaluate_case(_case(expected), ActionAnswer(answer="", success=success), (), recorded)

    assert outcome.state == "correct"


def _read_answer(
    subject: str,
    value: str | int | float | bool | None,
    *,
    unit: str | None = None,
    when: str | None = None,
) -> ReadAnswer:
    return ReadAnswer(answer="", findings=[Finding(subject=subject, value=value, unit=unit, when=when)])


def _aggregate_answer(claim: AggregateClaim) -> ReadAnswer:
    return _read_answer(claim.subject_ids[0], claim.value, unit=claim.unit)


def _case(expected: Expected) -> EvalCase:
    return EvalCase("scoring", "unit", "home_default", "request", False, expected, llm_context=CaseContext())
