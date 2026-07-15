from llm_sandbox_evals.schema import AnswerPredicate, EvalCase, RequestVariant
from llm_sandbox_evals.scoring.evaluate import evaluate_case
from llm_sandbox_evals.scoring.read_answers import score_answer
import pytest


@pytest.mark.parametrize(
    ("predicate", "answer", "extracted"),
    [
        pytest.param(AnswerPredicate("boolean", value=True), "Yes, it has.", "yes", id="boolean"),
        pytest.param(AnswerPredicate("count", count=2), "There are 2 lights.", "2", id="count"),
        pytest.param(
            AnswerPredicate("entity_set", entity_ids=("light.a", "switch.b")),
            "switch.b and light.a",
            "switch.b,light.a",
            id="entity-set",
        ),
        pytest.param(
            AnswerPredicate("scalar", scalar_value=21.5, unit="°C", tolerance=0.2),
            "The temperature is 21.7 °C.",
            "21.7",
            id="scalar-tolerance-boundary",
        ),
        pytest.param(AnswerPredicate("state", state="on"), "It is ON.", "on", id="state"),
        pytest.param(
            AnswerPredicate(
                "time_range",
                start="2026-07-14T10:00:00Z",
                end="2026-07-14T11:00:00Z",
            ),
            "Last changed at 2026-07-14T12:00:00+01:00.",
            "2026-07-14T12:00:00+01:00",
            id="time-range-utc-boundary",
        ),
    ],
)
def test_typed_answers_extract_and_match(predicate: AnswerPredicate, answer: str, extracted: str) -> None:
    result = score_answer(predicate, answer)

    assert result.passed is True
    assert result.reason == "answer_correct"
    assert result.extracted_value == extracted


@pytest.mark.parametrize(
    ("predicate", "answer"),
    [
        pytest.param(AnswerPredicate("boolean", value=True), "No.", id="boolean"),
        pytest.param(AnswerPredicate("count", count=2), "There are 3.", id="count"),
        pytest.param(
            AnswerPredicate("entity_set", entity_ids=("light.a",)),
            "light.a and light.b",
            id="entity-set-extra",
        ),
        pytest.param(
            AnswerPredicate("scalar", scalar_value=21.5, tolerance=0.1),
            "21.61",
            id="scalar-outside-tolerance",
        ),
        pytest.param(AnswerPredicate("state", state="off"), "It is on.", id="state"),
        pytest.param(
            AnswerPredicate(
                "time_range",
                start="2026-07-14T10:00:00Z",
                end="2026-07-14T11:00:00Z",
            ),
            "2026-07-14T11:00:01Z",
            id="time-range-outside",
        ),
    ],
)
def test_typed_answers_reject_extracted_wrong_values(predicate: AnswerPredicate, answer: str) -> None:
    result = score_answer(predicate, answer)

    assert result.passed is False
    assert result.reason == "answer_incorrect"
    assert result.extracted_value is not None


@pytest.mark.parametrize(
    ("predicate", "answer"),
    [
        pytest.param(AnswerPredicate("boolean", value=True), "Yesterday was sunny.", id="substring"),
        pytest.param(AnswerPredicate("count", count=1), "One light.", id="count-word"),
        pytest.param(AnswerPredicate("entity_set", entity_ids=("light.a",)), "the light", id="entity-set"),
        pytest.param(AnswerPredicate("scalar", scalar_value=1.0, tolerance=0.0), "unknown", id="scalar"),
        pytest.param(AnswerPredicate("state", state="on"), "Only one light.", id="state-substring"),
        pytest.param(
            AnswerPredicate("time_range", start="2026-07-14T10:00:00Z", end="2026-07-14T11:00:00Z"),
            "2026-07-14 10:30:00",
            id="timestamp-without-zone",
        ),
        pytest.param(AnswerPredicate("count", count=0), None, id="none"),
        pytest.param(AnswerPredicate("count", count=0), "  ", id="empty"),
    ],
)
def test_typed_answers_reject_unparseable_text(predicate: AnswerPredicate, answer: str | None) -> None:
    result = score_answer(predicate, answer)

    assert result.passed is False
    assert result.reason == "answer_unparseable"
    assert result.extracted_value is None


def test_answer_oracle_is_primary_and_retains_effect_diagnostics() -> None:
    predicate = AnswerPredicate("count", count=1)
    case = EvalCase(
        "answer-case",
        "home_minimal",
        "read_answer",
        (RequestVariant("canonical", "How many?"),),
        (),
        oracle="answer",
        expected_answer=predicate,
    )

    evaluation = evaluate_case(
        case,
        (),
        overlay_seeds=(),
        invoker_calls=(),
        answer="There is 1 light.",
    )

    assert evaluation.outcome.scoring_mode == "answer"
    assert evaluation.outcome.state == "correct"
    assert evaluation.answer_result is not None
    assert evaluation.tool_call_result is None
    assert evaluation.action_result.passed is True
    assert evaluation.end_state_result.status == "not_authored"


@pytest.mark.parametrize(
    ("predicate", "answer", "extracted"),
    [
        pytest.param(
            AnswerPredicate("count", count=1),
            "Of the 2 lights in the Utility Room, 1 is on.",
            "1",
            id="count-last-match-after-context",
        ),
        pytest.param(
            AnswerPredicate("state", state="on"),
            "It's not off — it's on.",
            "on",
            id="state-last-match-after-negation",
        ),
        pytest.param(
            AnswerPredicate("boolean", value=True),
            "No, that's not right. Yes, it is on.",
            "yes",
            id="boolean-last-match-after-correction",
        ),
    ],
)
def test_last_match_extracts_answer_from_explanatory_prose(
    predicate: AnswerPredicate, answer: str, extracted: str
) -> None:
    """Last-match extraction handles context-before-answer prose without substring scoring."""
    result = score_answer(predicate, answer)

    assert result.passed is True
    assert result.reason == "answer_correct"
    assert result.extracted_value == extracted
