"""Deterministic typed extraction and scoring for plain-text read answers."""

from datetime import UTC, datetime
import re

from llm_sandbox_evals.schema import AnswerPredicate, AnswerResult

_BOOLEAN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"\d+")
_ENTITY_ID_RE = re.compile(r"\b\w+\.\w+\b")
_NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")
_STATE_RE = re.compile(r"\b(on|off)\b", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b",
    re.IGNORECASE,
)


def score_answer(predicate: AnswerPredicate, answer: str | None) -> AnswerResult:
    """Extract the predicate's typed value from plain text and compare it deterministically."""
    if answer is None or not answer.strip():
        return AnswerResult(False, "answer_unparseable")

    if predicate.kind == "boolean":
        match = _BOOLEAN_RE.search(answer)
        if match is None or predicate.value is None:
            return AnswerResult(False, "answer_unparseable")
        extracted = match.group(1).lower()
        return _answer_result((extracted == "yes") is predicate.value, extracted)

    if predicate.kind == "count":
        match = _COUNT_RE.search(answer)
        if match is None or predicate.count is None:
            return AnswerResult(False, "answer_unparseable")
        extracted = match.group(0)
        return _answer_result(int(extracted) == predicate.count, extracted)

    if predicate.kind == "entity_set":
        extracted_ids = tuple(_ENTITY_ID_RE.findall(answer))
        if not extracted_ids:
            return AnswerResult(False, "answer_unparseable")
        extracted = ",".join(extracted_ids)
        return _answer_result(set(extracted_ids) == set(predicate.entity_ids), extracted)

    if predicate.kind == "scalar":
        match = _NUMBER_RE.search(answer)
        if match is None or predicate.scalar_value is None or predicate.tolerance is None:
            return AnswerResult(False, "answer_unparseable")
        extracted = match.group(0)
        return _answer_result(abs(float(extracted) - predicate.scalar_value) <= predicate.tolerance, extracted)

    if predicate.kind == "state":
        match = _STATE_RE.search(answer)
        if match is None or predicate.state is None:
            return AnswerResult(False, "answer_unparseable")
        extracted = match.group(1).lower()
        return _answer_result(extracted == predicate.state.lower(), extracted)

    match = _TIMESTAMP_RE.search(answer)
    if match is None or predicate.start is None or predicate.end is None:
        return AnswerResult(False, "answer_unparseable")
    extracted = match.group(0)
    timestamp = _parse_timestamp(extracted)
    start = _parse_timestamp(predicate.start)
    end = _parse_timestamp(predicate.end)
    if timestamp is None or start is None or end is None:
        return AnswerResult(False, "answer_unparseable")
    return _answer_result(start <= timestamp <= end, extracted)


def _answer_result(passed: bool, extracted: str) -> AnswerResult:
    """Build the stable result reason for one successful extraction."""
    return AnswerResult(passed, "answer_correct" if passed else "answer_incorrect", extracted)


def _parse_timestamp(value: str) -> datetime | None:
    """Parse one timezone-aware ISO/RFC3339 timestamp and normalize it to UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)
