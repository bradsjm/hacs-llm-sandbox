"""JSON-serializable guidance payload contracts."""

from dataclasses import dataclass

from .policy import Confidence


@dataclass(frozen=True, slots=True)
class Candidate:
    """One bounded recovery candidate shown to the LLM."""

    id: str
    name: str
    match: str
    detail: str


@dataclass(frozen=True, slots=True)
class Guidance:
    """Structured recovery guidance with confidence-gated wording."""

    confidence: Confidence
    candidates: list[Candidate]
    reason: str
    next_step: str
    cross_kind: str = ""

    def to_payload(self) -> dict[str, object]:
        """Return the documented JSON-serializable guidance shape."""
        return {
            "confidence": self.confidence.value,
            "candidates": [
                {"id": candidate.id, "name": candidate.name, "match": candidate.match, "detail": candidate.detail}
                for candidate in self.candidates
            ],
            "reason": self.reason,
            "next_step": self.next_step,
            "cross_kind": self.cross_kind,
        }


EMPTY = Guidance(
    confidence=Confidence.NONE,
    candidates=[],
    reason="No guidance is available.",
    next_step="Check the requested literal and retry with a visible Home Assistant id.",
)


def none(reason: str, next_step: str) -> Guidance:
    """Build a confidence=none outcome without candidates."""
    return Guidance(confidence=Confidence.NONE, candidates=[], reason=reason, next_step=next_step)
