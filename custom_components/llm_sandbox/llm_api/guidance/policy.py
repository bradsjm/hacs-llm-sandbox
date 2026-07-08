"""Confidence policy and wording for recovery guidance."""

from collections.abc import Sequence
from enum import StrEnum

from .context import FailureContext

# Two textual matches within this overlap ratio of each other are treated as
# ambiguous (never imperative, never written to memory).
_AMBIGUITY_OVERLAP_RATIO = 0.8


class Confidence(StrEnum):
    """Guidance confidence levels that gate imperative wording and memory writes."""

    EXACT = "exact"
    HIGH = "high"
    AMBIGUOUS = "ambiguous"
    LISTING = "listing"
    NONE = "none"


def decide(ranked: Sequence[tuple[dict[str, object], object]], _ctx: FailureContext) -> Confidence:
    """Decide confidence from ranked candidates without consulting live state."""
    from .scoring import Match

    if not ranked:
        return Confidence.NONE
    top = ranked[0][1]
    if not isinstance(top, Match):
        return Confidence.NONE
    # Exact confidence permits imperative wording because id/name/alias equality is unambiguous.
    if top.exact and (len(ranked) == 1 or not isinstance(ranked[1][1], Match) or not ranked[1][1].exact):
        return Confidence.EXACT
    if len(ranked) > 1 and isinstance(ranked[1][1], Match):
        second = ranked[1][1]
        # Comparable top scores are ambiguous: never say "Use X" or write resolution memory here.
        if top.semantic_key() == second.semantic_key() or (
            top.token_overlap > 0 and second.token_overlap >= top.token_overlap * _AMBIGUITY_OVERLAP_RATIO
        ):
            return Confidence.AMBIGUOUS
    # High confidence requires a clear semantic signal, not merely a domain/service fallback or memory hit.
    if top.strong_non_exact and (
        len(ranked) == 1 or not isinstance(ranked[1][1], Match) or top.semantic_key() > ranked[1][1].semantic_key()
    ):
        return Confidence.HIGH
    # Listing is intentionally non-imperative for weak candidates such as domain-only or memory-only matches.
    return Confidence.LISTING


def reason_next_step(
    confidence: Confidence,
    top_candidate: dict[str, object] | None,
    ctx: FailureContext,
    *,
    shown: int = 0,
    overflow: int = 0,
) -> tuple[str, str]:
    """Return confidence-gated human reason and mandatory next step.

    ``shown``/``overflow`` are the actual bounded candidate count and the number
    dropped beyond the bound, so the overflow notice reports honest totals
    instead of reconstructing them from a magic literal.
    """
    requested = ctx.requested
    top_id = str(top_candidate.get("id", "")) if top_candidate is not None else ""
    if confidence == Confidence.EXACT:
        return f"`{requested}` resolved exactly to `{top_id}`.", ""
    if confidence == Confidence.HIGH:
        return (
            f"`{requested}` does not exactly exist, but `{top_id}` is the strongest visible match.",
            f"Use `{top_id}` and retry.",
        )
    if confidence == Confidence.AMBIGUOUS:
        return (
            f"`{requested}` matches multiple visible candidates with comparable scores.",
            "Choose one listed candidate explicitly and retry.",
        )
    if confidence == Confidence.LISTING:
        suffix = f" Showing the first {shown} of {shown + overflow}." if overflow else ""
        return (
            f"`{requested}` was not found; visible candidates exist but none is a strong match.{suffix}",
            "Pick the correct listed id or inspect the visible Home Assistant inventory before retrying.",
        )
    return f"`{requested}` does not exist in the visible snapshot.", "Use a visible Home Assistant id and retry."


MEMORY_WRITE_ALLOWED: dict[Confidence, bool] = {
    Confidence.EXACT: True,
    Confidence.HIGH: True,
    Confidence.AMBIGUOUS: False,
    Confidence.LISTING: False,
    Confidence.NONE: False,
}
