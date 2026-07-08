"""Pure recovery-suggestion engine for frozen Home Assistant snapshots."""

from ...snapshot.models import HomeSnapshot
from ..resolution import resolve_target_entity
from .context import FailureContext, Intent
from .payload import Candidate, Guidance
from .policy import Confidence, decide, reason_next_step
from .scoring import Match, enumerate_for_context, ranked_candidates, score
from .sources import CandidateDict, entity_candidates

__all__ = (
    "Candidate",
    "Confidence",
    "FailureContext",
    "Guidance",
    "Intent",
    "advise",
    "score",
)


def advise(snapshot: HomeSnapshot, ctx: FailureContext) -> Guidance:
    """Return recovery guidance for a failed literal using only a frozen snapshot."""
    # Exact/unique entity resolution is the trusted fast path already used by service-call recovery.
    if (
        ctx.intent in {Intent.READ_STATE, Intent.RESOLVE_SELECTOR, Intent.QUERY_HISTORY, Intent.CAPTURE_IMAGE}
        and ctx.domain
    ):
        resolution = resolve_target_entity(snapshot, ctx.requested, ctx.domain)
        if resolution.resolved is not None:
            candidate = _candidate_for_entity(snapshot, resolution.resolved)
            reason = f"`{ctx.requested}` resolves to visible entity `{resolution.resolved}`."
            return Guidance(
                confidence=Confidence.EXACT,
                candidates=[
                    _payload_candidate(
                        candidate,
                        Match(
                            exact=1,
                            token_overlap=0.0,
                            capability=0,
                            area_floor=0,
                            service_support=0,
                            field_overlap=0,
                            tiebreak=resolution.resolved,
                            label="id",
                        ),
                    )
                ],
                reason=reason,
                next_step="",
            )
        if resolution.candidates:
            candidates = tuple(
                _candidate_for_entity(snapshot, candidate.entity_id) for candidate in resolution.candidates
            )
        else:
            candidates = _cross_kind_candidates(snapshot, ctx) or enumerate_for_context(snapshot, ctx)
    else:
        candidates = enumerate_for_context(snapshot, ctx)

    cross_kind = ""
    # RESOLVE_SELECTOR can recover area->floor when an area has no matching domain but its floor does.
    if ctx.intent == Intent.RESOLVE_SELECTOR and ctx.requested in snapshot.areas:
        scoped = _area_or_floor_domain_candidates(snapshot, ctx)
        if scoped:
            candidates, cross_kind = scoped

    ranked, overflow = ranked_candidates(snapshot, ctx, candidates)
    confidence = decide(ranked, ctx)
    top = ranked[0][0] if ranked else None
    reason, next_step = reason_next_step(confidence, top, ctx, shown=len(ranked), overflow=overflow)
    return Guidance(
        confidence=confidence,
        candidates=[_payload_candidate(candidate, match) for candidate, match in ranked],
        reason=reason,
        next_step=next_step,
        cross_kind=cross_kind,
    )


def _candidate_for_entity(snapshot: HomeSnapshot, entity_id: str) -> CandidateDict:
    """Return the rich entity candidate for an entity id known to be visible."""
    for candidate in entity_candidates(snapshot, snapshot.states[entity_id].domain):
        if candidate["id"] == entity_id:
            return candidate
    raise KeyError(entity_id)


def _payload_candidate(candidate: CandidateDict, match: Match) -> Candidate:
    """Convert an internal candidate mapping into the public payload dataclass."""
    candidate_id = str(candidate.get("id", ""))
    name = str(candidate.get("name", ""))
    detail_parts = [
        str(candidate.get("area_name", "")),
        str(candidate.get("floor_name", "")),
        str(candidate.get("device_class", "")),
        str(candidate.get("unit", "")),
    ]
    detail = ", ".join(part for part in detail_parts if part)
    return Candidate(id=candidate_id, name=name, match=match.label, detail=detail)


def _cross_kind_candidates(snapshot: HomeSnapshot, ctx: FailureContext) -> tuple[CandidateDict, ...]:
    """Return floor-scoped candidates when an area selector implies its floor."""
    scoped = _area_or_floor_domain_candidates(snapshot, ctx)
    return scoped[0] if scoped else ()


def _area_or_floor_domain_candidates(
    snapshot: HomeSnapshot, ctx: FailureContext
) -> tuple[tuple[CandidateDict, ...], str] | None:
    """Scope selector candidates to a containing floor when an area has no domain match."""
    area = snapshot.areas.get(ctx.requested)
    if area is None or not area.floor_id:
        return None
    area_entities = [
        entity_id
        for entity_id in snapshot.indexes.entity_ids_by_area_id.get(area.area_id, ())
        if _is_domain(snapshot, entity_id, ctx.domain)
    ]
    floor_area_ids = snapshot.indexes.area_ids_by_floor_id.get(area.floor_id, ())
    floor_entities = [
        entity_id
        for area_id in floor_area_ids
        for entity_id in snapshot.indexes.entity_ids_by_area_id.get(area_id, ())
        if _is_domain(snapshot, entity_id, ctx.domain)
    ]
    # Cross-kind is only safe when the requested area has no domain match and the floor has scoped matches.
    if area_entities or not floor_entities:
        return None
    floor = snapshot.floors.get(area.floor_id)
    candidates = tuple(_candidate_for_entity(snapshot, entity_id) for entity_id in sorted(floor_entities))
    floor_hint = floor.floor_id if floor is not None else area.floor_id
    return candidates, floor_hint


def _is_domain(snapshot: HomeSnapshot, entity_id: str, domain: str) -> bool:
    """Return whether a visible entity belongs to the requested domain."""
    state = snapshot.states.get(entity_id)
    return state is not None and (not domain or state.domain == domain)
