"""Pure snapshot-aware resolution for service-call targets.

Exact entity-id matches win. Otherwise, target resolution uses a deterministic
same-domain fuzzy match over snapshot-derived ``object_id`` and ``name`` token
overlap. A unique match auto-resolves; ambiguous matches return candidates so
the caller can self-describe available targets. This module never touches live
Home Assistant objects and performs no I/O.
"""

import re
from dataclasses import dataclass

from ..snapshot.models import HomeSnapshot


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """A single resolved service-call target entity id."""

    entity_id: str


@dataclass(frozen=True, slots=True)
class CandidateTarget:
    """A visible same-domain entity offered when the requested target is ambiguous."""

    entity_id: str
    name: str | None
    object_id: str


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """Outcome of resolving a requested target against the snapshot.

    ``resolved`` is set when exactly one entity matches. ``candidates`` is set
    when the request is ambiguous (multiple matches) or the exact id is absent
    but same-domain alternatives exist. Both are None only when nothing in the
    snapshot matches at all.
    """

    resolved: str | None = None
    candidates: tuple[CandidateTarget, ...] = ()

    @property
    def is_resolved(self) -> bool:
        """Return whether this resolution identifies exactly one target."""
        return self.resolved is not None


def resolve_target_entity(
    snapshot: HomeSnapshot,
    requested_entity_id: str,
    domain: str,
) -> TargetResolution:
    """Resolve a requested target entity id or name against visible snapshot states."""
    # Exact visible entity ids are authoritative and bypass fuzzy matching.
    if requested_entity_id in snapshot.states:
        return TargetResolution(resolved=requested_entity_id)

    query_tokens = _tokens(requested_entity_id)
    scored_candidates: list[tuple[int, CandidateTarget]] = []

    for state in snapshot.states.values():
        # Fuzzy alternatives are intentionally constrained to the service domain.
        if state.domain != domain:
            continue

        candidate_tokens = _tokens(state.object_id)
        if state.name is not None:
            candidate_tokens |= _tokens(state.name)

        score = len(candidate_tokens & query_tokens)
        if score >= 1:
            scored_candidates.append(
                (
                    score,
                    CandidateTarget(
                        entity_id=state.entity_id,
                        name=state.name,
                        object_id=state.object_id,
                    ),
                )
            )

    # No shared tokens means the caller should treat the request as not found.
    if not scored_candidates:
        return TargetResolution(candidates=())

    candidates = tuple(
        sorted(
            (candidate for _score, candidate in scored_candidates),
            key=lambda candidate: candidate.entity_id,
        )
    )

    # A unique fuzzy match is safe to auto-resolve deterministically.
    if len(candidates) == 1:
        return TargetResolution(resolved=candidates[0].entity_id)

    # Multiple token-overlap matches are ambiguous, so surface deterministic hints.
    return TargetResolution(candidates=candidates)


def candidates_for_domain(
    snapshot: HomeSnapshot,
    domain: str,
    limit: int = 10,
) -> tuple[CandidateTarget, ...]:
    """Return visible candidate targets for a domain, ordered for discovery hints."""
    candidates = sorted(
        (
            CandidateTarget(
                entity_id=state.entity_id,
                name=state.name,
                object_id=state.object_id,
            )
            for state in snapshot.states.values()
            if state.domain == domain
        ),
        key=lambda candidate: (candidate.object_id, candidate.entity_id),
    )
    return tuple(candidates[:limit])


def available_hint(snapshot: HomeSnapshot, domain: str) -> str:
    """Return a short hint describing visible entity ids in a domain."""
    candidates = candidates_for_domain(snapshot, domain)
    ids = [candidate.entity_id for candidate in candidates]
    if ids:
        suffix = " ..." if len(ids) > 8 else ""
        return f"Visible entities in the '{domain}' domain: {', '.join(ids[:8])}{suffix}"

    return f"No visible entities in the '{domain}' domain."


def _tokens(text: str) -> frozenset[str]:
    """Return lowercase alphanumeric tokens from text."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))
