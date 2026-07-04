"""Pure snapshot-aware resolution for service-call targets.

Exact entity-id matches win. Otherwise, target resolution uses deterministic
same-domain token containment (one token set is a subset of the other) over
snapshot-derived ``object_id`` and ``name`` tokens. A unique match
auto-resolves; ambiguous matches return candidates so the caller can
self-describe available targets. Optional conversation memory is advisory only:
it can reorder or break ties only for entity ids still present in the fresh
snapshot candidate set. This module never touches live Home Assistant objects
and performs no I/O.
"""

import re
from dataclasses import dataclass

from ..snapshot.models import HomeSnapshot
from .resolution_memory import ResolutionMemory
from .target_matching import entities_for_service

_DISCOVERY_LIMIT = 8


def bounded_strings(values: list[str], limit: int = _DISCOVERY_LIMIT) -> list[str]:
    """Bound deterministic repair lists to the discovery limit plus overflow marker."""
    if len(values) > limit:
        return [*values[: limit - 1], "..."]
    return values


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
    *,
    memory: ResolutionMemory | None = None,
) -> TargetResolution:
    """Resolve by exact id or same-domain token containment over visible states."""
    # Exact visible entity ids are authoritative and bypass fuzzy matching.
    if requested_entity_id in snapshot.states:
        return TargetResolution(resolved=requested_entity_id)

    query_tokens = _tokens(requested_entity_id)
    if not query_tokens:
        return TargetResolution(candidates=())

    matches: list[CandidateTarget] = []

    for state in snapshot.states.values():
        # Alternatives are intentionally constrained to the service domain.
        if state.domain != domain:
            continue

        candidate_tokens = _tokens(state.object_id)
        if state.name is not None:
            candidate_tokens |= _tokens(state.name)
        if not candidate_tokens:
            continue

        if query_tokens <= candidate_tokens or candidate_tokens <= query_tokens:
            matches.append(
                CandidateTarget(
                    entity_id=state.entity_id,
                    name=state.name,
                    object_id=state.object_id,
                )
            )

    # No containment match means the caller should treat the request as not found.
    if not matches:
        return TargetResolution(candidates=())

    candidates = tuple(
        sorted(
            matches,
            key=lambda candidate: candidate.entity_id,
        )
    )

    remembered = memory.lookup(requested_entity_id) if memory is not None else None
    if remembered is not None and remembered in {candidate.entity_id for candidate in candidates}:
        return TargetResolution(resolved=remembered)

    # A unique containment match is safe to auto-resolve deterministically.
    if len(candidates) == 1:
        return TargetResolution(resolved=candidates[0].entity_id)

    # Multiple containment matches are ambiguous, so surface deterministic hints.
    return TargetResolution(candidates=candidates)


def candidates_for_domain(
    snapshot: HomeSnapshot,
    domain: str,
    limit: int = _DISCOVERY_LIMIT,
    *,
    memory: ResolutionMemory | None = None,
) -> tuple[CandidateTarget, ...]:
    """Return visible candidate targets for a domain, ordered for discovery hints."""
    remembered_rank = _remembered_rank(snapshot, domain, memory)
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
        key=lambda candidate: (
            remembered_rank.get(candidate.entity_id, len(remembered_rank)),
            candidate.object_id,
            candidate.entity_id,
        ),
    )
    return tuple(candidates[:limit])


def available_hint(snapshot: HomeSnapshot, domain: str) -> str:
    """Return a short hint describing visible entity ids in a domain."""
    candidates = candidates_for_domain(snapshot, domain, limit=_DISCOVERY_LIMIT + 1)
    ids = [candidate.entity_id for candidate in candidates]
    if ids:
        suffix = " ..." if len(ids) > _DISCOVERY_LIMIT else ""
        return f"Visible entities in the '{domain}' domain: {', '.join(ids[:_DISCOVERY_LIMIT])}{suffix}"

    return f"No visible entities in the '{domain}' domain."


def rank_candidates_for_service(
    snapshot: HomeSnapshot,
    candidates: tuple[CandidateTarget, ...],
    domain: str,
    service: str,
    *,
    requested: str | None = None,
    memory: ResolutionMemory | None = None,
) -> tuple[CandidateTarget, ...]:
    """Order candidates so entities the service targets sort first.

    Target-aware ranking for fix lists: when a service declares target metadata,
    entities it accepts rank ahead of unrelated same-domain entities so the
    surfaced candidates reflect what the service can actually act on. Returns the
    input unchanged (already deterministic) when the service has no target
    metadata, preserving HA as the final arbiter.
    """
    matched = set(entities_for_service(snapshot, domain, service))
    remembered = memory.lookup(requested) if memory is not None and requested is not None else None
    if remembered is not None and remembered not in {candidate.entity_id for candidate in candidates}:
        remembered = None
    if not matched and remembered is None:
        return candidates
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.entity_id != remembered,
                candidate.entity_id not in matched,
                candidate.entity_id,
            ),
        )
    )


def _tokens(text: str) -> frozenset[str]:
    """Return lowercase alphanumeric tokens from text."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _remembered_rank(
    snapshot: HomeSnapshot,
    domain: str,
    memory: ResolutionMemory | None,
) -> dict[str, int]:
    """Return fresh-snapshot-valid remembered entity ordering for ``domain``."""
    if memory is None:
        return {}
    ranked: dict[str, int] = {}
    for entity_id in memory.remembered_entity_ids():
        state = snapshot.states.get(entity_id)
        if state is None or state.domain != domain or entity_id in ranked:
            continue
        ranked[entity_id] = len(ranked)
    return ranked
