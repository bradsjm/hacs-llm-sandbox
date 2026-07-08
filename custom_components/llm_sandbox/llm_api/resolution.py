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


def _tokens(text: str) -> frozenset[str]:
    """Return lowercase alphanumeric tokens from text."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))
