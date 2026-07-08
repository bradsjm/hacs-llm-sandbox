"""Deterministic candidate scoring for recovery guidance.

All ranking signals are derived from snapshot records and morphology, never from
hand-maintained allowlists. Abbreviation near-misses (``temp``/``temperature``,
``hum``/``humidity``) are handled by shared-prefix matching, and device-class /
unit vocabulary is read from each candidate's own snapshot attributes.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ...snapshot.models import HomeSnapshot
from ..resolution import bounded_strings
from ..resolution_memory import ResolutionMemory
from ..target_matching import service_accepts_domain
from .context import FailureContext, Intent
from .sources import CandidateDict, entity_candidates, service_candidates

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Minimum shared prefix length for abbreviation-style token matching (temp/temperature).
_PREFIX_MIN = 3
# Overlap fraction at which a textual match counts as a strong semantic signal.
_MIN_STRONG_OVERLAP = 0.5
# Capability weights: device_class is a stronger disambiguator than a unit hint.
_CAPABILITY_DEVICE_CLASS_WEIGHT = 2
_CAPABILITY_UNIT_WEIGHT = 1


@dataclass(frozen=True, slots=True)
class Match:
    """Per-signal score tuple plus a human label for the strongest signal."""

    exact: int
    token_overlap: float
    capability: int
    area_floor: int
    service_support: int
    memory: int
    field_overlap: int
    tiebreak: str
    label: str

    def key(self) -> tuple[int, float, int, int, int, int, int, str]:
        """Return a sortable key where higher signal values win and ids sort deterministically."""
        return (
            self.exact,
            self.token_overlap,
            self.capability,
            self.area_floor,
            self.service_support,
            self.memory,
            self.field_overlap,
            _reverse_tiebreak(self.tiebreak),
        )

    @property
    def strong_non_exact(self) -> bool:
        """Return whether the match has a semantic signal beyond exact equality."""
        return (
            self.token_overlap >= _MIN_STRONG_OVERLAP
            or self.capability > 0
            or self.area_floor > 0
            or self.field_overlap > 0
        )


def score(
    requested: str,
    candidate: Mapping[str, object],
    ctx: FailureContext,
    *,
    memory: ResolutionMemory | None,
    snapshot: HomeSnapshot | None = None,
) -> Match:
    """Score one candidate against the requested literal and failure context."""
    candidate_id = _text(candidate, "id") or _text(candidate, "entity_id") or _text(candidate, "service")
    candidate_name = _text(candidate, "name")
    aliases = _aliases(candidate)
    requested_normalized = requested.strip().lower()
    exact = 0
    label = "id"
    # Exact id/name/alias equality is authoritative and sits above every fuzzy signal.
    if requested_normalized and requested_normalized in {
        candidate_id.lower(),
        candidate_name.lower(),
        *(alias.lower() for alias in aliases),
    }:
        exact = 1
        label = "id" if requested_normalized == candidate_id.lower() else "name"

    requested_tokens = _tokens(requested)
    # Entity ids include a domain prefix; the prefix alone must not create a fuzzy match.
    if candidate.get("kind") == "entity" and ctx.domain:
        requested_tokens = frozenset(token for token in requested_tokens if token != ctx.domain)
    candidate_tokens = _candidate_tokens(candidate)
    # Shared-prefix matching catches object_id/name/alias near misses (temp/temperature)
    # without a hand-maintained abbreviation table.
    matched = _matched_tokens(requested_tokens, candidate_tokens)
    overlap = (len(matched) / len(requested_tokens)) if requested_tokens else 0.0
    if exact == 0 and overlap > 0.0:
        label = "name"

    capability = _capability_signal(requested_tokens, candidate)
    # Device-class/unit vocabulary comes from the candidate's own snapshot attributes.
    if exact == 0 and capability > 0:
        device_class = _text(candidate, "device_class")
        label = f"device_class: {device_class}" if device_class else f"unit: {_text(candidate, 'unit')}"

    area_floor = _area_floor_signal(requested_tokens, candidate)
    # Area/floor tokens explain why a visible entity is relevant to a selector or natural phrase.
    if exact == 0 and capability == 0 and area_floor > 0:
        label = f"area: {_text(candidate, 'area_name') or _text(candidate, 'floor_name')}"

    service_support = _service_support_signal(candidate, ctx, snapshot)
    # Service-support is a capability hint, not enough by itself for imperative guidance.
    if exact == 0 and capability == 0 and area_floor == 0 and overlap == 0.0 and service_support > 0:
        label = f"supports: {ctx.domain}.{ctx.service}"

    remembered = memory.lookup(requested) if memory is not None else None
    memory_signal = int(bool(remembered and remembered == candidate_id))
    # Memory can boost a still-visible candidate but cannot create high-confidence guidance alone.
    if exact == 0 and not label.startswith(("name", "device_class", "unit", "area", "supports")) and memory_signal:
        label = "memory"

    field_overlap = _field_overlap_signal(candidate, ctx)
    # Service-data field overlap ranks service name near-misses by the fields the LLM attempted to pass.
    if exact == 0 and field_overlap > 0 and candidate.get("kind") == "service":
        label = "service fields"

    return Match(
        exact=exact,
        token_overlap=overlap,
        capability=capability,
        area_floor=area_floor,
        service_support=service_support,
        memory=memory_signal,
        field_overlap=field_overlap,
        tiebreak=candidate_id,
        label=label,
    )


def ranked_candidates(
    snapshot: HomeSnapshot,
    ctx: FailureContext,
    candidates: tuple[CandidateDict, ...],
    *,
    memory: ResolutionMemory | None,
) -> tuple[list[tuple[CandidateDict, Match]], int]:
    """Return up to the discovery limit of ranked candidates plus the overflow count.

    Overflow (not the bound itself) is returned so callers report honest totals
    instead of reconstructing them from a magic literal.
    """
    scored = [
        (candidate, score(ctx.requested, candidate, ctx, memory=memory, snapshot=snapshot)) for candidate in candidates
    ]
    scored.sort(key=lambda item: item[1].key(), reverse=True)
    bounded_ids = bounded_strings([str(item[0].get("id", "")) for item in scored])
    limit = len(bounded_ids) - (1 if bounded_ids and bounded_ids[-1] == "..." else 0)
    overflow = max(0, len(scored) - limit)
    return scored[:limit], overflow


def enumerate_for_context(snapshot: HomeSnapshot, ctx: FailureContext) -> tuple[CandidateDict, ...]:
    """Enumerate the default candidate set for an intent."""
    from .sources import (
        area_candidates,
        code_attribute_candidates,
        code_global_candidates,
        device_candidates,
        floor_candidates,
        label_candidates,
        sql_column_candidates,
        sql_table_candidates,
    )

    # Entity-reading/history/image failures suggest visible entities from the requested domain.
    if ctx.intent in {Intent.READ_STATE, Intent.QUERY_HISTORY, Intent.CAPTURE_IMAGE}:
        return entity_candidates(snapshot, ctx.domain)
    # Service-name failures suggest services, while selector failures suggest target entities.
    if ctx.intent == Intent.CALL_SERVICE:
        return service_candidates(snapshot, ctx.domain)
    if ctx.intent == Intent.RESOLVE_SELECTOR:
        return entity_candidates(snapshot, ctx.domain)
    if ctx.intent == Intent.SQL_TABLE:
        return sql_table_candidates()
    if ctx.intent == Intent.SQL_COLUMN:
        return sql_column_candidates(ctx.table_name)
    if ctx.intent == Intent.CODE_NAME:
        return code_global_candidates()
    if ctx.intent == Intent.CODE_ATTRIBUTE:
        return code_attribute_candidates(ctx.available_attributes)
    return (
        *area_candidates(snapshot),
        *floor_candidates(snapshot),
        *label_candidates(snapshot),
        *device_candidates(snapshot),
    )


def _tokens(text: str) -> frozenset[str]:
    """Return lowercase alphanumeric tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _matched_tokens(requested: frozenset[str], pool: frozenset[str]) -> frozenset[str]:
    """Return requested tokens that match a pool token by equality or abbreviation.

    A token matches when it equals a pool token or one is a prefix of the other
    with the shorter at least ``_PREFIX_MIN`` characters, so ``temp`` matches
    ``temperature`` and ``hum`` matches ``humidity`` without a curated table. The
    prefix-of relation excludes mere typos (``statez``/``states`` share a prefix
    but neither contains the other), keeping abbreviation matching precise.
    """
    if not requested or not pool:
        return frozenset()
    matched: set[str] = set()
    for req in requested:
        for cand in pool:
            if req == cand or _is_abbreviation(req, cand):
                matched.add(req)
                break
    return frozenset(matched)


def _is_abbreviation(a: str, b: str) -> bool:
    """Return whether one token is a prefix of the other (a true abbreviation)."""
    shorter = _PREFIX_MIN
    return len(a) >= shorter and len(b) >= shorter and (a.startswith(b) or b.startswith(a))


def _candidate_tokens(candidate: Mapping[str, object]) -> frozenset[str]:
    """Tokenize object_id/name/aliases/service fields for fuzzy matching."""
    parts = [
        _text(candidate, "object_id"),
        _text(candidate, "name"),
        _text(candidate, "id"),
        _text(candidate, "service"),
    ]
    parts.extend(_aliases(candidate))
    return frozenset().union(*(_tokens(part) for part in parts if part))


def _capability_signal(requested_tokens: frozenset[str], candidate: Mapping[str, object]) -> int:
    """Return a device-class/unit capability score from the candidate's snapshot attributes.

    Device classes are tokenized so multi-word classes (carbon_monoxide,
    signal_strength, pm25) match tokenized requests; the vocabulary is the
    candidate's own attribute, never a hand-picked list.
    """
    device_class = _text(candidate, "device_class")
    unit = _text(candidate, "unit")
    if device_class and _matched_tokens(requested_tokens, _tokens(device_class)):
        return _CAPABILITY_DEVICE_CLASS_WEIGHT
    if unit and _matched_tokens(requested_tokens, _tokens(unit) | {unit.lower()}):
        return _CAPABILITY_UNIT_WEIGHT
    return 0


def _area_floor_signal(requested_tokens: frozenset[str], candidate: Mapping[str, object]) -> int:
    """Return whether requested tokens mention the candidate area or floor."""
    area_tokens = _tokens(_text(candidate, "area_name"))
    floor_tokens = _tokens(_text(candidate, "floor_name"))
    return int(bool(_matched_tokens(requested_tokens, area_tokens))) + int(
        bool(_matched_tokens(requested_tokens, floor_tokens))
    )


def _service_support_signal(
    candidate: Mapping[str, object], ctx: FailureContext, snapshot: HomeSnapshot | None
) -> int:
    """Return whether an entity candidate is accepted by the requested service."""
    if snapshot is None or ctx.intent not in {Intent.CALL_SERVICE, Intent.RESOLVE_SELECTOR} or not ctx.service:
        return 0
    candidate_domain = _text(candidate, "domain")
    brief = snapshot.services_target.get(ctx.domain, {}).get(ctx.service)
    # Target metadata is the strongest service support source when present.
    if brief is not None:
        accepts = service_accepts_domain(brief, candidate_domain)
        return int(accepts is True)
    # Without metadata, fall back to Home Assistant's domain.service convention.
    return int(bool(candidate_domain and candidate_domain == ctx.domain))


def _field_overlap_signal(candidate: Mapping[str, object], ctx: FailureContext) -> int:
    """Return how many attempted service_data fields this service declares."""
    fields = candidate.get("fields")
    if not isinstance(fields, frozenset):
        return 0
    return len(set(ctx.service_data) & {str(field) for field in fields})


def _text(candidate: Mapping[str, object], key: str) -> str:
    """Return a string value from a candidate mapping."""
    value = candidate.get(key, "")
    return value if isinstance(value, str) else ""


def _aliases(candidate: Mapping[str, object]) -> tuple[str, ...]:
    """Return string aliases from a candidate mapping."""
    aliases = candidate.get("aliases", ())
    if not isinstance(aliases, tuple):
        return ()
    return tuple(alias for alias in aliases if isinstance(alias, str))


def _reverse_tiebreak(value: str) -> str:
    """Invert id text so reverse sorting still yields ascending ids for ties."""
    return "".join(chr(0x10FFFF - ord(char)) for char in value)
