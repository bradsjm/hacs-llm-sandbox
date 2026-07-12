"""Immutable evidence records used by the v4 scorer."""
# These records are intentionally structural; their fields are the evidence API.
# ruff: noqa: D101, D102

from dataclasses import dataclass

from llm_sandbox_evals.schema import JsonScalar


@dataclass(frozen=True, slots=True)
class Provenance:
    tool_name: str
    call_index: int
    turn_index: int
    batch_index: int
    record_id: str


type FactValue = tuple[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    kind: str
    values: tuple[FactValue, ...]
    provenance: Provenance

    def get(self, key: str) -> JsonScalar:
        return dict(self.values).get(key)


@dataclass(frozen=True, slots=True)
class NormalizedEvidence:
    facts: tuple[EvidenceFact, ...] = ()

    def for_kind(self, kind: str) -> tuple[EvidenceFact, ...]:
        return tuple(fact for fact in self.facts if fact.kind == kind)
