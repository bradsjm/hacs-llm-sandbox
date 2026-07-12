"""Execute payload normalization entry point."""
# ruff: noqa: D103

from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance
from llm_sandbox_evals.scoring.evidence import _normalize_execute


def normalize_execute_output(output: object, provenance: Provenance) -> tuple[EvidenceFact, ...]:
    facts: list[EvidenceFact] = []
    _normalize_execute(output, provenance, facts)
    return tuple(facts)
