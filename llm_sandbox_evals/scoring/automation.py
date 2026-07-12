"""Automation payload normalization entry point."""
# ruff: noqa: D103

from collections.abc import Mapping

from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance
from llm_sandbox_evals.scoring.evidence import _normalize_automation


def normalize_automation_output(output: Mapping[str, object], provenance: Provenance) -> tuple[EvidenceFact, ...]:
    facts: list[EvidenceFact] = []
    _normalize_automation(output, provenance, facts)
    return tuple(facts)
