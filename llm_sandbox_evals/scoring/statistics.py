"""Normalize keyed statistics payloads."""

from collections.abc import Mapping

from llm_sandbox_evals.scoring._facts import fact
from llm_sandbox_evals.scoring.contracts import EvidenceFact, Provenance


def normalize_statistics_output(output: Mapping[str, object], provenance: Provenance) -> tuple[EvidenceFact, ...]:
    """Normalize statistic rows keyed by their statistic identifier."""
    facts: list[EvidenceFact] = []
    statistics = output.get("statistics")
    if isinstance(statistics, Mapping):
        for statistic_id, payload in statistics.items():
            if not isinstance(statistic_id, str) or not isinstance(payload, Mapping):
                continue
            fact("scope", {"source": "statistics", "entity_id": statistic_id}, provenance, facts, statistic_id)
            for row in payload.get("rows", []):
                if isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) and isinstance(row[1], Mapping):
                    for field, value in row[1].items():
                        fact(
                            "statistic_value",
                            {"statistic_id": statistic_id, "when": row[0], "field": field, "value": value},
                            provenance,
                            facts,
                            statistic_id,
                        )
    return tuple(facts)
