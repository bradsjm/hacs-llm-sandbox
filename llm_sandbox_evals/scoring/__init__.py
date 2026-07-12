"""Small stable public surface for v2 eval scoring.

Integration note: the harness/report migration must pass ``EvalAnswer``, v2
``ToolEvent`` provenance, and separate successful/rejected action records into
``evaluate_case``. Legacy ``check_case``/``CheckResult`` and efficiency APIs
are intentionally removed; old dataset/report consumers must migrate rather
than be adapted here.
"""

from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence, Provenance
from llm_sandbox_evals.scoring.evaluate import evaluate_case, is_incomplete, score_case
from llm_sandbox_evals.scoring.evidence import normalize_events, successful_events

__all__ = [
    "EvidenceFact",
    "NormalizedEvidence",
    "Provenance",
    "build_action_ledger",
    "evaluate_case",
    "is_incomplete",
    "normalize_events",
    "score_actions",
    "score_case",
    "successful_events",
]
