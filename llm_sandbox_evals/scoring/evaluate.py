"""Score only successful service invocation effects."""

from collections.abc import Mapping, Sequence

from llm_sandbox_evals.schema import ActionLedger, ActionResult, CaseOutcome, EvalCase
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions


def evaluate_case(
    case: EvalCase,
    recorded_actions: Sequence[Mapping[str, object]],
) -> tuple[CaseOutcome, ActionResult, ActionLedger]:
    """Compare one case's authored actions with its successful recorded effects."""
    ledger = build_action_ledger(recorded_actions)
    result = score_actions(case.expected_actions, ledger)
    return (
        CaseOutcome("correct" if result.passed else "incorrect", result.reason),
        result,
        ledger,
    )
