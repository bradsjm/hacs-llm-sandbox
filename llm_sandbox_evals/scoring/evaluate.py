"""Score eval cases: end-state predicates primary, exact action multiset fallback."""

from collections.abc import Mapping, Sequence
from typing import Literal

from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    EndStateResult,
    EvalCase,
    OverlayStateSeed,
    ScoreReason,
    ScoringMode,
)
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.end_state import assess_end_state


def evaluate_case(
    case: EvalCase,
    recorded_actions: Sequence[Mapping[str, object]],
    *,
    overlay_seeds: Sequence[OverlayStateSeed],
    invoker_calls: Sequence[Mapping[str, object]],
) -> tuple[CaseOutcome, ActionResult, ActionLedger, EndStateResult]:
    """Evaluate one case: end-state predicates primary, action multiset fallback.

    Always builds the action ledger and exact action result for diagnostics.
    When desired states are evaluable, the end-state assessment is primary —
    a satisfied state passes even if the action ledger mismatches.  When no
    predicates exist or they are unevaluable, the exact action result
    determines the outcome.
    """
    ledger = build_action_ledger(recorded_actions)
    action_result = score_actions(case.required_actions, ledger)
    end_state = assess_end_state(case.desired_states, overlay_seeds, invoker_calls)

    if end_state.evaluable:
        mode: ScoringMode = "end_state"
        reason: ScoreReason = "end_state_satisfied" if end_state.passed else "end_state_unsatisfied"
        outcome_state: Literal["correct", "incorrect"] = "correct" if end_state.passed else "incorrect"
    else:
        mode = "actions"
        reason = action_result.reason
        outcome_state = "correct" if action_result.passed else "incorrect"

    return CaseOutcome(outcome_state, mode, reason), action_result, ledger, end_state
