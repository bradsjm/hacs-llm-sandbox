"""Registry dispatch for explicit eval-case primary oracles."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    AnswerResult,
    CaseOutcome,
    EndStateResult,
    EvalCase,
    OverlayStateSeed,
    ScoreReason,
    ScoringMode,
    ToolCallResult,
    ToolEvent,
)
from llm_sandbox_evals.scoring.actions import build_action_ledger, score_actions
from llm_sandbox_evals.scoring.end_state import assess_end_state
from llm_sandbox_evals.scoring.read_answers import score_answer
from llm_sandbox_evals.scoring.tool_calls import score_tool_calls


@dataclass(frozen=True, slots=True)
class CaseEvalResult:
    """Primary outcome plus every diagnostic oracle assessment for one case."""

    outcome: CaseOutcome
    action_result: ActionResult
    action_ledger: ActionLedger
    end_state_result: EndStateResult
    tool_call_result: ToolCallResult | None = None
    answer_result: AnswerResult | None = None


type _OracleScorer = Callable[
    [EvalCase, ActionResult, ActionLedger, EndStateResult, Sequence[ToolEvent], str | None],
    CaseEvalResult,
]


def evaluate_case(
    case: EvalCase,
    recorded_actions: Sequence[Mapping[str, object]],
    *,
    overlay_seeds: Sequence[OverlayStateSeed],
    invoker_calls: Sequence[Mapping[str, object]],
    tool_events: Sequence[ToolEvent] = (),
    answer: str | None = None,
) -> CaseEvalResult:
    """Compute shared diagnostics, then dispatch to the case's explicit primary oracle."""
    ledger = build_action_ledger(recorded_actions)
    action_result = score_actions(case.required_actions, ledger)
    end_state = assess_end_state(case.desired_entities, overlay_seeds, invoker_calls)
    return _SCORERS[case.oracle](case, action_result, ledger, end_state, tool_events, answer)


def _score_effect(
    _case: EvalCase,
    action_result: ActionResult,
    ledger: ActionLedger,
    end_state: EndStateResult,
    _tool_events: Sequence[ToolEvent],
    _answer: str | None,
) -> CaseEvalResult:
    """Apply the existing end-state-primary and action-fallback effect contract."""
    if end_state.evaluable:
        mode: ScoringMode = "end_state"
        reason: ScoreReason = "end_state_satisfied" if end_state.passed else "end_state_unsatisfied"
        outcome_state: Literal["correct", "incorrect"] = "correct" if end_state.passed else "incorrect"
    else:
        mode = "actions"
        reason = action_result.reason
        outcome_state = "correct" if action_result.passed else "incorrect"
    return CaseEvalResult(CaseOutcome(outcome_state, mode, reason), action_result, ledger, end_state)


def _score_tool_contract(
    case: EvalCase,
    action_result: ActionResult,
    ledger: ActionLedger,
    end_state: EndStateResult,
    tool_events: Sequence[ToolEvent],
    _answer: str | None,
) -> CaseEvalResult:
    """Use successful tool-call contracts as primary while retaining effect diagnostics."""
    result = score_tool_calls(case.expected_tool_calls, tool_events)
    outcome = CaseOutcome("correct" if result.passed else "incorrect", "tool_calls", result.reason)
    return CaseEvalResult(outcome, action_result, ledger, end_state, tool_call_result=result)


def _score_read_answer(
    case: EvalCase,
    action_result: ActionResult,
    ledger: ActionLedger,
    end_state: EndStateResult,
    _tool_events: Sequence[ToolEvent],
    answer: str | None,
) -> CaseEvalResult:
    """Use the authored typed answer as primary while retaining effect diagnostics."""
    predicate = case.expected_answer
    if predicate is None:
        raise ValueError("answer oracle requires expected_answer")
    result = score_answer(predicate, answer)
    outcome = CaseOutcome("correct" if result.passed else "incorrect", "answer", result.reason)
    return CaseEvalResult(outcome, action_result, ledger, end_state, answer_result=result)


_SCORERS: dict[Literal["effect", "tool_calls", "answer"], _OracleScorer] = {
    "effect": _score_effect,
    "tool_calls": _score_tool_contract,
    "answer": _score_read_answer,
}
