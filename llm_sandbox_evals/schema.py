"""Action-only eval case, trace, and report contracts."""

from dataclasses import dataclass, field
from typing import Literal

type ActionOutcomeReason = Literal[
    "ok",
    "no_action",
    "action_rejected",
    "wrong_service",
    "wrong_target",
    "wrong_service_data",
    "wrong_service_and_target",
    "wrong_service_and_data",
    "wrong_target_and_data",
    "wrong_service_target_and_data",
    "missing_action",
    "unexpected_action",
    "duplicate_action",
    "multiple_action_mismatches",
    "action_mismatch",
]


@dataclass(frozen=True, slots=True)
class PromptCandidate:
    """One prompt candidate evaluated across the model matrix."""

    id: str
    api_prompt: str
    execute_home_code_description: str
    get_history_description: str
    get_statistics_description: str
    get_logbook_description: str
    get_automation_description: str


@dataclass(frozen=True, slots=True)
class RequiredAction:
    """One successful service effect required by an eval case."""

    domain: str
    service: str
    target_entity_ids: tuple[str, ...]
    service_data: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate the required successful effect identity."""
        if not self.domain or not self.service:
            raise ValueError("required action domain and service must be nonempty")
        if not self.target_entity_ids or any(not entity_id for entity_id in self.target_entity_ids):
            raise ValueError("required action target_entity_ids must be nonempty strings")


@dataclass(frozen=True, slots=True)
class ObservedAction:
    """One normalized successful service effect used for diagnostics."""

    domain: str
    service: str
    target_entity_ids: tuple[str, ...]
    service_data: dict[str, object]


@dataclass(frozen=True, slots=True)
class ActionComparison:
    """Dimension-level assessment of one expected action and its closest effect."""

    expected: RequiredAction
    actual: ObservedAction | None
    service_matches: bool
    target_matches: bool
    service_data_matches: bool
    matched: bool


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One action request and its required successful service effects."""

    id: str
    home: str
    user_request: str
    required_actions: tuple[RequiredAction, ...]


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One production tool call and its JSON-safe return envelope."""

    tool_name: str
    args: dict[str, object]
    output: dict[str, object]
    call_index: int = 0
    turn_index: int = 0
    batch_index: int = 0
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Exact successful-ledger comparison for one case."""

    passed: bool
    reason: ActionOutcomeReason
    comparisons: tuple[ActionComparison, ...] = ()
    unexpected_actions: tuple[ObservedAction, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionLedger:
    """Successful effects used for scoring and rejected effects kept for diagnostics."""

    successful: tuple[dict[str, object], ...] = ()
    rejected: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class EvalDiagnostics:
    """Operational facts that never change action correctness."""

    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    execute_repairs: int = 0
    model_turns: int = 0
    parallel_batches: int = 0
    max_batch_size: int = 1
    elapsed_seconds: float | None = None
    cap_exhausted: bool = False
    usage: dict[str, int | float | None] | None = None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Binary action quality outcome or an incomplete operational result."""

    state: Literal["correct", "incorrect", "incomplete"]
    reason: ActionOutcomeReason
    score: float = field(init=False)

    def __post_init__(self) -> None:
        """Derive the binary score from the outcome state."""
        object.__setattr__(self, "score", 1.0 if self.state == "correct" else 0.0)


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """Self-contained scoring-v5 action eval trace."""

    case_id: str
    candidate_id: str
    model_id: str
    answer: str | None
    required_actions: tuple[RequiredAction, ...]
    outcome: CaseOutcome
    action_result: ActionResult
    action_ledger: ActionLedger
    tool_events: tuple[ToolEvent, ...]
    diagnostics: EvalDiagnostics
    scoring_version: Literal[5] = 5
    provider_error: str | None = None
    user_request: str = ""
    conversation_id: str | None = None
