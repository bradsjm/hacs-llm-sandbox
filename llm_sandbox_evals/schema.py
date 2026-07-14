"""Action-only eval case, trace, and report contracts."""

from dataclasses import dataclass, field
from typing import Literal

type ActionOutcomeReason = Literal[
    "ok",
    "equivalent_target_partition",
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

type FailureClassification = Literal[
    "cap_exhausted",
    "timeout",
    "model_protocol_error",
    "rate_limit",
    "provider_error",
]

type ScoringMode = Literal["end_state", "actions", "cap_exhausted"]

type EndStateStatus = Literal["not_authored", "unevaluable", "satisfied", "unsatisfied"]

type ScoreReason = Literal[
    "end_state_satisfied",
    "end_state_unsatisfied",
    ActionOutcomeReason,
    "cap_exhausted",
]


def variant_label(model_id: str, reasoning_effort: str | None) -> str:
    """Return the display-only identity for one resolved model variant."""
    return f"{model_id}({reasoning_effort or 'default'})"


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
class ModelDescriptor:
    """Resolved run-wide settings for one provider model."""

    model_id: str
    reasoning_effort: str | None
    temperature: float | None
    variant_label: str


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """Reload-safe snapshot of the configuration that defines one matrix run."""

    run_id: str
    created_at: str
    models: tuple[ModelDescriptor, ...]
    candidates: tuple[str, ...]
    cases: tuple[str, ...]
    prompt_profile: str
    concurrency: int
    model_timeout: float
    max_tool_calls: int


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


_SUPPORTED_STATE_DOMAINS = ("light", "switch")
_SUPPORTED_STATE_VALUES = ("on", "off")


@dataclass(frozen=True, slots=True)
class DesiredState:
    """One desired post-run entity state used as the primary scoring predicate.

    Restricted to ``light``/``switch`` domains with a binary ``on``/``off``
    state so the eval-only overlay reducer can deterministically evaluate it
    without emulating arbitrary Home Assistant service semantics.
    """

    entity_id: str
    state: str

    def __post_init__(self) -> None:
        """Validate the predicate's entity domain and desired state vocabulary."""
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("desired state entity_id must be a nonempty string")
        domain = self.entity_id.split(".", 1)[0] if "." in self.entity_id else ""
        if domain not in _SUPPORTED_STATE_DOMAINS:
            raise ValueError(
                f"desired state entity_id must be a {_SUPPORTED_STATE_DOMAINS} entity, got {self.entity_id!r}"
            )
        if self.state not in _SUPPORTED_STATE_VALUES:
            raise ValueError(f"desired state must be one of {_SUPPORTED_STATE_VALUES}, got {self.state!r}")


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One action request, its required successful service effects, and optional end-state predicates."""

    id: str
    home: str
    user_request: str
    required_actions: tuple[RequiredAction, ...]
    desired_states: tuple[DesiredState, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate desired-state entity IDs so contradictory goals cannot silently pick a score."""
        seen: set[str] = set()
        for predicate in self.desired_states:
            if predicate.entity_id in seen:
                raise ValueError(f"duplicate desired state entity_id: {predicate.entity_id!r}")
            seen.add(predicate.entity_id)


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
class OverlayStateSeed:
    """Frozen initial state for one predicate-relevant entity.

    Captured from the scoped snapshot so a saved trace can be re-scored
    without consulting fixture code.
    """

    entity_id: str
    domain: str
    state: str


@dataclass(frozen=True, slots=True)
class EndStateComparison:
    """Per-predicate observed final state versus the desired state."""

    desired: DesiredState
    actual_state: str | None
    matched: bool


@dataclass(frozen=True, slots=True)
class EndStateResult:
    """End-state assessment: primary score when evaluable, diagnostics otherwise."""

    status: EndStateStatus
    evaluable: bool
    passed: bool
    comparisons: tuple[EndStateComparison, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionError:
    """Structured provider or harness failure metadata for one execution attempt."""

    exception_type: str
    message: str
    status_code: int | None = None
    provider_code: str | None = None
    provider_model: str | None = None
    provider_detail: str | None = None


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
    usage: dict[str, int | float | bool | None] | None = None
    failure: FailureClassification | None = None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Binary quality outcome or an incomplete operational result.

    ``scoring_mode`` identifies which assessment determined the verdict:
    ``end_state`` for evaluable desired-state predicates, ``actions`` for the
    exact action-ledger fallback, ``cap_exhausted`` for the operational
    override, or ``None`` for incomplete execution.  ``score_reason`` carries
    the stable reason for that mode.
    """

    state: Literal["correct", "incorrect", "incomplete"]
    scoring_mode: ScoringMode | None
    score_reason: ScoreReason | None
    score: float = field(init=False)

    def __post_init__(self) -> None:
        """Derive the binary score from the outcome state."""
        object.__setattr__(self, "score", 1.0 if self.state == "correct" else 0.0)


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """Self-contained scoring-v8 eval trace with end-state and action evidence."""

    case_id: str
    candidate_id: str
    model_id: str
    answer: str | None
    required_actions: tuple[RequiredAction, ...]
    desired_states: tuple[DesiredState, ...]
    overlay_state_seeds: tuple[OverlayStateSeed, ...]
    recorded_invocations: tuple[dict[str, object], ...]
    end_state_result: EndStateResult
    outcome: CaseOutcome
    action_result: ActionResult
    action_ledger: ActionLedger
    tool_events: tuple[ToolEvent, ...]
    diagnostics: EvalDiagnostics
    reasoning_effort: str | None = None
    temperature: float | None = None
    scoring_version: Literal[8] = 8
    provider_error: str | None = None
    execution_error: ExecutionError | None = None
    user_request: str = ""
    conversation_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompletedCellRecord:
    """One terminal cell captured in a partial-run journal, never a report."""

    cell: dict[str, str | float | None]
    trace: CaseTrace
    completion_index: int
    finished_at: str


@dataclass(frozen=True, slots=True)
class PartialRunArtifact:
    """Typed cancellation/failure journal; intentionally not an EvaluationReport."""

    artifact_type: Literal["llm_sandbox_partial_run"]
    run_id: str
    descriptor: RunDescriptor
    status: Literal["cancelled", "failed"]
    finished: int
    total: int
    records: tuple[CompletedCellRecord, ...]
    error: str | None
    saved_at: str
