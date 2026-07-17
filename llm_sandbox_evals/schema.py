"""Eval case, oracle, trace, and report contracts."""

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

type ScoringMode = Literal["end_state", "actions", "tool_calls", "answer", "cap_exhausted"]

type EndStateStatus = Literal["not_authored", "unevaluable", "satisfied", "unsatisfied"]

type ScoreReason = Literal[
    "end_state_satisfied",
    "end_state_unsatisfied",
    "tool_calls_matched",
    "tool_calls_missing",
    "tool_calls_mismatched",
    "tool_calls_no_events",
    "answer_correct",
    "answer_incorrect",
    "answer_unparseable",
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
    get_energy_description: str
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
    judge_model: str | None = None
    judge_rubric_id: str = "llm_sandbox_code_quality"
    judge_rubric_version: int = 2


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
class DesiredEntity:
    """Sparse authored final values for one predicate-relevant entity."""

    entity_id: str
    state: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require entity identity and at least one authored final value."""
        if not self.entity_id or not isinstance(self.entity_id, str):
            raise ValueError("desired entity entity_id must be a nonempty string")
        if self.state is None and not self.attributes:
            raise ValueError("desired entity must author state or attributes")


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    """One successful production tool call required by a tool-contract case."""

    tool_name: str
    args: dict[str, object] = field(default_factory=dict)
    arg_contains: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnswerPredicate:
    """One deterministic typed predicate for a plain-text read answer."""

    kind: Literal["boolean", "count", "entity_set", "scalar", "state", "time_range"]
    value: bool | None = None
    count: int | None = None
    entity_ids: tuple[str, ...] = ()
    scalar_value: float | None = None
    unit: str | None = None
    tolerance: float | None = None
    entity_id: str | None = None  # documentation-only; the deterministic parser does not use this field
    state: str | None = None
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True, slots=True)
class RequestVariant:
    """One stable authored request text for an eval task."""

    id: str
    text: str

    def __post_init__(self) -> None:
        """Require stable nonempty variant identity and agent input text."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("request variant id must be a nonempty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("request variant text must be a nonempty string")


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One stable task with authored request variants and scoring evidence."""

    id: str
    home: str
    category: str
    requests: tuple[RequestVariant, ...]
    required_actions: tuple[RequiredAction, ...]
    desired_entities: tuple[DesiredEntity, ...] = ()
    tags: tuple[str, ...] = ()
    oracle: Literal["effect", "tool_calls", "answer"] = "effect"
    expected_tool_calls: tuple[ExpectedToolCall, ...] = ()
    expected_answer: AnswerPredicate | None = None
    judge_code: bool = False

    def __post_init__(self) -> None:
        """Validate task identity, request variants, tags, and desired-entity uniqueness."""
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("eval case category must be a nonempty string")
        if not self.requests:
            raise ValueError("eval case must contain at least one request variant")
        if self.requests[0].id != "canonical":
            raise ValueError("first request variant id must be 'canonical'")
        variant_ids = [variant.id for variant in self.requests]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("request variant ids must be unique")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("eval case tags must be unique")
        seen: set[str] = set()
        for predicate in self.desired_entities:
            if predicate.entity_id in seen:
                raise ValueError(f"duplicate desired entity entity_id: {predicate.entity_id!r}")
            seen.add(predicate.entity_id)
        if self.oracle == "answer" and self.expected_answer is None:
            raise ValueError("answer oracle requires expected_answer")
        if self.oracle == "tool_calls" and not self.expected_tool_calls:
            raise ValueError("tool_calls oracle requires at least one expected_tool_call")


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
class ToolCallComparison:
    """One authored tool call and the successful event matched to it, if any."""

    expected: ExpectedToolCall
    matched_event: ToolEvent | None


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """One-to-one successful tool-call contract assessment."""

    passed: bool
    reason: Literal[
        "tool_calls_matched",
        "tool_calls_missing",
        "tool_calls_mismatched",
        "tool_calls_no_events",
    ]
    comparisons: tuple[ToolCallComparison, ...] = ()
    unmatched_events: tuple[ToolEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Deterministic typed read-answer assessment."""

    passed: bool
    reason: Literal["answer_correct", "answer_incorrect", "answer_unparseable"]
    extracted_value: str | None = None


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
    """Frozen initial values for one predicate-relevant entity.

    Captured from the scoped snapshot so a saved trace can be re-scored
    without consulting fixture code.
    """

    entity_id: str
    domain: str
    state: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EndStateComparison:
    """Per-predicate observed final values versus authored desired values."""

    desired: DesiredEntity
    actual_state: str | None
    actual_attributes: dict[str, object]
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
    exact action-ledger fallback, ``tool_calls`` or ``answer`` for their
    explicitly authored primary oracles, ``cap_exhausted`` for the operational
    override, or ``None`` for incomplete execution. ``score_reason`` carries
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
    """Self-contained scoring-v9 trace with authored inputs and oracle evidence."""

    case_id: str
    candidate_id: str
    model_id: str
    request_variant_id: str
    request_text: str
    answer: str | None
    required_actions: tuple[RequiredAction, ...]
    desired_entities: tuple[DesiredEntity, ...]
    overlay_state_seeds: tuple[OverlayStateSeed, ...]
    recorded_invocations: tuple[dict[str, object], ...]
    end_state_result: EndStateResult
    outcome: CaseOutcome
    action_result: ActionResult
    action_ledger: ActionLedger
    tool_events: tuple[ToolEvent, ...]
    diagnostics: EvalDiagnostics
    oracle: Literal["effect", "tool_calls", "answer"] = "effect"
    expected_tool_calls: tuple[ExpectedToolCall, ...] = ()
    expected_answer: AnswerPredicate | None = None
    tool_call_result: ToolCallResult | None = None
    answer_result: AnswerResult | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    scoring_version: Literal[9] = 9
    provider_error: str | None = None
    execution_error: ExecutionError | None = None
    category: str = ""
    tags: tuple[str, ...] = ()
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
