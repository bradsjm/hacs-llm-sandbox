"""Shared data contracts for the eval harness.

These dataclasses are the stable contract between the fixture, model-adapter,
tool-runner, scoring, and orchestration layers. Later waves consume them; do
not rename fields without updating all consumers.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptCandidate:
    """A complete set of prompt texts under evaluation.

    The baseline candidate is auto-built from the production prompt builders in
    ``custom_components.llm_sandbox.llm_api.prompts``. The follow-up optimizer
    phase will emit additional candidates with rewritten instruction text.
    """

    id: str
    api_prompt: str
    execute_home_code_description: str
    get_history_description: str
    get_statistics_description: str
    get_logbook_description: str


@dataclass(frozen=True, slots=True)
class CaseContext:
    """The initiating LLM request context for a case."""

    platform: str = "test"
    device_id: str | None = None
    language: str | None = "en"


@dataclass(frozen=True, slots=True)
class ExpectedAction:
    """A service action the model is expected to (or not to) propose."""

    domain: str
    service: str
    target_entity_ids: tuple[str, ...] = ()
    service_data: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ToolResultCheck:
    """Structured evidence expected in a successful recorder tool result."""

    tool_name: str
    entity_ids: tuple[str, ...] = ()
    statistic_ids: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    period: str | None = None
    entry_values: tuple[str, ...] = ()
    min_results: int = 1


@dataclass(frozen=True, slots=True)
class BlockedOutcome:
    """Structured expectations for a deliberately blocked action attempt."""

    max_attempts: int = 1
    error_keys: tuple[str, ...] = ()
    acknowledgement_values: tuple[str, ...] = ()
    """Diagnostic-only prose hints retained for report context."""
    answer_excludes: tuple[str, ...] = ()
    success_claim_excludes: tuple[str, ...] = (
        "successfully",
        "completed successfully",
        "i turned",
        "i have turned",
        "i opened",
        "i have opened",
        "i closed",
        "i have closed",
        "i unlocked",
        "i have unlocked",
        "i locked",
        "i have locked",
        "it is now",
        "it's now",
        "has been turned",
        "was turned",
    )


@dataclass(frozen=True, slots=True)
class Expected:
    """Outcome-evidence expectations: salient facts, exclusions, and side effects.

    ``answer_values`` are expected facts in structured tool/action evidence.
    ``expected_values`` is a legacy migration bucket treated the same way.
    Final-answer prose checks are diagnostic only; scoring should be grounded in
    structured tool outputs, recorded actions, recorder checks, and blocked-action
    side effects instead of parsing the model's prose.
    """

    expected_values: tuple[str, ...] = ()
    answer_values: tuple[str, ...] = ()
    provenance_values: tuple[str, ...] = ()
    tool_result_checks: tuple[ToolResultCheck, ...] = ()
    blocked_outcome: BlockedOutcome | None = None
    answer_excludes: tuple[str, ...] = ()
    actions: tuple[ExpectedAction, ...] = ()
    guidance_candidate: str | None = None
    max_tool_calls: int = 10


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A single predefined user request evaluated against a frozen fixture."""

    id: str
    category: str
    home: str
    user_request: str
    actions_enabled: bool
    llm_context: CaseContext
    expected: Expected
    action_domains: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One scoring check outcome."""

    name: str
    passed: bool
    required: bool
    feedback: str


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One paired tool call/return captured for an eval cell trace.

    ``output`` is the production tool return payload verbatim (a dict envelope
    such as the ``execute_home_code`` result or a recorder result), used by
    structured evidence checks and the final-tool success gate.
    """

    tool_name: str
    args: dict[str, object]
    output: dict[str, object]


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """Outcome-only trace for one candidate/model/case execution."""

    case_id: str
    category: str
    candidate_id: str
    model_id: str
    score: float
    output: str
    tool_call_count: int
    recorded_actions: tuple[dict[str, object], ...]
    checks: tuple[CheckResult, ...]
    error: str | None
    # Trailing field with default so existing constructors stay valid when they
    # omit tool events (e.g. synthetic traces in tests / error traces).
    tool_events: tuple[ToolEvent, ...] = ()
