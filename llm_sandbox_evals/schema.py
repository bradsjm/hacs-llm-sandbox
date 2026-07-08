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


@dataclass(frozen=True, slots=True)
class Expected:
    """Outcome-evidence expectations: salient facts, exclusions, and side effects.

    ``expected_values`` are the tokens that prove the task was accomplished. They
    are audited case-insensitively as substrings across an any-source evidence
    blob (final answer + every tool return payload), not against any single tool
    call, tool order, or tool argument. Keep each token distinctive so it cannot
    accidentally match noise (e.g. prefer ``23.4`` over ``1``).
    """

    expected_values: tuple[str, ...] = ()
    answer_excludes: tuple[str, ...] = ()
    actions: tuple[ExpectedAction, ...] = ()
    max_tool_calls: int = 8
    reference_tool_calls: int | None = None


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
    such as the ``execute_home_code`` result or a recorder result), used by the
    any-source evidence audit and the ``execution_ok`` gate.
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
