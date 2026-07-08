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
    """Outcome-only expectations: what the final answer + actions must look like."""

    answer_facts: tuple[str, ...] = ()
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
