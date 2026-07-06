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
    """Deterministic expectations used to score a case."""

    tool_name: str
    required_tool_names: tuple[str, ...] = ()
    required_tool_sequence: tuple[str, ...] = ()
    # execution.status value: ok|code_error|helper_error|setup_error, or "na" when not applicable.
    execution_status: str = "ok"
    output_contains_entities: tuple[str, ...] = ()
    output_excludes_entities: tuple[str, ...] = ()
    evidence_contains_entities: tuple[str, ...] = ()
    evidence_excludes_entities: tuple[str, ...] = ()
    required_error_keys: tuple[str, ...] = ()
    # Dotted paths that must appear in any tool result or recorded action.
    required_result_paths: tuple[str, ...] = ()
    # Dotted tool_args paths that must all equal the expected JSON-compatible values on one observed tool call.
    required_tool_arg_values: tuple[tuple[str, object], ...] = ()
    actions: tuple[ExpectedAction, ...] = ()
    # (start_iso, end_iso) recorder window expectation, or None when not a recorder case.
    recorder_window: tuple[str, str] | None = None
    # Optional hard gates for metadata/no-retry cases.
    max_tool_turns: int | None = None
    max_successful_actions: int | None = None
    max_tool_calls: int | None = None


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
    par_turns: int
    action_domains: frozenset[str] = frozenset()
    max_turns: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One native provider tool call selected by the model."""

    id: str
    tool_name: str
    tool_args: dict[str, object]


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One assistant message in the agent loop."""

    tool_calls: tuple[ToolCall, ...]
    text: str
    assistant_message: dict[str, object]
    raw: str


@dataclass(frozen=True, slots=True)
class StepTrace:
    """Persisted trace of one tool-turn."""

    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[dict[str, object] | None, ...]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Result of running the selected tool against a frozen fixture.

    The real executor path populates ``result`` with the ``async_execute_home_code``
    return dict (keys include ``execution.status``, ``output``, ``printed``,
    ``actions``). Recorder emulation populates ``result`` with the tool's public
    response envelope. ``recorded_actions`` carries ProposedAction dicts captured
    by the non-live RecordingInvoker.
    """

    ok: bool
    tool_name: str
    result: dict[str, object] | None
    recorded_actions: tuple[dict[str, object], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One scoring check outcome."""

    name: str
    passed: bool
    required: bool
    feedback: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool's name, description, and JSON Schema exposed to the model."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """Full trace for one candidate/model/case execution."""

    case_id: str
    category: str
    candidate_id: str
    model_id: str
    score: float
    prompt: str
    raw_output: str
    tool_call: dict[str, object] | None
    tool_result: dict[str, object] | None
    recorded_actions: tuple[dict[str, object], ...]
    checks: tuple[CheckResult, ...]
    turns: int
    par_turns: int
    final_answer: str
    steps: tuple[StepTrace, ...]


@dataclass(frozen=True, slots=True)
class CandidateModelScore:
    """Aggregated scores for one candidate/model pair."""

    candidate_id: str
    model_id: str
    mean: float
    mean_turns: float
    per_category: dict[str, float]
    case_scores: dict[str, float]
    # Authored prompt sizes; defaults keep old run.json files loadable.
    api_prompt_chars: int = 0
    prompt_chars: int = 0


@dataclass(frozen=True, slots=True)
class RunResult:
    """Complete result for one eval matrix run."""

    run_id: str
    created_at: str
    candidate_ids: list[str]
    model_ids: list[str]
    case_ids: list[str]
    traces: list[CaseTrace]
    scores: list[CandidateModelScore]
