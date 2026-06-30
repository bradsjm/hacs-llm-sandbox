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
    # execution.status value: ok|code_error|helper_error|setup_error, or "na" when not applicable.
    execution_status: str = "ok"
    output_contains_entities: tuple[str, ...] = ()
    output_excludes_entities: tuple[str, ...] = ()
    actions: tuple[ExpectedAction, ...] = ()
    # (start_iso, end_iso) recorder window expectation, or None when not a recorder case.
    recorder_window: tuple[str, str] | None = None
    visible_only: bool = True


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


@dataclass(frozen=True, slots=True)
class ModelResult:
    """A model adapter's raw and parsed response."""

    raw_text: str
    tool_call: dict[str, object] | None
    error: str | None


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
    """A tool's name, description, and argument names exposed to the model."""

    name: str
    description: str
    args: tuple[str, ...]
