"""Advisory native evaluator for the quality of submitted sandbox code."""

import asyncio
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json

from custom_components.llm_sandbox.llm_api.executor import MAX_MONTY_CODE_CHARS
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

from llm_sandbox_evals.schema import CaseTrace, ToolEvent

CODE_QUALITY_RUBRIC = """Assess the submitted Monty/Python calls collectively as ephemeral glue code
for one Home Assistant Assist request, not as production Python.

Prioritize whether the code contributes effectively to the request, uses the available facade appropriately,
minimizes avoidable model/tool round trips and repeated or overly broad reads, performs dependent discovery,
filtering, computation, conditions, and actions together when practical, and returns compact useful evidence
for the assistant's response. Treat successful runtime normalization and transparent entity resolution as
intended behavior. Literal IDs established by the request, earlier discovery, or resolution are not defects.

Do not assess Ruff, linting, formatting, comments, docstrings, typing, abstraction, reuse, tests, defensive
production hardening, or long-term maintainability unless they directly caused failed or wasteful execution.
Do not reconstruct exact action, end-state, tool-call, or answer correctness; the supplied deterministic outcome
is trusted context and those checks are evaluated separately.

Score 0.9-1.0 for effective near-minimal execution, 0.7-0.89 for effective execution with limited avoidable work,
0.4-0.69 for useful but materially fragmented, broad, or repetitive execution, and 0.0-0.39 for weak or largely
ineffective execution. Set pass true exactly when the score is at least 0.7. Return a concise reason naming the
strongest positive and largest avoidable cost."""

_STATUS_BOUND_MARKER = "[status truncated]"
_ACTION_BOUND_MARKER = "[action content truncated]"
_ACTION_MARKER_KEY = "_projection"
_CONTEXT_BOUND_MARKER = "[content summarized]"

# These projection-only limits keep one complete trajectory and compact runtime
# evidence inside a conservative judge context. They do not change production
# execution or scoring limits.
MAX_CODE_QUALITY_REQUEST_CHARS = 2_000
MAX_CODE_QUALITY_TOTAL_SOURCE_CHARS = 20_000
MAX_CODE_QUALITY_EXECUTION_STATUS_CHARS = 128
MAX_CODE_QUALITY_ACTIONS = 32
MAX_CODE_QUALITY_ACTIONS_PER_SUBMISSION = 32
MAX_CODE_QUALITY_ACTION_RECORD_CHARS = 1_024
MAX_CODE_QUALITY_SERIALIZED_ACTION_CHARS = 16_384
MAX_CODE_QUALITY_ACTION_VALUE_CHARS = 768
MAX_CODE_QUALITY_CONTEXT_VALUE_CHARS = 4_096
MAX_CODE_QUALITY_CONTEXT_SAMPLE_ITEMS = 8

_ACTION_MARKER_RESERVE_CHARS = 128

type CodeQualityObserver = Callable[[], None]


@dataclass(frozen=True, slots=True)
class CodeQualityValue:
    """Bounded JSON-safe value plus its original projection size."""

    value: object
    serialized_chars: int | None
    item_count: int | None
    truncated: bool


@dataclass(frozen=True, slots=True)
class CodeQualitySubmission:
    """Bounded projection of one execute_home_code submission."""

    call_index: int
    source: str
    execution_status: str | None
    output: CodeQualityValue
    actions: tuple[dict[str, object], ...]
    resolutions: CodeQualityValue
    notes: CodeQualityValue


@dataclass(frozen=True, slots=True)
class CodeQualityToolContext:
    """Compact context for one interleaved non-code tool call."""

    call_index: int
    tool_name: str
    args: CodeQualityValue
    status: str


@dataclass(frozen=True, slots=True)
class CodeQualityProjection:
    """Complete bounded trajectory shown to the judge."""

    request_text: str
    outcome_state: str
    scoring_mode: str | None
    score_reason: str | None
    submissions: tuple[CodeQualitySubmission, ...]
    other_tool_calls: tuple[CodeQualityToolContext, ...]


def build_code_quality_projection(trace: CaseTrace) -> CodeQualityProjection | None:
    """Build complete bounded judge context, or return ``None`` when it cannot fit."""
    if len(trace.request_text) > MAX_CODE_QUALITY_REQUEST_CHARS:
        return None
    ordered_events = sorted(trace.tool_events, key=lambda event: event.call_index)
    execute_events = tuple(event for event in ordered_events if event.tool_name == "execute_home_code")
    submissions = _project_submissions(execute_events)
    if submissions is None:
        return None
    return CodeQualityProjection(
        request_text=trace.request_text,
        outcome_state=trace.outcome.state,
        scoring_mode=trace.outcome.scoring_mode,
        score_reason=trace.outcome.score_reason,
        submissions=submissions,
        other_tool_calls=tuple(
            _tool_context(event) for event in ordered_events if event.tool_name != "execute_home_code"
        ),
    )


def _project_submissions(events: Sequence[ToolEvent]) -> tuple[CodeQualitySubmission, ...] | None:
    """Keep every code call or decline judging instead of hiding part of the trajectory."""
    sources: list[str] = []
    source_chars = 0
    for event in events:
        code = event.args.get("code")
        if not isinstance(code, str) or len(code) > MAX_MONTY_CODE_CHARS:
            return None
        source_chars += len(code)
        if source_chars > MAX_CODE_QUALITY_TOTAL_SOURCE_CHARS:
            return None
        sources.append(code)

    action_rows = _project_actions(events)
    if action_rows is None:
        return None
    return tuple(
        _submission(event, sources[index], action_rows[index]) for index, event in enumerate(events)
    )


def _submission(event: ToolEvent, source: str, actions: tuple[dict[str, object], ...]) -> CodeQualitySubmission:
    """Project code plus the runtime evidence needed to assess its utility."""
    execution = event.output.get("execution")
    raw_status = execution.get("status") if isinstance(execution, dict) else None
    return CodeQualitySubmission(
        call_index=event.call_index,
        source=source,
        execution_status=(
            _truncate_text(raw_status, MAX_CODE_QUALITY_EXECUTION_STATUS_CHARS, _STATUS_BOUND_MARKER)
            if isinstance(raw_status, str)
            else None
        ),
        output=_bounded_context_value(event.output.get("output")),
        actions=actions,
        resolutions=_bounded_context_value(event.output.get("resolutions")),
        notes=_bounded_context_value(event.output.get("notes")),
    )


def _project_actions(
    events: Sequence[ToolEvent],
) -> tuple[tuple[dict[str, object], ...], ...] | None:
    """Copy every action record or decline judging when the complete set cannot fit."""
    rows: list[tuple[dict[str, object], ...]] = []
    total_actions = 0
    for event in events:
        raw_actions = event.output.get("actions")
        if raw_actions is None:
            rows.append(())
            continue
        if not isinstance(raw_actions, list) or any(not isinstance(action, dict) for action in raw_actions):
            return None
        if len(raw_actions) > MAX_CODE_QUALITY_ACTIONS_PER_SUBMISSION:
            return None
        total_actions += len(raw_actions)
        if total_actions > MAX_CODE_QUALITY_ACTIONS:
            return None
        bounded_actions = tuple(_bounded_action_record(dict(action)) for action in raw_actions)
        if _contains_action_bound_marker(bounded_actions):
            return None
        rows.append(bounded_actions)
    if (rows_length := _serialized_length(rows)) is None or rows_length > MAX_CODE_QUALITY_SERIALIZED_ACTION_CHARS:
        return None
    return tuple(rows)


def _bounded_action_record(action: dict[str, object]) -> dict[str, object]:
    """Copy one action with bounded keys, values, nesting, and serialized size."""
    content_budget = MAX_CODE_QUALITY_ACTION_RECORD_CHARS - _ACTION_MARKER_RESERVE_CHARS
    bounded: dict[str, object] = {}
    omitted = False
    for raw_key, raw_value in action.items():
        key = raw_key if isinstance(raw_key, str) else str(raw_key)
        key = _truncate_text(key, MAX_CODE_QUALITY_ACTION_VALUE_CHARS, _ACTION_BOUND_MARKER)
        value = _bounded_action_value(raw_value)
        candidate = dict(bounded)
        candidate[key] = value
        candidate_length = _serialized_length(candidate)
        if candidate_length is None or candidate_length > content_budget:
            omitted = True
            continue
        bounded[key] = value
    if omitted:
        bounded[_ACTION_MARKER_KEY] = _ACTION_BOUND_MARKER
    bounded_length = _serialized_length(bounded)
    if bounded_length is None or bounded_length > MAX_CODE_QUALITY_ACTION_RECORD_CHARS:
        return {_ACTION_MARKER_KEY: _ACTION_BOUND_MARKER}
    return bounded


def _bounded_action_value(value: object) -> object:
    """Keep ordinary JSON values or replace oversized nested content with a marker."""
    if isinstance(value, str):
        return _truncate_text(value, MAX_CODE_QUALITY_ACTION_VALUE_CHARS, _ACTION_BOUND_MARKER)
    if isinstance(value, dict | list | tuple):
        json_value = list(value) if isinstance(value, tuple) else value
        value_length = _serialized_length(json_value)
        if value_length is not None and value_length <= MAX_CODE_QUALITY_ACTION_VALUE_CHARS:
            return deepcopy(json_value)
        return _ACTION_BOUND_MARKER
    value_length = _serialized_length(value)
    if value_length is not None and value_length <= MAX_CODE_QUALITY_ACTION_VALUE_CHARS:
        return value
    return _ACTION_BOUND_MARKER


def _contains_action_bound_marker(value: object) -> bool:
    """Return whether action bounding had to omit any essential content."""
    if isinstance(value, str):
        return _ACTION_BOUND_MARKER in value
    if isinstance(value, dict):
        return any(
            _ACTION_BOUND_MARKER in str(key) or _contains_action_bound_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_action_bound_marker(item) for item in value)
    return False


def _tool_context(event: ToolEvent) -> CodeQualityToolContext:
    """Project one non-code tool call without returning its potentially large result."""
    execution = event.output.get("execution")
    execution_status = execution.get("status") if isinstance(execution, dict) else None
    raw_status = event.output.get("status")
    if isinstance(execution_status, str):
        status = execution_status
    elif isinstance(raw_status, str):
        status = raw_status
    elif "error" in event.output:
        status = "error"
    else:
        status = "ok"
    return CodeQualityToolContext(
        call_index=event.call_index,
        tool_name=event.tool_name,
        args=_bounded_context_value(event.args),
        status=_truncate_text(status, MAX_CODE_QUALITY_EXECUTION_STATUS_CHARS, _STATUS_BOUND_MARKER),
    )


def _bounded_context_value(value: object) -> CodeQualityValue:
    """Keep small JSON values and summarize large values with their original size."""
    json_value = list(value) if isinstance(value, tuple) else value
    serialized_chars = _serialized_length(json_value)
    item_count = len(json_value) if isinstance(json_value, dict | list) else None
    if serialized_chars is not None and serialized_chars <= MAX_CODE_QUALITY_CONTEXT_VALUE_CHARS:
        return CodeQualityValue(deepcopy(json_value), serialized_chars, item_count, False)
    bounded: object
    if isinstance(json_value, str):
        bounded = _truncate_text(
            json_value,
            MAX_CODE_QUALITY_CONTEXT_VALUE_CHARS,
            _CONTEXT_BOUND_MARKER,
        )
    elif isinstance(json_value, dict):
        bounded = {
            "kind": "mapping",
            "sample": {
                str(key): _bounded_action_value(item)
                for key, item in list(json_value.items())[:MAX_CODE_QUALITY_CONTEXT_SAMPLE_ITEMS]
            },
        }
    elif isinstance(json_value, list):
        bounded = {
            "kind": "sequence",
            "sample": [
                _bounded_action_value(item)
                for item in json_value[:MAX_CODE_QUALITY_CONTEXT_SAMPLE_ITEMS]
            ],
        }
    else:
        bounded = {"kind": type(value).__name__, "value": _CONTEXT_BOUND_MARKER}
    return CodeQualityValue(bounded, serialized_chars, item_count, True)


def _serialized_length(value: object) -> int | None:
    """Return compact strict-JSON length, or None for non-JSON-safe values."""
    try:
        return len(json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    except TypeError, ValueError, OverflowError:
        return None


def _truncate_text(value: str, limit: int, marker: str) -> str:
    """Keep text within a character limit while making truncation explicit."""
    if len(value) <= limit:
        return value
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker


@dataclass(slots=True)
class CodeQualityJudge(Evaluator[object, CaseTrace, object]):
    """Run one advisory code-quality judge call for a matrix cell."""

    model: str
    model_timeout: float
    on_judging: CodeQualityObserver | None = None
    on_terminal: CodeQualityObserver | None = None

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the stable evaluator identity used in native eval artifacts."""
        return "code_quality_judge"

    def build_serialization_arguments(self) -> dict[str, object]:
        """Serialize configuration while keeping runtime observers out of artifacts."""
        return {"model": self.model, "model_timeout": self.model_timeout}

    async def evaluate(self, ctx: EvaluatorContext[object, CaseTrace, object]) -> dict[str, EvaluationReason]:
        """Judge complete bounded context without affecting deterministic scoring."""
        projection = build_code_quality_projection(ctx.output)
        _notify(self.on_judging)
        if projection is None:
            _notify(self.on_terminal)
            return {}
        judge_input = {
            "request_text": projection.request_text,
            "deterministic_outcome": {
                "state": projection.outcome_state,
                "scoring_mode": projection.scoring_mode,
                "score_reason": projection.score_reason,
            },
        }
        judge_output = {
            "execute_home_code": [asdict(submission) for submission in projection.submissions],
            "other_tool_calls": [asdict(tool_call) for tool_call in projection.other_tool_calls],
        }
        try:
            grading = await asyncio.wait_for(
                judge_input_output(
                    judge_input,
                    judge_output,
                    CODE_QUALITY_RUBRIC,
                    model=self.model,
                    model_settings=None,
                ),
                timeout=self.model_timeout,
            )
            results = {
                "code_quality_score": EvaluationReason(value=grading.score, reason=grading.reason),
                "code_quality_pass": EvaluationReason(value=grading.pass_, reason=grading.reason),
            }
        except asyncio.CancelledError:
            raise
        except Exception:
            _notify(self.on_terminal)
            raise
        _notify(self.on_terminal)
        return results


def _notify(observer: CodeQualityObserver | None) -> None:
    """Keep optional lifecycle observers from changing evaluator behavior."""
    if observer is None:
        return
    try:
        observer()
    except Exception:  # noqa: BLE001 - observer failures must not affect evaluation.
        return
