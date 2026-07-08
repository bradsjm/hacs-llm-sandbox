"""Structured error payload helpers for LLM Sandbox LLM tools."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict, cast
from uuid import uuid4

import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.json import JsonObjectType

from ..types import ActionRecord, TranslationPlaceholders


class HelperErrorExecutionPayload(TypedDict):
    """Execution metadata for helper validation failures."""

    status: Literal["helper_error"]
    message: str
    kind: NotRequired[str]
    guidance: NotRequired[Mapping[str, object]]
    adjustments: NotRequired[list[dict[str, object]]]


class CodeErrorExecutionPayload(TypedDict):
    """Execution metadata for Monty code failures."""

    status: Literal["code_error"]
    message: str
    kind: NotRequired[str]
    guidance: NotRequired[Mapping[str, object]]
    adjustments: NotRequired[list[dict[str, object]]]


class SetupErrorExecutionPayload(TypedDict):
    """Execution metadata for pre-execution setup failures."""

    status: Literal["setup_error"]
    message: str
    kind: NotRequired[str]
    guidance: NotRequired[Mapping[str, object]]


class HelperErrorPayload(TypedDict):
    """Top-level helper-error execution response."""

    execution: HelperErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]
    notes: NotRequired[list[str]]


class CodeErrorPayload(TypedDict):
    """Top-level code-error execution response."""

    execution: CodeErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]
    notes: NotRequired[list[str]]


class SetupErrorPayload(TypedDict):
    """Top-level setup-error execution response."""

    execution: SetupErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]


type _ExecutionStatus = Literal["helper_error", "code_error", "setup_error"]
type _ToolErrorMessageBuilder = Callable[[TranslationPlaceholders], str]


@dataclass(frozen=True, slots=True)
class HelperExecutionError(Exception):
    """Recoverable helper failure surfaced as code-mode execution metadata."""

    helper: str
    key: str
    placeholders: TranslationPlaceholders
    guidance: Mapping[str, object] | None = None
    marker: str = field(default_factory=lambda: f"llm_sandbox_helper_error:{uuid4().hex}")

    def __post_init__(self) -> None:
        """Expose only the opaque marker when Monty stringifies this error."""
        Exception.__init__(self, self.marker)


@dataclass(frozen=True, slots=True)
class RecoverableToolError(Exception):
    """Controlled recorder-tool failure converted to an error envelope."""

    key: str
    placeholders: TranslationPlaceholders

    def __post_init__(self) -> None:
        """Initialize Exception with the stable error key."""
        Exception.__init__(self, self.key)


def _build_execution_payload(
    status: _ExecutionStatus,
    message: str,
    *,
    kind: str | None = None,
    guidance: Mapping[str, object] | None = None,
    adjustments: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return shared execution metadata for code/helper/setup errors."""
    execution: dict[str, object] = {"status": status, "message": message}
    if kind:
        execution["kind"] = kind
    # Guidance is omitted when unavailable to keep the error payload compact.
    if guidance:
        execution["guidance"] = dict(guidance)
    if adjustments:
        execution["adjustments"] = adjustments
    return execution


def _build_error_payload(
    execution: dict[str, object],
    *,
    printed: list[str] | None = None,
    actions: list[ActionRecord] | None = None,
) -> dict[str, object]:
    """Return shared top-level error envelope for execution failures."""
    payload: dict[str, object] = {"execution": execution, "output": None}
    if printed:
        payload["printed"] = printed
    if actions:
        payload["actions"] = actions
    return payload


def helper_error_payload(
    err: HelperExecutionError,
    *,
    message: str,
    kind: str | None = None,
    guidance: Mapping[str, object] | None = None,
    adjustments: list[dict[str, object]],
    printed: list[str],
    actions: list[ActionRecord] | None = None,
) -> HelperErrorPayload:
    """Return compact helper-error execution payload."""
    execution = _build_execution_payload(
        "helper_error",
        _message_or_key_fallback(message, err.key),
        kind=kind,
        guidance=guidance,
        adjustments=adjustments,
    )
    return cast(HelperErrorPayload, _build_error_payload(execution, printed=printed, actions=actions))


def tool_error_message(key: str, placeholders: TranslationPlaceholders) -> str | None:
    """Return a compact placeholder-aware LLM-facing message for a known key."""
    builder = _TOOL_ERROR_MESSAGES.get(key)
    if builder is None:
        return None
    return builder(placeholders)


def code_error_payload(
    *,
    kind: str | None,
    message: str,
    adjustments: list[dict[str, object]],
    printed: list[str],
    actions: list[ActionRecord] | None = None,
    guidance: Mapping[str, object] | None = None,
) -> CodeErrorPayload:
    """Return compact code-error execution payload."""
    execution = _build_execution_payload(
        "code_error",
        _message_or_key_fallback(message, kind),
        kind=kind,
        guidance=guidance,
        adjustments=adjustments,
    )
    return cast(CodeErrorPayload, _build_error_payload(execution, printed=printed, actions=actions))


def setup_error_payload(
    key: str,
    placeholders: TranslationPlaceholders,
    *,
    guidance: Mapping[str, object] | None = None,
) -> SetupErrorPayload:
    """Return compact setup-error execution payload."""
    execution = _build_execution_payload(
        "setup_error",
        _message_or_key_fallback(_setup_error_message(key, placeholders), key),
        guidance=guidance,
    )
    return cast(SetupErrorPayload, _build_error_payload(execution))


def helper_error_from_exception(err: Exception) -> HelperExecutionError | None:
    """Return a helper execution error carried by err, if any."""
    if isinstance(err, HelperExecutionError):
        return err
    cause = err.__cause__ or err.__context__
    if isinstance(cause, HelperExecutionError):
        return cause
    return None


def tool_error_envelope(
    key: str,
    placeholders: TranslationPlaceholders,
    *,
    message: str | None = None,
    guidance: Mapping[str, object] | None = None,
) -> JsonObjectType:
    """Return a JSON-safe recoverable error envelope for LLM tool callers."""
    error: dict[str, Any] = {
        "key": key,
        "message": _message_or_key_fallback(message or tool_error_message(key, placeholders), key),
    }
    # Guidance is omitted when unavailable to match execution error payloads.
    if guidance:
        error["guidance"] = dict(guidance)
    return cast(
        JsonObjectType,
        {
            "status": "error",
            "error": error,
        },
    )


def tool_error_from_exception(err: Exception) -> tuple[str, TranslationPlaceholders] | None:
    """Return a recoverable tool error from expected input/setup exceptions."""
    if isinstance(err, vol.Invalid):
        return "invalid_tool_input", {"error": str(err)}
    if isinstance(err, HomeAssistantError):
        key = str(getattr(err, "translation_key", None) or err.__class__.__name__)
        placeholders = getattr(err, "translation_placeholders", None)
        if not isinstance(placeholders, Mapping):
            return key, {}
        return key, _string_placeholders(cast(Mapping[str, object], placeholders))
    return None


def _setup_error_message(key: str, placeholders: TranslationPlaceholders) -> str | None:
    """Return a specific setup-error reason when the placeholders carry one."""
    if message := tool_error_message(key, placeholders):
        return message
    if reason := placeholders.get("error") or placeholders.get("reason"):
        return f"Setup failed: {reason}."
    return None


def _string_placeholders(placeholders: Mapping[str, object]) -> TranslationPlaceholders:
    """Normalize placeholder values to the string-only translation contract."""
    return {str(key): str(value) for key, value in placeholders.items()}


def _message_or_key_fallback(message: str | None, key: str | None) -> str:
    """Return a clean one-sentence message that never echoes the stable key."""
    clean = " ".join((message or "").split())
    if clean and clean != key:
        return clean
    if key:
        return f"Resolve the '{key}' error before retrying."
    return "Fix the sandbox error before retrying."


_TOOL_ERROR_MESSAGES: dict[str, _ToolErrorMessageBuilder] = {
    "monty_code_required": lambda _p: "Pass a non-empty code string.",
    "monty_code_too_long": lambda p: f"Shorten code to {p.get('max_length', 'the allowed')} characters or less.",
    "unknown_config_entry": lambda p: (
        f"Tool instance for config entry {p.get('config_entry_id', 'unknown')} cannot run because the entry is unknown or unloaded."
    ),
    "config_entry_not_loaded": lambda p: (
        f"Integration entry {p.get('config_entry_id', 'unknown')} must be loaded before retrying."
    ),
    "invalid_tool_input": lambda p: (
        f"Correct the tool arguments against the schema: {p.get('error')}."
        if p.get("error")
        else "Correct the tool arguments against the schema."
    ),
    "time_window_too_large": lambda p: (
        f"Requested time window is too large; shorten start/end or use hours <= {p.get('max_hours', 'the limit')}."
    ),
    "recorder_unavailable": lambda _p: (
        "Recorder-backed history/statistics/logbook queries require the recorder integration."
    ),
    "logbook_unavailable": lambda _p: "get_logbook requires the logbook integration.",
    "query_failed": lambda p: (
        f"Recorder query failed: {p.get('error')}." if p.get("error") else "Recorder query failed."
    ),
    "invalid_cursor": lambda _p: "Pagination cursor is invalid; restart pagination without cursor.",
    "analytics_unknown_op": lambda p: (
        f"Analytics operation '{p.get('op', 'unknown')}' is unsupported. Valid operations: {p.get('valid', 'none listed')}."
    ),
    "analytics_unknown_group_key": lambda p: (
        f"Analytics group key '{p.get('group_key', 'unknown')}' is unsupported. Valid group keys: {p.get('valid', 'none listed')}."
    ),
    "analytics_bad_bucket": lambda p: (
        f"Analytics bucket '{p.get('bucket', 'unknown')}' is invalid. Examples: {p.get('examples', '15m, 1h, 1d')}."
    ),
    "capture_failed": lambda p: (
        f"Capture failed for {p.get('entity_id', 'the requested entity')}: {p.get('error')}."
        if p.get("error")
        else f"Capture failed for {p.get('entity_id', 'the requested entity')}."
    ),
    "image_too_large": lambda p: (
        f"Captured image for {p.get('entity_id', 'the requested entity')} exceeds {p.get('max_bytes', 'the byte budget')} bytes at target_width {p.get('target_width', 'the requested width')}; retry with a smaller target_width."
    ),
    "sql_too_long": lambda p: f"Shorten the SQL query to {p.get('max_length', 'the allowed')} characters or less.",
    "sql_timeout": lambda _p: "SQL query timed out; narrow the query before retrying.",
}
