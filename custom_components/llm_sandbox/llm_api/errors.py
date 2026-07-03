"""Structured error payload helpers for LLM Sandbox LLM tools."""

from collections.abc import Mapping
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
    fix: NotRequired[list[str]]
    adjustments: NotRequired[list[dict[str, object]]]


class CodeErrorExecutionPayload(TypedDict):
    """Execution metadata for Monty code failures."""

    status: Literal["code_error"]
    message: str
    kind: NotRequired[str]
    fix: NotRequired[list[str]]
    adjustments: NotRequired[list[dict[str, object]]]


class SetupErrorExecutionPayload(TypedDict):
    """Execution metadata for pre-execution setup failures."""

    status: Literal["setup_error"]
    message: str
    kind: NotRequired[str]
    fix: NotRequired[list[str]]


class HelperErrorPayload(TypedDict):
    """Top-level helper-error execution response."""

    execution: HelperErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]


class CodeErrorPayload(TypedDict):
    """Top-level code-error execution response."""

    execution: CodeErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]


class SetupErrorPayload(TypedDict):
    """Top-level setup-error execution response."""

    execution: SetupErrorExecutionPayload
    output: None
    printed: NotRequired[list[str]]
    actions: NotRequired[list[ActionRecord]]


@dataclass(frozen=True, slots=True)
class HelperExecutionError(Exception):
    """Recoverable helper failure surfaced as code-mode execution metadata."""

    helper: str
    key: str
    placeholders: TranslationPlaceholders
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


def helper_error_payload(
    err: HelperExecutionError,
    *,
    message: str,
    kind: str | None = None,
    fix: list[str] | None = None,
    adjustments: list[dict[str, object]],
    printed: list[str],
    actions: list[ActionRecord] | None = None,
) -> HelperErrorPayload:
    """Return compact helper-error execution payload."""
    execution: HelperErrorExecutionPayload = {
        "status": "helper_error",
        "message": _message_or_key_fallback(message, err.key),
    }
    if kind:
        execution["kind"] = kind
    if fix:
        execution["fix"] = fix
    if adjustments:
        execution["adjustments"] = adjustments
    payload: HelperErrorPayload = {
        "execution": execution,
        "output": None,
    }
    if printed:
        payload["printed"] = printed
    if actions:
        payload["actions"] = actions
    return payload


def code_error_payload(
    *,
    kind: str | None,
    message: str,
    adjustments: list[dict[str, object]],
    printed: list[str],
    actions: list[ActionRecord] | None = None,
    fix: list[str] | None = None,
) -> CodeErrorPayload:
    """Return compact code-error execution payload."""
    execution: CodeErrorExecutionPayload = {
        "status": "code_error",
        "message": _message_or_key_fallback(message, kind),
    }
    if kind:
        execution["kind"] = kind
    if fix:
        execution["fix"] = fix
    if adjustments:
        execution["adjustments"] = adjustments
    payload: CodeErrorPayload = {
        "execution": execution,
        "output": None,
    }
    if printed:
        payload["printed"] = printed
    if actions:
        payload["actions"] = actions
    return payload


def setup_error_payload(key: str, placeholders: TranslationPlaceholders) -> SetupErrorPayload:
    """Return compact setup-error execution payload."""
    execution: SetupErrorExecutionPayload = {
        "status": "setup_error",
        "message": _message_or_key_fallback(_setup_error_message(placeholders), key),
    }
    return {"execution": execution, "output": None}


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
    _placeholders: TranslationPlaceholders,
    *,
    message: str | None = None,
    fix: list[str] | None = None,
) -> JsonObjectType:
    """Return a JSON-safe recoverable error envelope for LLM tool callers."""
    error: dict[str, Any] = {
        "key": key,
        "message": _message_or_key_fallback(message, key),
    }
    if fix:
        error["fix"] = fix
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


def _setup_error_message(placeholders: TranslationPlaceholders) -> str | None:
    """Return a specific setup-error reason when the placeholders carry one."""
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
