"""Structured error payload helpers for LLM Sandbox LLM tools."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, cast
from uuid import uuid4

import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.json import JsonObjectType

from ..types import TranslationPlaceholders


class HelperErrorExecutionPayload(TypedDict):
    """Execution metadata for helper validation failures."""

    status: Literal["helper_error"]
    helper: str
    code: str
    placeholders: TranslationPlaceholders
    message: str
    helper_calls: int
    helper_call_limit: int
    available_globals: list[str]
    suggested_methods: list[str]
    # Forgiveness-layer metadata so the LLM can see what was auto-rewritten.
    normalizations: list[str]


class CodeErrorExecutionPayload(TypedDict):
    """Execution metadata for Monty code failures."""

    status: Literal["code_error"]
    kind: str
    message: str
    helper_calls: int
    helper_call_limit: int
    available_globals: list[str]
    suggested_methods: list[str]
    normalizations: list[str]
    location: NotRequired[dict[str, int]]


class SetupErrorExecutionPayload(TypedDict):
    """Execution metadata for pre-execution setup failures."""

    status: Literal["setup_error"]
    code: str
    message: str
    placeholders: TranslationPlaceholders


class HelperErrorPayload(TypedDict):
    """Top-level helper-error execution response."""

    execution: HelperErrorExecutionPayload
    output: None
    printed: list[str]


class CodeErrorPayload(TypedDict):
    """Top-level code-error execution response."""

    execution: CodeErrorExecutionPayload
    output: None
    printed: list[str]


class SetupErrorPayload(TypedDict):
    """Top-level setup-error execution response."""

    execution: SetupErrorExecutionPayload
    output: None
    printed: list[str]


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


def helper_error_payload(
    err: HelperExecutionError,
    *,
    helper_calls: int,
    helper_call_limit: int,
    available_globals: list[str],
    suggested_methods: list[str],
    normalizations: list[str],
    printed: list[str],
) -> HelperErrorPayload:
    """Return compact helper-error execution payload."""
    return {
        "execution": {
            "status": "helper_error",
            "helper": err.helper,
            "code": err.key,
            "placeholders": err.placeholders,
            "message": err.key,
            "helper_calls": helper_calls,
            "helper_call_limit": helper_call_limit,
            "available_globals": available_globals,
            "suggested_methods": suggested_methods,
            "normalizations": normalizations,
        },
        "output": None,
        "printed": printed,
    }


def code_error_payload(
    *,
    kind: str,
    message: str,
    helper_calls: int,
    helper_call_limit: int,
    available_globals: list[str],
    suggested_methods: list[str],
    normalizations: list[str],
    printed: list[str],
) -> CodeErrorPayload:
    """Return compact code-error execution payload."""
    return {
        "execution": {
            "status": "code_error",
            "kind": kind,
            "message": message,
            "helper_calls": helper_calls,
            "helper_call_limit": helper_call_limit,
            "available_globals": available_globals,
            "suggested_methods": suggested_methods,
            "normalizations": normalizations,
        },
        "output": None,
        "printed": printed,
    }


def setup_error_payload(key: str, placeholders: TranslationPlaceholders) -> SetupErrorPayload:
    """Return compact setup-error execution payload."""
    return {
        "execution": {
            "status": "setup_error",
            "code": key,
            "message": key,
            "placeholders": _string_placeholders(placeholders),
        },
        "output": None,
        "printed": [],
    }


def helper_error_from_exception(err: Exception) -> HelperExecutionError | None:
    """Return a helper execution error carried by err, if any."""
    if isinstance(err, HelperExecutionError):
        return err
    cause = err.__cause__ or err.__context__
    if isinstance(cause, HelperExecutionError):
        return cause
    return None


def tool_error_envelope(key: str, placeholders: TranslationPlaceholders) -> JsonObjectType:
    """Return a JSON-safe recoverable error envelope for LLM tool callers."""
    return cast(
        JsonObjectType,
        {
            "status": "error",
            "error": {
                "key": key,
                "placeholders": _string_placeholders(placeholders),
            },
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


def _string_placeholders(placeholders: Mapping[str, object]) -> TranslationPlaceholders:
    """Normalize placeholder values to the string-only translation contract."""
    return {str(key): str(value) for key, value in placeholders.items()}
