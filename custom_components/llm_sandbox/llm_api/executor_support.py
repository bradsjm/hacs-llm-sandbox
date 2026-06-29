"""Shared helper and runtime support for the Monty executor."""

import ast
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.util.json import JsonValueType

from ..const import DEFAULT_HELPER_CALL_BUDGET, DOMAIN
from ..types import TranslationPlaceholders
from .errors import (
    CodeErrorPayload,
    HelperErrorPayload,
    HelperExecutionError,
    code_error_payload,
    helper_error_payload,
)


class MontyRunner(Protocol):
    """Runtime shape required from the optional Monty dependency."""

    async def run_async(
        self,
        *,
        inputs: Mapping[str, object],
        limits: pydantic_monty.ResourceLimits | None = None,
        external_functions: Mapping[str, object],
        print_callback: object = None,
    ) -> object:
        """Execute Monty code with JSON inputs and helper functions."""
        ...


class MontyFactory(Protocol):
    """Callable factory exposed as pydantic_monty.Monty."""

    def __call__(
        self,
        code: str,
        *,
        inputs: list[str],
        script_name: str,
        type_check: bool,
        type_check_stubs: str,
        dataclass_registry: list[type[object]] | None,
    ) -> MontyRunner:
        """Construct a runnable Monty program instance."""
        ...


@dataclass(slots=True)
class ExecutionState:
    """Mutable per-run bookkeeping for helper call budget enforcement."""

    helper_calls: int = 0
    helper_call_limit: int = DEFAULT_HELPER_CALL_BUDGET
    # Forgiveness-layer metadata, surfaced in the execution payload so the
    # LLM can see what was rewritten without forcing a retry.
    normalizations: list[str] = field(default_factory=list)
    # Captured print() output, one entry per call. Independent of the helper
    # call budget: print() routes through Monty's print_callback, not helper_response.
    printed: list[str] = field(default_factory=list)
    # Proposed service actions recorded by the propose-only services facade.
    # In the MVP (Option A) these are never executed; they surface in the
    # tool response payload so a caller can confirm/execute them later.
    proposed_actions: list[dict[str, object]] = field(default_factory=list)
    # Last helper validation error raised inside a facade method. Monty wraps
    # such exceptions into MontyRuntimeError without preserving __cause__, so
    # the executor recovers the structured error from here only when Monty's
    # generic wrapper carries the error's opaque per-run marker. Cleared at
    # the start of each helper call so a user try/except that swallows a
    # helper error cannot shadow a later helper-path failure.
    last_helper_error: HelperExecutionError | None = None


async def helper_response(
    state: ExecutionState,
    helper: str,
    callback: Callable[[], object],
    *,
    count_call: bool = True,
) -> JsonValueType:
    """Run one approved helper and return a raw JSON-safe result."""
    if count_call:
        state.helper_calls += 1
        if state.helper_calls > state.helper_call_limit:
            err = HelperExecutionError(helper, "call_budget_exceeded", {})
            state.last_helper_error = err
            raise err
    # Clear any prior helper error so a user try/except that swallowed a
    # previous helper error cannot shadow a later genuine code error.
    state.last_helper_error = None
    try:
        value = callback()
        if inspect.isawaitable(value):
            value = await cast(Awaitable[JsonValueType], value)
    except ServiceValidationError as err:
        helper_err = HelperExecutionError(
            helper,
            error_key(err),
            error_placeholders(err),
        )
        state.last_helper_error = helper_err
        raise helper_err from err
    return json_safe(value)


def json_safe(value: object) -> JsonValueType:
    """Convert arbitrary values into JSON-safe structures."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    sandbox_json = getattr(value, "__llm_sandbox_json__", None)
    if callable(sandbox_json):
        return json_safe(sandbox_json())
    if isinstance(value, Mapping):
        mapping_items = cast(Mapping[object, object], value)
        return {str(key): json_safe(item) for key, item in mapping_items.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        sequence_items = cast(Sequence[object], value)
        return [json_safe(item) for item in sequence_items]
    if isinstance(value, set):
        set_items = cast(set[object], value)
        return [json_safe(item) for item in set_items]
    return str(value)


def helper_error_payload_for_state(
    err: HelperExecutionError,
    state: ExecutionState,
) -> HelperErrorPayload:
    """Build a helper-error response using current execution state."""
    from .contracts import AVAILABLE_GLOBALS, suggested_methods

    return helper_error_payload(
        err,
        helper_calls=state.helper_calls,
        helper_call_limit=state.helper_call_limit,
        available_globals=list(AVAILABLE_GLOBALS),
        suggested_methods=suggested_methods(),
        normalizations=list(state.normalizations),
        printed=list(state.printed),
    )


def code_error_payload_for_state(
    *,
    kind: str,
    message: str,
    state: ExecutionState,
    location: dict[str, int] | None = None,
    available_attributes: list[str] | None = None,
) -> CodeErrorPayload:
    """Build a code-execution error response using current state."""
    from .contracts import AVAILABLE_GLOBALS, suggested_methods

    payload = code_error_payload(
        kind=kind,
        message=message,
        helper_calls=state.helper_calls,
        helper_call_limit=state.helper_call_limit,
        available_globals=list(AVAILABLE_GLOBALS),
        suggested_methods=suggested_methods(),
        normalizations=list(state.normalizations),
        printed=list(state.printed),
        available_attributes=available_attributes,
    )
    if location is not None:
        payload["execution"]["location"] = location
    return payload


def refine_code_error(kind: str, message: str, code: str) -> tuple[str, str, list[str] | None]:
    """Refine Monty/type-check errors into actionable sandbox guidance."""
    from .builtin_normalization import GLOBAL_TYPE_MAP, public_surface, surface_for_class_name

    # Monty appends informational lines that are noisy in LLM-facing payloads.
    cleaned_message = "\n".join(line for line in message.splitlines() if not line.strip().startswith("info:"))
    if "unresolved-reference" in cleaned_message or "used when not defined" in cleaned_message:
        # Convert Monty unresolved-reference wording into a familiar NameError.
        if (name_match := re.search(r"`([A-Za-z_]\w*)`", cleaned_message)) is None:
            return kind, cleaned_message, None
        name = name_match.group(1)
        available_attributes: list[str] | None = None
        if name in {"dir", "vars"}:
            cleaned_message = (
                f"`{name}` is not available in the sandbox; use the listed attributes or direct attribute access."
            )
            available_attributes = _attributes_for_first_discovery_call(code, name, GLOBAL_TYPE_MAP, public_surface)
        elif name in {"setattr", "delattr"}:
            cleaned_message = f"`{name}` is not available in the sandbox; use direct local values instead of mutating facade objects."
        return "NameError", cleaned_message, available_attributes

    if (attr_match := re.search(r"'(\w+)' object has no attribute '(\w+)'", cleaned_message)) is not None:
        # Surface the known public attributes for safe facade/record objects.
        class_name = attr_match.group(1)
        if (surface := surface_for_class_name(class_name)) is not None:
            return "AttributeError", cleaned_message, sorted(surface)
        return "AttributeError", cleaned_message, None

    return kind, cleaned_message, None


def _attributes_for_first_discovery_call(
    code: str,
    function_name: str,
    global_type_map: Mapping[str, type],
    surface_func: Callable[[type], frozenset[str]],
) -> list[str] | None:
    """Return attributes for the first ``dir``/``vars`` call on a facade global."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(module):
        # Only a single bare-global argument is eligible for discovery help.
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != function_name:
            continue
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
            continue
        if (cls := global_type_map.get(node.args[0].id)) is not None:
            return sorted(surface_func(cls))
    return None


def underlying_exception(err: Exception) -> Exception:
    """Return the concrete Python exception carried by a Monty runtime wrapper."""
    exception_factory = getattr(err, "exception", None)
    if not callable(exception_factory):
        return err
    try:
        inner = exception_factory()
    except Exception:  # noqa: BLE001 - diagnostics extraction must not mask the original error
        return err
    if isinstance(inner, Exception):
        return inner
    return err


def error_key(err: Exception) -> str:
    """Extract the stable translation key or fallback class name."""
    return str(getattr(err, "translation_key", None) or getattr(err, "key", None) or err.__class__.__name__)


def error_placeholders(err: Exception) -> TranslationPlaceholders:
    """Extract translation placeholders from Home Assistant errors."""
    placeholders = getattr(err, "translation_placeholders", None) or getattr(err, "placeholders", None)
    if not isinstance(placeholders, dict):
        return {}
    values = cast(Mapping[object, object], placeholders)
    return {str(key): str(value) for key, value in values.items()}


def load_monty_factory() -> MontyFactory:
    """Return the Monty factory.

    pydantic-monty is a required integration dependency (see manifest.json),
    imported at module load — do NOT switch to a dynamic import helper (HA's
    blocking-call detector instruments that path inside the event loop) and do
    NOT treat the dependency as optional.
    """
    return cast(MontyFactory, pydantic_monty.Monty)


def validation_error(key: str, placeholders: TranslationPlaceholders) -> ServiceValidationError:
    """Create a localized service validation error for executor helpers."""
    return ServiceValidationError(
        translation_domain=DOMAIN,
        translation_key=key,
        translation_placeholders=placeholders,
    )


def tool_setup_error(err: HomeAssistantError) -> None:
    """Marker function reserved for future setup-error mapping.

    Kept to preserve a stable import surface for callers that map setup errors
    to tool envelopes; the MVP routes HomeAssistantError through
    ``tool_error_from_exception`` directly.
    """
    raise err
