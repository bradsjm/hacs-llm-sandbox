"""Shared helper and runtime support for the Monty executor."""

import ast
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util.json import JsonValueType

from ..const import DEFAULT_HELPER_CALL_BUDGET, DOMAIN
from ..types import ActionRecord, TranslationPlaceholders
from .errors import (
    CodeErrorPayload,
    HelperErrorPayload,
    HelperExecutionError,
    RecoverableToolError,
    code_error_payload,
    helper_error_payload,
)
from .home_db import HomeDatabase


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
    # Internal forgiveness-layer labels. Payloads expose these as concise
    # adjustments that tell the model the change was already applied.
    normalizations: list[str] = field(default_factory=list)
    adjustments: list[dict[str, object]] = field(default_factory=list)
    # Captured print() output, one entry per call. Independent of the helper
    # call budget: print() routes through Monty's print_callback, not helper_response.
    printed: list[str] = field(default_factory=list)
    # Service action outcomes recorded by the services facade. Actions execute
    # sequentially, and prior successful entries remain when a later call fails.
    actions: list[ActionRecord] = field(default_factory=list)
    # Last helper validation error raised inside a facade method. Monty wraps
    # such exceptions into MontyRuntimeError without preserving __cause__, so
    # the executor recovers the structured error from here only when Monty's
    # generic wrapper carries the error's opaque per-run marker. Cleared at
    # the start of each helper call so a user try/except that swallows a
    # helper error cannot shadow a later helper-path failure.
    last_helper_error: HelperExecutionError | None = None
    # Per-run SQL database and transparency notes. The database is created lazily
    # by hass.query() and closed by executor cleanup before runtime context reset.
    home_db: HomeDatabase | None = None
    notes: list[str] = field(default_factory=list)


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
    except HelperExecutionError as err:
        state.last_helper_error = err
        raise
    except RecoverableToolError as err:
        helper_err = HelperExecutionError(helper, err.key, err.placeholders)
        state.last_helper_error = helper_err
        raise helper_err from err
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
    payload = helper_error_payload(
        err,
        message=_helper_error_message(err, state),
        kind=err.key,
        adjustments=list(state.adjustments),
        printed=list(state.printed),
        actions=cast(list[ActionRecord], json_safe(state.actions)),
    )
    if state.notes:
        payload["notes"] = list(state.notes)
    return payload


def code_error_payload_for_state(
    *,
    kind: str,
    message: str,
    state: ExecutionState,
    available_attributes: list[str] | None = None,
) -> CodeErrorPayload:
    """Build a code-execution error response using current state."""
    fix = available_attributes
    if kind == "NameError" and fix is None and "not defined" in message:
        from .contracts import AVAILABLE_GLOBALS

        fix = list(AVAILABLE_GLOBALS)
    payload = code_error_payload(
        kind=kind,
        message=message,
        adjustments=list(state.adjustments),
        printed=list(state.printed),
        actions=cast(list[ActionRecord], json_safe(state.actions)),
        fix=fix,
    )
    if state.notes:
        payload["notes"] = list(state.notes)
    return payload


def _helper_error_message(err: HelperExecutionError, state: ExecutionState) -> str:
    """Return one actionable sentence for a helper execution error."""
    if err.key == "call_budget_exceeded":
        return f"Stopped after {state.helper_call_limit} service calls; do not retry the same call."
    if reason := err.placeholders.get("reason"):
        return f"Fix the {err.helper} call failure: {reason}."
    return f"Resolve the {err.helper} error '{err.key}' before retrying."


type RefineResult = tuple[str, str, list[str] | None]
type ErrorRefiner = Callable[[str, str, str], RefineResult | None]


def refine_code_error(kind: str, message: str, code: str) -> RefineResult:
    """Refine Monty/type-check errors into actionable sandbox guidance.

    Runs the ordered ``REFINERS`` registry; the first rule that applies wins.
    A rule returns ``None`` to defer; when none apply, the cleaned message is
    returned unchanged so Monty's natural error surfaces. Internal class names
    are scrubbed and the known traps are reclassified into familiar Python
    error types with concrete next steps (imports, formatting, missing attrs).
    """
    # Monty appends informational lines that are noisy in LLM-facing payloads.
    cleaned_message = "\n".join(line for line in message.splitlines() if not line.strip().startswith("info:"))
    for refine in REFINERS:
        if (result := refine(kind, cleaned_message, code)) is not None:
            return result
    return kind, cleaned_message, None


def _refine_unresolved_reference(kind: str, message: str, code: str) -> RefineResult | None:
    """Convert Monty unresolved-reference wording into a familiar NameError."""
    if "unresolved-reference" not in message and "used when not defined" not in message:
        return None
    clean_message = _strip_monty_code_frame(message)
    if (name_match := re.search(r"`([A-Za-z_]\w*)`", message)) is None:
        # Guard matched but no backticked name: surface the cleaned message as-is.
        return kind, clean_message, None
    name = name_match.group(1)
    if name in {"dir", "vars"}:
        from .builtin_normalization import GLOBAL_TYPE_MAP, public_surface

        attributes = _attributes_for_first_discovery_call(code, name, GLOBAL_TYPE_MAP, public_surface)
        return (
            "NameError",
            f"`{name}` is not available in the sandbox; use the listed attributes or direct attribute access.",
            attributes,
        )
    if name in {"setattr", "delattr"}:
        return (
            "NameError",
            f"`{name}` is not available in the sandbox; use direct local values instead of mutating facade objects.",
            None,
        )
    if name == "__import__":
        return (
            "NameError",
            "`__import__` is not available in the sandbox; only json, math, re are importable as import statements. "
            "Use the pre-bound globals instead.",
            None,
        )
    return (
        "NameError",
        f"`{name}` is not defined; use an available sandbox global or assign it before use.",
        None,
    )


def _refine_unresolved_import(_kind: str, message: str, _code: str) -> RefineResult | None:
    """Only json/math/re are importable; guide toward built-in equivalents."""
    if (module_name := _extract_unresolved_import(message)) is None:
        return None
    return (
        "ImportError",
        f"`{module_name}` is not available in the sandbox; only json, math, re are importable. "
        "Use built-ins instead (e.g. sum()/len() for an average, a dict loop for counting).",
        None,
    )


def _refine_percent_format(_kind: str, message: str, _code: str) -> RefineResult | None:
    """Redirect % string formatting to f-strings."""
    if "unsupported operand type(s) for %" not in message:
        return None
    return (
        "TypeError",
        'Percent (%) string formatting is not available in the sandbox; use an f-string (e.g. f"{x}") instead.',
        None,
    )


def _refine_collection_dict_method(_kind: str, message: str, _code: str) -> RefineResult | None:
    """Guide dict-method misuse on list/tuple/set results (e.g. async_all().items())."""
    if re.search(r"'(list|tuple|set)' object has no attribute '(items|keys|values)'", message) is None:
        return None
    return (
        "AttributeError",
        "Facade read methods (states.async_all, registry .values()) return a list, not a dict; "
        "iterate it directly or with a comprehension (e.g. [s.entity_id for s in states.async_all()]) "
        "instead of .items()/.keys()/.values().",
        None,
    )


def _refine_none_deref(_kind: str, message: str, _code: str) -> RefineResult | None:
    """Guide method calls on None, typically a missing state or attribute."""
    if re.search(r"'NoneType' object has no attribute '\w+'", message) is None:
        return None
    return (
        "AttributeError",
        "The value is None — usually a missing state or attribute. Guard with `if value is not None:` "
        "before accessing it (e.g. check hass.states.get(...) for None first).",
        None,
    )


def _refine_missing_attribute(_kind: str, message: str, _code: str) -> RefineResult | None:
    """Surface the known public attributes for safe facade/record objects."""
    if (attr_match := re.search(r"'(\w+)' object has no attribute '(\w+)'", message)) is None:
        return None
    class_name = attr_match.group(1)
    attr = attr_match.group(2)
    if class_name == "str" and attr == "format":
        return (
            "AttributeError",
            'str.format() is not available in the sandbox; use an f-string (e.g. f"{x}") instead.',
            None,
        )
    from .builtin_normalization import surface_for_class_name

    # Scrub internal class names and surface the known public attribute set.
    if (surface := surface_for_class_name(class_name)) is not None:
        return "AttributeError", _scrub_class_name(message, class_name), sorted(surface)
    return "AttributeError", message, None


# Ordered rule registry consumed by ``refine_code_error``. Each rule returns a
# refined result when it applies or ``None`` to defer to the next rule. Append
# new recovery hints here (imports, attribute surfaces, formatting traps).
REFINERS: tuple[ErrorRefiner, ...] = (
    _refine_unresolved_reference,
    _refine_unresolved_import,
    _refine_percent_format,
    _refine_collection_dict_method,
    _refine_none_deref,
    _refine_missing_attribute,
)


def _extract_unresolved_import(message: str) -> str | None:
    """Return the module name from a Monty unresolved-import / ModuleNotFoundError message."""
    if "unresolved-import" not in message and "No module named" not in message:
        return None
    if match := re.search(
        r"Cannot resolve imported module ['`]([^'`]+)['`]|No module named ['\"]([^'\"]+)['\"]", message
    ):
        return str(match.group(1) or match.group(2))
    return "<module>"


def _strip_monty_code_frame(message: str) -> str:
    """Remove Monty source-frame lines from an otherwise useful diagnostic."""
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped or "llm_sandbox_agent.py" in stripped or stripped.startswith("-->"):
            continue
        if set(stripped) <= {"^", "|", " ", "-"}:
            continue
        lines.append(stripped)
    return " ".join(lines)


def _scrub_class_name(message: str, class_name: str) -> str:
    """Replace an internal facade/record class name with its friendly global reference."""
    if (friendly := _friendly_class_name(class_name)) is not None:
        return message.replace(f"'{class_name}'", f"'{friendly}'")
    return message


def _friendly_class_name(class_name: str) -> str | None:
    """Map an internal class name to the LLM-visible name it was accessed through."""
    if class_name in _RECORD_FRIENDLY_NAMES:
        return _RECORD_FRIENDLY_NAMES[class_name]
    from .builtin_normalization import GLOBAL_TYPE_MAP

    # Facades: prefer the longest global alias (long names read clearer than er/dr).
    aliases = [name for name, cls in GLOBAL_TYPE_MAP.items() if cls.__name__ == class_name]
    return max(aliases, key=len) if aliases else None


# Friendly references for types reached via attribute access or iteration rather
# than a bare global. Facade registry classes are derived from GLOBAL_TYPE_MAP.
_RECORD_FRIENDLY_NAMES: dict[str, str] = {
    "SafeHass": "hass",
    "SafeStateMachine": "states",
    "SafeServiceRegistry": "hass.services",
    "SafeConfig": "hass.config",
    "SafeLLMContext": "llm_context",
    "SafeState": "state",
    "SafeRegistryEntry": "entity entry",
    "SafeDeviceEntry": "device",
    "SafeAreaEntry": "area",
    "SafeFloorEntry": "floor",
    "SafeLabelEntry": "label",
    "SafeCategoryEntry": "category",
    "SafeIssueEntry": "issue",
    "SafeNotificationEntry": "notification",
    "SafeConfigEntry": "config entry",
    "SafeContext": "context",
    "SafeDate": "date value",
    "SafeDateTime": "datetime value",
}


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
