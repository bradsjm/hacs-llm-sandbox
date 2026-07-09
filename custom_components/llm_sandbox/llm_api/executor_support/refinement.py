"""Monty error refinement and exception helpers."""

import ast
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import ServiceValidationError

from ...const import DOMAIN
from ...snapshot.models import HomeSnapshot
from ...types import TranslationPlaceholders
from ..guidance import FailureContext, Intent, advise

if TYPE_CHECKING:
    from .state import MontyFactory

type RefineResult = tuple[str, str, dict[str, object] | None]
type ErrorRefiner = Callable[[str, str, str, HomeSnapshot], RefineResult | None]


def refine_code_error(kind: str, message: str, code: str, snapshot: HomeSnapshot) -> RefineResult:
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
        if (result := refine(kind, cleaned_message, code, snapshot)) is not None:
            return result
    return kind, cleaned_message, None


def _refine_unresolved_reference(kind: str, message: str, code: str, snapshot: HomeSnapshot) -> RefineResult | None:
    """Convert Monty unresolved-reference wording into a familiar NameError."""
    if "unresolved-reference" not in message and "used when not defined" not in message:
        return None
    clean_message = _strip_monty_code_frame(message)
    if (name_match := re.search(r"`([A-Za-z_]\w*)`", message)) is None:
        # Guard matched but no backticked name: surface the cleaned message as-is.
        return kind, clean_message, None
    name = name_match.group(1)
    if name in {"dir", "vars"}:
        from ..facade_registry import GLOBAL_TYPE_MAP
        from ..normalization.surfaces import public_surface

        attributes = tuple(_attributes_for_first_discovery_call(code, name, GLOBAL_TYPE_MAP, public_surface) or ())
        guidance = None
        # Discovery helpers fail on a specific facade surface, so rank that surface when known.
        if attributes:
            guidance = advise(
                snapshot,
                FailureContext(intent=Intent.CODE_ATTRIBUTE, requested=name, available_attributes=attributes),
            ).to_payload()
        return (
            "NameError",
            f"`{name}` is not available in the sandbox; use the listed attributes or direct attribute access.",
            guidance,
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
    guidance = advise(snapshot, FailureContext(intent=Intent.CODE_NAME, requested=name)).to_payload()
    confidence = guidance.get("confidence")
    if confidence in {"exact", "high"}:
        return (
            "NameError",
            f"`{name}` is not defined; use an available sandbox global or assign it before use.",
            guidance,
        )
    from ..contracts import AVAILABLE_GLOBALS

    globals_hint = ", ".join(sorted(AVAILABLE_GLOBALS)[:8])
    return (
        "NameError",
        f"`{name}` is not defined; assign it before use. Available sandbox globals include: {globals_hint}.",
        None,
    )


def _refine_unresolved_import(_kind: str, message: str, _code: str, _snapshot: HomeSnapshot) -> RefineResult | None:
    """Only json/math/re are importable; guide toward built-in equivalents."""
    if (module_name := _extract_unresolved_import(message)) is None:
        return None
    return (
        "ImportError",
        f"`{module_name}` is not available in the sandbox; only json, math, re are importable. "
        "Use built-ins instead (e.g. sum()/len() for an average, a dict loop for counting).",
        None,
    )


def _refine_percent_format(_kind: str, message: str, _code: str, _snapshot: HomeSnapshot) -> RefineResult | None:
    """Redirect % string formatting to f-strings."""
    if "unsupported operand type(s) for %" not in message:
        return None
    return (
        "TypeError",
        'Percent (%) string formatting is not available in the sandbox; use an f-string (e.g. f"{x}") instead.',
        None,
    )


def _refine_collection_dict_method(
    _kind: str, message: str, _code: str, _snapshot: HomeSnapshot
) -> RefineResult | None:
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


def _refine_none_deref(_kind: str, message: str, _code: str, _snapshot: HomeSnapshot) -> RefineResult | None:
    """Guide method calls on None, typically a missing state or attribute."""
    if re.search(r"'NoneType' object has no attribute '\w+'", message) is None:
        return None
    return (
        "AttributeError",
        "The value is None — usually a missing state or attribute. Guard with `if value is not None:` "
        "before accessing it (e.g. check hass.states.get(...) for None first).",
        None,
    )


def _refine_hass_data(_kind: str, message: str, code: str, _snapshot: HomeSnapshot) -> RefineResult | None:
    """Guide HA-internal ``hass.data`` lookups toward sandbox read surfaces."""
    if "'hass' object has no attribute 'data'" not in message:
        return None
    for key in _hass_data_keys(code):
        # Branch boundary: only provide a domain-specific redirect when the
        # submitted key itself identifies the HA surface the model was seeking.
        if (hint := _hass_data_hint(key)) is not None:
            return (
                "AttributeError",
                f"hass.data is not available in the sandbox. {hint}",
                None,
            )
    return (
        "AttributeError",
        "hass.data is not available in the sandbox. Use the exposed read surfaces instead: "
        "states, registry facades (er/dr/ar/fr/lr/cr), repairs, persistent_notifications, "
        "config_entries, or the get_history/get_statistics/get_logbook tools.",
        None,
    )


def _refine_missing_attribute(_kind: str, message: str, _code: str, snapshot: HomeSnapshot) -> RefineResult | None:
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
    from ..normalization.surfaces import surface_for_class_name

    # Scrub internal class names and surface the known public attribute set.
    if (surface := surface_for_class_name(class_name)) is not None:
        guidance = advise(
            snapshot,
            FailureContext(intent=Intent.CODE_ATTRIBUTE, requested=attr, available_attributes=tuple(sorted(surface))),
        ).to_payload()
        return "AttributeError", _scrub_class_name(message, class_name), guidance
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
    _refine_hass_data,
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


def _hass_data_keys(code: str) -> tuple[str, ...]:
    """Return literal or identifier keys used with ``hass.data`` in submitted code."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return ()
    aliases = _hass_key_aliases(module)
    keys: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Subscript) and _is_hass_data_attribute(node.value):
            if (key := _literal_or_name(node.slice)) is not None:
                keys.append(aliases.get(key, key))
        elif (
            isinstance(node, ast.Call)
            and _is_hass_data_get_call(node)
            and node.args
            and (key := _literal_or_name(node.args[0])) is not None
        ):
            keys.append(aliases.get(key, key))
    return tuple(keys)


def _hass_key_aliases(module: ast.Module) -> dict[str, str]:
    """Resolve simple ``DOMAIN = 'x'; KEY = HassKey(DOMAIN)`` aliases."""
    string_names: dict[str, str] = {}
    for statement in module.body:
        for target in _assignment_names(statement):
            # State mutation point: remember literal constants before resolving HassKey aliases.
            if (
                (value := _assignment_value(statement)) is not None
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                string_names[target] = value.value

    aliases: dict[str, str] = {}
    for statement in module.body:
        value = _assignment_value(statement)
        if value is None or not isinstance(value, ast.Call) or not _is_hass_key_call(value) or not value.args:
            continue
        key = _literal_or_name(value.args[0])
        if key is None:
            continue
        for target in _assignment_names(statement):
            aliases[target] = string_names.get(key, key)
    return aliases


def _assignment_names(statement: ast.stmt) -> tuple[str, ...]:
    """Return simple variable names assigned by ``statement``."""
    if isinstance(statement, ast.Assign):
        return tuple(target.id for target in statement.targets if isinstance(target, ast.Name))
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return (statement.target.id,)
    return ()


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    """Return the assigned value expression for simple assignment statements."""
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        return statement.value
    return None


def _is_hass_key_call(node: ast.AST) -> bool:
    """Return whether ``node`` is a call to ``HassKey(...)``."""
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Name):
        return function.id == "HassKey"
    if isinstance(function, ast.Attribute):
        return function.attr == "HassKey"
    if isinstance(function, ast.Subscript):
        return _is_hass_key_call(ast.Call(func=function.value, args=[], keywords=[]))
    return False


def _is_hass_data_get_call(node: ast.Call) -> bool:
    """Return whether ``node`` calls ``hass.data.get(...)``."""
    function = node.func
    return isinstance(function, ast.Attribute) and function.attr == "get" and _is_hass_data_attribute(function.value)


def _is_hass_data_attribute(node: ast.AST) -> bool:
    """Return whether ``node`` is the attribute expression ``hass.data``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "data"
        and isinstance(node.value, ast.Name)
        and node.value.id == "hass"
    )


def _literal_or_name(node: ast.AST) -> str | None:
    """Return a string literal value or identifier name from a key expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _hass_data_hint(key: str) -> str | None:
    """Return a specific sandbox alternative when the hass.data key identifies one."""
    normalized = key.lower()
    if hint := _HASS_DATA_HINTS.get(normalized):
        return hint
    tokens = frozenset(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
    for expected_tokens, hint in _HASS_DATA_TOKEN_HINTS.items():
        if expected_tokens <= tokens:
            return hint
    return None


_HASS_DATA_GENERIC_HINT = (
    "That integration-internal storage is not exposed; use the available facade or recorder tool for the task."
)


_HASS_DATA_HINTS: dict[str, str] = {
    "persistent_notification": "Persistent notifications are read with "
    "persistent_notifications.async_get_notifications().",
    "persistent_notifications": "Persistent notifications are read with "
    "persistent_notifications.async_get_notifications().",
    "repair": "Repairs are read with repairs.async_active_issues() or repairs.async_issues().",
    "repairs": "Repairs are read with repairs.async_active_issues() or repairs.async_issues().",
    "issue_registry": "Repairs issues are read with repairs.async_active_issues() or repairs.async_issues().",
    "entity_registry": "Entity registry data is read with the er or entity_registry facade.",
    "device_registry": "Device registry data is read with the dr or device_registry facade.",
    "area_registry": "Area registry data is read with the ar or area_registry facade.",
    "floor_registry": "Floor registry data is read with the fr or floor_registry facade.",
    "label_registry": "Label registry data is read with the lr or label_registry facade.",
    "category_registry": "Category registry data is read with the cr or category_registry facade.",
    "recorder": "Recorder history is read with get_history or get_statistics.",
    "history": "History data is read with the get_history tool.",
    "statistics": "Statistics data is read with the get_statistics tool.",
    "logbook": "Logbook data is read with the get_logbook tool.",
    "config_entries": "Config entries are read with config_entries.async_entries('<domain>').",
    "automation": _HASS_DATA_GENERIC_HINT,
    "script": _HASS_DATA_GENERIC_HINT,
    "mqtt": _HASS_DATA_GENERIC_HINT,
    "zha": _HASS_DATA_GENERIC_HINT,
    "deconz": _HASS_DATA_GENERIC_HINT,
    "esphome": _HASS_DATA_GENERIC_HINT,
    "mobile_app": _HASS_DATA_GENERIC_HINT,
}

_HASS_DATA_TOKEN_HINTS: dict[frozenset[str], str] = {
    frozenset(key.split("_")): hint for key, hint in _HASS_DATA_HINTS.items()
}


def _friendly_class_name(class_name: str) -> str | None:
    """Map an internal class name to the LLM-visible name it was accessed through."""
    if class_name in _RECORD_FRIENDLY_NAMES:
        return _RECORD_FRIENDLY_NAMES[class_name]
    from ..facade_registry import GLOBAL_TYPE_MAP

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
    from .state import MontyFactory

    return cast(MontyFactory, pydantic_monty.Monty)


def validation_error(key: str, placeholders: TranslationPlaceholders) -> ServiceValidationError:
    """Create a localized service validation error for executor helpers."""
    return ServiceValidationError(
        translation_domain=DOMAIN,
        translation_key=key,
        translation_placeholders=placeholders,
    )
