"""Success-side legacy and repair notes for ``execute_home_code`` payloads.

The executor keeps the public response shape as a single optional ``note``
string. Internally, this module owns an ordered checker registry so future
legacy Home Assistant patterns can be added without one-off executor branches.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass

from ..snapshot.models import HomeSnapshot
from .resolution import _DISCOVERY_LIMIT, bounded_strings, candidates_for_domain, resolve_target_entity


@dataclass(frozen=True, slots=True)
class LegacyNoteContext:
    """Inputs available to success-side legacy note checkers."""

    code: str
    snapshot: HomeSnapshot
    result: object


type LegacyNoteChecker = Callable[[LegacyNoteContext], str | None]

_FORECAST_ATTRIBUTE_NOTE = (
    "Weather forecasts are no longer exposed as the state attribute 'forecast'. "
    "Use weather.get_forecasts via hass.services.async_call('weather', 'get_forecasts', {'type': 'daily'}, "
    "target={'entity_id': entity_id}, blocking=True, return_response=True), then read response[entity_id]['forecast']."
)


def compute_legacy_note(ctx: LegacyNoteContext) -> str | None:
    """Return the first applicable success-side legacy note, if any."""
    for checker in LEGACY_NOTE_CHECKERS:
        if (note := checker(ctx)) is not None:
            return note
    return None


def _forecast_attribute_checker(ctx: LegacyNoteContext) -> str | None:
    """Guide removed weather ``forecast`` state-attribute access to the service API."""
    if _references_forecast_attribute(ctx.code):
        return _FORECAST_ATTRIBUTE_NOTE
    return None


def _missing_state_checker(ctx: LegacyNoteContext) -> str | None:
    """Preserve the existing empty-result note for literal missing state ids."""
    referenced_missing = _referenced_missing(ctx.code, ctx.snapshot)
    if referenced_missing and _is_empty_output(ctx.result):
        return _missing_state_note(ctx.snapshot, referenced_missing[0])
    return None


LEGACY_NOTE_CHECKERS: tuple[LegacyNoteChecker, ...] = (
    _forecast_attribute_checker,
    _missing_state_checker,
)


def _references_forecast_attribute(code: str) -> bool:
    """Return True when code reads the removed ``attributes['forecast']`` value."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        _is_attributes_get_forecast(node) or _is_attributes_subscript_forecast(node) for node in ast.walk(module)
    )


def _is_attributes_get_forecast(node: ast.AST) -> bool:
    """Whether ``node`` is ``attributes.get('forecast')`` or ``*.attributes.get('forecast')``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_attributes_expression(node.func.value)
        and bool(node.args)
        and _literal_str(node.args[0]) == "forecast"
    )


def _is_attributes_subscript_forecast(node: ast.AST) -> bool:
    """Whether ``node`` is ``attributes['forecast']`` or ``*.attributes['forecast']``."""
    return (
        isinstance(node, ast.Subscript)
        and _is_attributes_expression(node.value)
        and _literal_str(node.slice) == "forecast"
    )


def _is_attributes_expression(node: ast.AST) -> bool:
    """Return True for the bare ``attributes`` name or an attribute named ``attributes``."""
    return (isinstance(node, ast.Name) and node.id == "attributes") or (
        isinstance(node, ast.Attribute) and node.attr == "attributes"
    )


def _is_empty_output(result: object) -> bool:
    """True for ``None`` or an empty collection, but not scalar 0/False outputs."""
    if result is None:
        return True
    if isinstance(result, list | dict | tuple | str | set):
        return len(result) == 0
    return False


def _states_root(node: ast.AST) -> bool:
    """Whether ``node`` is the ``states`` or ``hass.states`` expression."""
    if isinstance(node, ast.Name):
        return node.id == "states"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "states"
        and isinstance(node.value, ast.Name)
        and node.value.id == "hass"
    )


def _referenced_state_ids(code: str) -> list[str]:
    """Entity ids read via literal ``states.get``/``states[...]``.

    Static analysis: Monty copies input objects and does not propagate the
    runtime contextvar to synchronous methods, so absent reads cannot be
    recorded inside ``states.get``. Scanning the submitted code for literal
    entity-id reads catches the dominant case (an LLM-typed integration id) and
    lets the executor self-describe available entities on an empty result.
    """
    try:
        module = ast.parse(code)
    except SyntaxError:
        return []
    references: list[str] = []
    seen: set[str] = set()

    def _consider(literal: object) -> None:
        if isinstance(literal, str) and literal not in seen:
            seen.add(literal)
            references.append(literal)

    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and _states_root(node.func.value) and node.args:
                _consider(_literal_str(node.args[0]))
        elif isinstance(node, ast.Subscript) and _states_root(node.value):
            _consider(_literal_str(node.slice))
    return references


def _referenced_missing(code: str, snapshot: HomeSnapshot) -> list[str]:
    """Entity ids read via literal state access that are absent from the snapshot."""
    return [entity_id for entity_id in _referenced_state_ids(code) if entity_id not in snapshot.states]


def _referenced_visible_state_id(code: str, snapshot: HomeSnapshot) -> str | None:
    """Return the only visible literal state id read via ``states.get``/``states[...]``."""
    visible = [entity_id for entity_id in _referenced_state_ids(code) if entity_id in snapshot.states]
    return visible[0] if len(visible) == 1 else None


def _literal_str(node: ast.AST) -> object:
    """Return the Python value of a string-literal AST node, else a sentinel."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return object()  # non-string / non-literal: never matches an entity id


def _entity_candidates(snapshot: HomeSnapshot, requested_entity_id: str) -> list[str]:
    """Return token-ranked visible entity candidates for a missing state id."""
    domain = requested_entity_id.split(".", 1)[0] if "." in requested_entity_id else requested_entity_id
    outcome = resolve_target_entity(snapshot, requested_entity_id, domain)
    if outcome.resolved is not None:
        return [outcome.resolved]
    candidates = outcome.candidates or candidates_for_domain(snapshot, domain, limit=_DISCOVERY_LIMIT + 1)
    return bounded_strings([candidate.entity_id for candidate in candidates])


def _missing_state_note(snapshot: HomeSnapshot, requested_entity_id: str) -> str:
    """Return an imperative empty-result repair note for a missing state id."""
    candidates = _entity_candidates(snapshot, requested_entity_id)
    if candidates and candidates[0] != "...":
        return f"No data: '{requested_entity_id}' does not exist. Use '{candidates[0]}' and re-run."
    return f"No data: '{requested_entity_id}' does not exist and no visible replacement was found."
