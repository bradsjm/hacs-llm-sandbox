"""Fail-open literal entity-id substitution from conversation memory."""

import ast
from dataclasses import dataclass
from typing import cast

from ..snapshot.models import HomeSnapshot
from .resolution_memory import ResolutionMemory


@dataclass(frozen=True, slots=True)
class ResolvedLiteral:
    """A code literal replaced by a remembered visible entity id."""

    requested: str
    applied: str


def substitute_remembered_literals(
    code: str,
    snapshot: HomeSnapshot,
    memory: ResolutionMemory | None,
) -> tuple[str, list[ResolvedLiteral]]:
    """Rewrite remembered missing ``states`` literals, returning original code on failure."""
    if memory is None:
        return code, []
    try:
        module = ast.parse(code)
        transformer = _RememberedLiteralTransformer(snapshot, memory)
        rewritten = transformer.visit(module)
        ast.fix_missing_locations(rewritten)
        if not transformer.resolutions:
            return code, []
        return ast.unparse(rewritten), transformer.resolutions
    except Exception:  # noqa: BLE001
        return code, []


class _RememberedLiteralTransformer(ast.NodeTransformer):
    """AST transformer for literal state reads guarded by fresh visibility."""

    def __init__(self, snapshot: HomeSnapshot, memory: ResolutionMemory) -> None:
        self.snapshot = snapshot
        self.memory = memory
        self.resolutions: list[ResolvedLiteral] = []
        self._seen: set[tuple[str, str]] = set()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Rewrite ``states.get('missing')`` and ``hass.states.get('missing')``."""
        updated = cast(ast.Call, self.generic_visit(node))
        if (
            isinstance(updated.func, ast.Attribute)
            and updated.func.attr == "get"
            and _states_root(updated.func.value)
            and updated.args
        ):
            updated.args[0] = self._replacement_for(updated.args[0])
        return updated

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Rewrite ``states['missing']`` and ``hass.states['missing']``."""
        updated = cast(ast.Subscript, self.generic_visit(node))
        if _states_root(updated.value):
            updated.slice = self._replacement_for(updated.slice)
        return updated

    def _replacement_for(self, node: ast.expr) -> ast.expr:
        """Return a visible remembered replacement for a string literal node."""
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            return node
        requested = node.value
        if requested in self.snapshot.states:
            return node
        applied = self.memory.lookup(requested)
        if applied is None or applied not in self.snapshot.states:
            return node
        resolution = (requested, applied)
        if resolution not in self._seen:
            self._seen.add(resolution)
            self.resolutions.append(ResolvedLiteral(requested=requested, applied=applied))
        return ast.copy_location(ast.Constant(value=applied), node)


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
