"""Builtin convenience rewrite rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

TYPE_NAME_RESOLVED = "type_name_resolved"
WRAPPED_NEXT_ITER = "wrapped_next_iter"


class TypeNameRule:
    """Resolve ``type(<facade global>).__name__`` statically."""

    label = TYPE_NAME_RESOLVED
    node_types = (ast.Attribute,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a constant class name for safe facade type-name reflection."""
        attr_node = node if isinstance(node, ast.Attribute) else None
        if attr_node is None or attr_node.attr != "__name__":
            return None
        if not isinstance(attr_node.value, ast.Call):
            return None
        call = attr_node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "type":
            return None
        if not ctx.builtin_intact("type") or len(call.args) != 1 or call.keywords:
            return None
        if not isinstance(call.args[0], ast.Name):
            return None
        cls = ctx.sandbox_global(call.args[0].id)
        if cls is None:
            return None
        return ast.Constant(value=cls.__name__)


class NextIterRule:
    """Wrap ``next(x)`` as ``next(iter(x))`` for Monty list-backed iterables."""

    label = WRAPPED_NEXT_ITER
    node_types = (ast.Call,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a call with the first ``next`` argument wrapped in ``iter``."""
        call = node if isinstance(node, ast.Call) else None
        if call is None:
            return None
        if not isinstance(call.func, ast.Name) or call.func.id != "next":
            return None
        if call.keywords or len(call.args) not in {1, 2}:
            return None
        if not ctx.builtin_intact("next") or not ctx.builtin_intact("iter"):
            return None
        first = call.args[0]
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name) and first.func.id == "iter":
            return None
        return ast.Call(
            func=ast.Name(id="next", ctx=ast.Load()),
            args=[
                ast.copy_location(
                    ast.Call(func=ast.Name(id="iter", ctx=ast.Load()), args=[first], keywords=[]),
                    first,
                ),
                *call.args[1:],
            ],
            keywords=[],
        )
