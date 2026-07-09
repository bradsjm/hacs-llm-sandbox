"""State-machine sugar rewrite rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

REWROTE_SYNC_SUBSCRIPT = "rewrote_sync_subscript"


class StateSubscriptRule:
    """Rewrite ``states['id']`` to ``states.get('id')``."""

    label = REWROTE_SYNC_SUBSCRIPT
    node_types = (ast.Subscript,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a ``get`` call for a safe state-machine subscript."""
        subscript = node if isinstance(node, ast.Subscript) else None
        if subscript is None or isinstance(subscript.slice, ast.Slice):
            return None
        if not ctx.is_state_machine_root(subscript.value):
            return None
        return ast.Call(
            func=ast.Attribute(value=subscript.value, attr="get", ctx=ast.Load()),
            args=[subscript.slice],
            keywords=[],
        )


class StateContainsRule:
    """Rewrite state-machine containment to ``get(...) is [not] None``."""

    label = REWROTE_SYNC_SUBSCRIPT
    node_types = (ast.Compare,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a ``get``/``None`` comparison for containment sugar."""
        compare = node if isinstance(node, ast.Compare) else None
        if compare is None or len(compare.ops) != 1 or len(compare.comparators) != 1:
            return None
        root = compare.comparators[0]
        if not ctx.is_state_machine_root(root):
            return None
        comparator: ast.cmpop
        if isinstance(compare.ops[0], ast.In):
            comparator = ast.IsNot()
        elif isinstance(compare.ops[0], ast.NotIn):
            comparator = ast.Is()
        else:
            return None
        return ast.Compare(
            left=ast.Call(
                func=ast.Attribute(value=root, attr="get", ctx=ast.Load()),
                args=[compare.left],
                keywords=[],
            ),
            ops=[comparator],
            comparators=[ast.Constant(value=None)],
        )


class StateLenRule:
    """Rewrite ``len(states)`` to the state-machine entity-id count."""

    label = REWROTE_SYNC_SUBSCRIPT
    node_types = (ast.Call,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a ``len(async_entity_ids())`` call for state roots."""
        call = node if isinstance(node, ast.Call) else None
        if call is None or not isinstance(call.func, ast.Name) or call.func.id != "len":
            return None
        if len(call.args) != 1 or call.keywords or not ctx.builtin_intact("len"):
            return None
        if not ctx.is_state_machine_root(call.args[0]):
            return None
        return ast.Call(
            func=ast.Name(id="len", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Attribute(value=call.args[0], attr="async_entity_ids", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            ],
            keywords=[],
        )
