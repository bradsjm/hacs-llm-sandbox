"""Await insertion and stripping rewrite rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

AWAITED_ASYNC_CALLS = "awaited_async_calls"
STRIPPED_AWAIT_FROM_SYNC = "stripped_await_from_sync"


class AwaitStripRule:
    """Strip awaits whose operand is provably synchronous."""

    label = STRIPPED_AWAIT_FROM_SYNC
    node_types = (ast.Await,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return the await operand when it is provably synchronous."""
        await_node = node if isinstance(node, ast.Await) else None
        if await_node is None or not _is_sync_operand(await_node.value, ctx):
            return None
        return await_node.value


class AwaitInsertRule:
    """Wrap missing awaits on known async facade method calls."""

    label = AWAITED_ASYNC_CALLS
    node_types = (ast.Call,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return an awaited call for unawaited async facade methods."""
        call = node if isinstance(node, ast.Call) else None
        if call is None or ctx.in_await_operand:
            return None
        if not isinstance(call.func, ast.Attribute):
            return None
        if call.func.attr not in ctx.async_methods:
            return None
        if not ctx.is_view_rooted(call.func.value):
            return None
        return ast.Await(value=call)


def _is_sync_operand(operand: ast.AST, ctx: ResolutionContext) -> bool:
    if isinstance(operand, ast.Name):
        return operand.id in ctx.sync_subscript_globals and ctx.sandbox_global(operand.id) is not None
    if isinstance(operand, ast.Attribute):
        return _is_sync_operand(operand.value, ctx)
    if isinstance(operand, ast.Subscript):
        return _is_sync_operand(operand.value, ctx)
    if isinstance(operand, ast.Call):
        if isinstance(operand.func, ast.Attribute) and operand.func.attr in ctx.sync_methods:
            return _is_sync_operand(operand.func.value, ctx)
        return False
    return False
