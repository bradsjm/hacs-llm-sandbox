"""Service-call target normalization rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

REWROTE_POSITIONAL_SERVICE_TARGET = "rewrote_positional_service_target"


class PositionalServiceTargetRule:
    """Move literal fourth positional service targets to the ``target`` keyword."""

    label = REWROTE_POSITIONAL_SERVICE_TARGET
    node_types = (ast.Call,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a keyword-target call when the target is unambiguously a dictionary."""
        call = node if isinstance(node, ast.Call) else None
        if (
            call is None
            or len(call.args) != 4
            or any(isinstance(argument, ast.Starred) for argument in call.args)
            or not isinstance(call.args[3], ast.Dict)
            or any(keyword.arg == "target" for keyword in call.keywords)
            or any(keyword.arg is None for keyword in call.keywords)
            or not _is_hass_services_async_call(call.func, ctx)
        ):
            return None

        target = call.args[3]
        # The original fourth positional expression runs before every keyword.
        # Keep that ordering while resolving the duplicate ``blocking=`` shape.
        return ast.Call(
            func=call.func,
            args=call.args[:3],
            keywords=[ast.keyword(arg="target", value=target), *call.keywords],
        )


def _is_hass_services_async_call(node: ast.AST, ctx: ResolutionContext) -> bool:
    """Return whether ``node`` is an intact ``hass.services.async_call`` attribute."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "async_call"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "services"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "hass"
        and ctx.sandbox_global("hass") is not None
    )
