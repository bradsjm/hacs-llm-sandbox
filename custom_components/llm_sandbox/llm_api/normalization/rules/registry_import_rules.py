"""Home Assistant helper-registry import rewrite rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

REGISTRY_IMPORTS_RESOLVED = "registry_imports_resolved"

_REGISTRY_GLOBALS = {
    "area_registry": frozenset({"ar", "area_registry"}),
    "category_registry": frozenset({"cr", "category_registry"}),
    "device_registry": frozenset({"dr", "device_registry"}),
    "entity_registry": frozenset({"er", "entity_registry"}),
    "floor_registry": frozenset({"fr", "floor_registry"}),
    "label_registry": frozenset({"lr", "label_registry"}),
}


class RegistryImportRule:
    """Drop supported top-level helper-registry imports in favor of facades."""

    label = REGISTRY_IMPORTS_RESOLVED
    node_types = (ast.ImportFrom,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a pruned import when every dropped alias has a matching facade."""
        from_node = node if isinstance(node, ast.ImportFrom) else None
        if (
            from_node is None
            or not ctx.in_module_scope()
            or not ctx.is_direct_module_body_statement(from_node)
            or from_node.level != 0
            or from_node.module != "homeassistant.helpers"
        ):
            return None

        kept_aliases: list[ast.alias] = []
        changed = False
        for alias in from_node.names:
            bound_name = alias.asname or alias.name
            supported_names = _REGISTRY_GLOBALS.get(alias.name)
            # Keep imports whose binding cannot be proven to be an intact facade.
            if supported_names is None or bound_name not in supported_names or ctx.sandbox_global(bound_name) is None:
                kept_aliases.append(alias)
                continue
            changed = True

        if not changed:
            return None
        if not kept_aliases:
            return ast.Pass()
        # Preserve unsupported aliases so this rule changes only proven registry imports.
        from_node.names = kept_aliases
        return from_node
