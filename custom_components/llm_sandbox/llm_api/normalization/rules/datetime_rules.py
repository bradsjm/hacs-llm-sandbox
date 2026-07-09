"""Datetime import and alias rewrite rules."""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import ResolutionContext

DATETIME_IMPORTS_RESOLVED = "datetime_imports_resolved"
_SUPPORTED_CLASS_NAMES = frozenset({"datetime", "date"})


class DatetimeImportRule:
    """Drop stdlib datetime module imports when a facade alias can replace them."""

    label = DATETIME_IMPORTS_RESOLVED
    node_types = (ast.Import,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a pruned import when datetime module aliases can be dropped."""
        import_node = node if isinstance(node, ast.Import) else None
        if import_node is None:
            return None
        kept: list[ast.alias] = []
        changed = False
        for alias in import_node.names:
            if alias.name != "datetime":
                kept.append(alias)
                continue
            bound_name = alias.asname or alias.name
            if not ctx.is_intact_name(bound_name) or ctx.datetime_alias_bound_anywhere(bound_name):
                kept.append(alias)
                continue
            # Dropped stdlib module aliases become rewrite aliases; kept imports
            # deliberately bind a live shadow and receive no alias mapping.
            ctx.record_dropped_datetime_alias(bound_name)
            changed = True
        if not changed:
            return None
        if not kept:
            return ast.Pass()
        import_node.names = kept
        return import_node


class DatetimeFromImportRule:
    """Rewrite supported datetime class imports to facade globals."""

    label = DATETIME_IMPORTS_RESOLVED
    node_types = (ast.ImportFrom,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return assignments and/or a pruned import-from for supported names."""
        from_node = node if isinstance(node, ast.ImportFrom) else None
        if from_node is None or from_node.module != "datetime":
            return None

        assignments: list[ast.stmt] = []
        kept_aliases: list[ast.alias] = []
        changed = False

        for alias in from_node.names:
            if alias.name not in _SUPPORTED_CLASS_NAMES:
                kept_aliases.append(alias)
                continue
            if alias.asname is not None:
                changed = True
                assignments.append(
                    ast.Assign(
                        targets=[ast.Name(id=alias.asname, ctx=ast.Store())],
                        value=ast.Name(id=alias.name, ctx=ast.Load()),
                    )
                )
                continue
            if ctx.sandbox_global(alias.name) is None:
                kept_aliases.append(alias)
                continue
            changed = True

        if not changed:
            return None
        if kept_aliases:
            from_node.names = kept_aliases
            if assignments:
                return ast.Module(body=[*assignments, from_node], type_ignores=[])
            return from_node
        if assignments:
            return ast.Module(body=assignments, type_ignores=[])
        return ast.Pass()


class DatetimeAttributeRule:
    """Rewrite dropped datetime module aliases to bare facade class globals."""

    label = DATETIME_IMPORTS_RESOLVED
    node_types = (ast.Attribute,)

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return the bare facade class name for dropped datetime aliases."""
        attr_node = node if isinstance(node, ast.Attribute) else None
        if attr_node is None or not isinstance(attr_node.value, ast.Name):
            return None
        if ctx.rewrite_alias(attr_node.value.id) != "datetime":
            return None
        if attr_node.attr not in _SUPPORTED_CLASS_NAMES:
            return None
        return ast.Name(id=attr_node.attr, ctx=ast.Load())
