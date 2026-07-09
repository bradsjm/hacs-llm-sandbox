"""Unified shadow-safe AST rewrite engine for sandbox normalization."""

import ast
import inspect
from typing import Any, Protocol, cast

from ..facade_registry import (
    AVAILABLE_GLOBALS,
    GLOBAL_TYPE_MAP,
    MONTY_DATACLASS_REGISTRY,
    SYNC_SUBSCRIPT_GLOBALS,
)
from .rules.await_rules import AwaitInsertRule, AwaitStripRule
from .rules.builtin_rules import NextIterRule, TypeNameRule
from .rules.datetime_rules import (
    DatetimeAttributeRule,
    DatetimeFromImportRule,
    DatetimeImportRule,
)
from .rules.state_sugar_rules import StateContainsRule, StateLenRule, StateSubscriptRule
from .scopes import (
    ScopeKind,
    ScopeStack,
    all_alias_names,
    argument_names,
    function_locals,
    import_names,
    target_names,
)

_BUILTIN_GLOBALS = frozenset({"type", "next", "iter", "len"})
_TRACKED_NAMES = frozenset(AVAILABLE_GLOBALS) | _BUILTIN_GLOBALS


class RewriteRule(Protocol):
    """Protocol implemented by AST rewrite rules."""

    label: str
    node_types: tuple[type[ast.AST], ...]

    def apply(self, node: ast.AST, ctx: ResolutionContext) -> ast.AST | None:
        """Return a replacement node, or ``None`` when the rule does not apply."""


def _classify_view_methods(view_classes: list[type[Any]]) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(async method names, sync method names)`` across the facades."""
    async_names: set[str] = set()
    sync_names: set[str] = set()
    for cls in view_classes:
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_") and name != "__getitem__":
                continue
            if inspect.iscoroutinefunction(member):
                async_names.add(name)
            else:
                sync_names.add(name)
    return frozenset(async_names), frozenset(sync_names)


_ASYNC_METHODS, _SYNC_METHODS = _classify_view_methods(MONTY_DATACLASS_REGISTRY)


class ResolutionContext:
    """Read-only resolution window exposed to rewrite rules."""

    def __init__(self, scopes: ScopeStack, module_bound_names: set[str]) -> None:
        """Initialize with the active scope stack and module-level binding scan."""
        self._scopes = scopes
        self._module_bound_names = module_bound_names
        # Two-tier datetime alias model: dropped stdlib module aliases are stored
        # for attribute rewrites, while kept imports bind shadows and are omitted.
        self._rewrite_aliases: dict[str, str] = {}
        self._pending_dropped_datetime_aliases: set[str] = set()
        self.async_methods = _ASYNC_METHODS
        self.sync_methods = _SYNC_METHODS
        self.sync_subscript_globals = SYNC_SUBSCRIPT_GLOBALS
        self.in_await_operand = False

    def is_intact_name(self, name: str) -> bool:
        """Return whether a name has no active tracked shadow binding."""
        return self._scopes.is_intact(name)

    def sandbox_global(self, name: str) -> type[Any] | None:
        """Return the facade global type only when the name is unshadowed."""
        if not self._scopes.is_intact(name):
            return None
        return GLOBAL_TYPE_MAP.get(name)

    def builtin_intact(self, name: str) -> bool:
        """Return whether a tracked builtin helper name is unshadowed."""
        return name in _BUILTIN_GLOBALS and self._scopes.is_intact(name)

    def is_view_rooted(self, node: ast.AST) -> bool:
        """Return whether an expression chain is rooted at an intact facade global."""
        if isinstance(node, ast.Name):
            return self.sandbox_global(node.id) is not None
        if isinstance(node, ast.Attribute | ast.Subscript):
            return self.is_view_rooted(node.value)
        return False

    def is_state_machine_root(self, node: ast.AST) -> bool:
        """Return whether a node is the pinned ``states`` or ``hass.states`` root."""
        if isinstance(node, ast.Name):
            return node.id == "states" and self.sandbox_global("states") is not None
        if isinstance(node, ast.Attribute):
            return (
                node.attr == "states"
                and isinstance(node.value, ast.Name)
                and node.value.id == "hass"
                and self.sandbox_global("hass") is not None
            )
        return False

    def rewrite_alias(self, name: str) -> str | None:
        """Return a dropped datetime module alias target if the alias is unshadowed."""
        if not self._scopes.is_intact(name):
            return None
        return self._rewrite_aliases.get(name)

    def record_dropped_datetime_alias(self, alias_name: str) -> None:
        """Record a datetime module alias dropped during import dispatch."""
        self._pending_dropped_datetime_aliases.add(alias_name)

    def commit_dropped_datetime_aliases(self) -> None:
        """Move pending dropped aliases into the active rewrite-alias map."""
        for alias_name in self._pending_dropped_datetime_aliases:
            self._rewrite_aliases[alias_name] = "datetime"
        self._pending_dropped_datetime_aliases.clear()

    def clear_dropped_datetime_aliases(self) -> None:
        """Clear pending alias records after a kept or failed import rewrite."""
        self._pending_dropped_datetime_aliases.clear()

    def datetime_alias_bound_anywhere(self, name: str) -> bool:
        """Return whether a datetime module alias is bound anywhere in the module.

        Dropping a ``import datetime as <name>`` is a whole-module safety
        decision, not an execution-order one: the alias must stay unshadowed
        everywhere, including rebinds (params, assignments, non-datetime
        imports) that appear textually *after* the import. The execution-order
        scope stack cannot see that at the import site, so the drop rule
        consults this flat module-wide pre-scan in addition to the live scope
        stack. Expression rewrites use the scope stack alone; only the import
        *drop* gate combines both notions.
        """
        return name in self._module_bound_names


class RewriteEngine(ast.NodeTransformer):
    """Apply ordered rewrite rules under one conservative shadow model."""

    def __init__(self, rules: tuple[RewriteRule, ...]) -> None:
        """Initialize the engine with an ordered rewrite rule registry."""
        self._rules = rules
        self._scopes = ScopeStack(_TRACKED_NAMES)
        self._ctx = ResolutionContext(self._scopes, set())
        self._applied: set[str] = set()

    def run(self, module: ast.Module) -> tuple[ast.Module, tuple[str, ...]]:
        """Rewrite a parsed module and return ordered unique labels."""
        self._ctx = ResolutionContext(self._scopes, _module_bound_names(module))
        new_module = self.visit(module)
        if not isinstance(new_module, ast.Module):
            return module, ()
        labels = tuple(dict.fromkeys(rule.label for rule in self._rules if rule.label in self._applied))
        return new_module, labels

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit top-level statements in execution order."""
        node.body = self._visit_statement_list(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit function definition metadata outside and body inside a function scope."""
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Visit async function definition metadata outside and body inside a function scope."""
        return self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        """Visit lambda defaults outside and body inside a lambda scope."""
        node.args = self.visit(node.args)
        self._scopes.push(ScopeKind.LAMBDA, argument_names(node.args, _TRACKED_NAMES))
        try:
            node.body = self.visit(node.body)
        finally:
            self._scopes.pop()
        return self._dispatch(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Bind the class name early and visit its body in a class scope."""
        self._scopes.bind({node.name})
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.bases = [self.visit(base) for base in node.bases]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        self._scopes.push(ScopeKind.CLASS, set())
        try:
            node.body = self._visit_statement_list(node.body)
        finally:
            self._scopes.pop()
        return self._dispatch(node)

    def visit_For(self, node: ast.For) -> ast.AST:
        """Visit loop iterator before target bindings shadow the body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._scopes.bind(target_names(node.target, _TRACKED_NAMES))
        node.body = self._visit_statement_list(node.body)
        node.orelse = self._visit_statement_list(node.orelse)
        return self._dispatch(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        """Visit async loop iterator before target bindings shadow the body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._scopes.bind(target_names(node.target, _TRACKED_NAMES))
        node.body = self._visit_statement_list(node.body)
        node.orelse = self._visit_statement_list(node.orelse)
        return self._dispatch(node)

    def visit_With(self, node: ast.With) -> ast.AST:
        """Visit each context expression before binding its optional target."""
        self._visit_with_items(node.items)
        node.body = self._visit_statement_list(node.body)
        return self._dispatch(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        """Visit async context expressions before binding optional targets."""
        self._visit_with_items(node.items)
        node.body = self._visit_statement_list(node.body)
        return self._dispatch(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        """Exception aliases shadow only inside the handler body."""
        if node.type is not None:
            node.type = self.visit(node.type)
        preset = {node.name} if node.name in _TRACKED_NAMES else set()
        self._scopes.push(ScopeKind.HANDLER, preset)
        try:
            node.body = self._visit_statement_list(node.body)
        finally:
            self._scopes.pop()
        return self._dispatch(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Visit assignment values before targets bind for following code."""
        node.value = self.visit(node.value)
        node.targets = [self.visit(target) for target in node.targets]
        current = self._dispatch(node)
        if isinstance(current, ast.Assign):
            for target in current.targets:
                self._scopes.bind(target_names(target, _TRACKED_NAMES))
        return current

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Visit annotated assignment metadata before binding the target."""
        node.annotation = self.visit(node.annotation)
        if node.value is not None:
            node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        current = self._dispatch(node)
        if isinstance(current, ast.AnnAssign):
            self._scopes.bind(target_names(current.target, _TRACKED_NAMES))
        return current

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Visit augmented assignment reads before binding future shadows."""
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        current = self._dispatch(node)
        if isinstance(current, ast.AugAssign):
            self._scopes.bind(target_names(current.target, _TRACKED_NAMES))
        return current

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        """Visit walrus values before targets bind in the enclosing PEP 572 scope."""
        node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        current = self._dispatch(node)
        if isinstance(current, ast.NamedExpr):
            self._scopes.bind_containing(target_names(current.target, _TRACKED_NAMES))
        return current

    def visit_Import(self, node: ast.Import) -> ast.AST | list[ast.stmt] | None:
        """Rewrite datetime imports before surviving aliases bind shadows."""
        current = self._dispatch(node)
        if isinstance(current, ast.Pass):
            self._ctx.commit_dropped_datetime_aliases()
            return None
        self._ctx.commit_dropped_datetime_aliases()
        if isinstance(current, ast.Import):
            self._scopes.bind(import_names(current.names, _TRACKED_NAMES))
        return current

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | list[ast.stmt] | None:
        """Rewrite datetime-from imports before surviving aliases bind shadows."""
        current = self._dispatch(node)
        self._ctx.clear_dropped_datetime_aliases()
        if isinstance(current, ast.Pass):
            return None
        statements = _module_body_or_statement(current)
        if statements is None:
            return current
        for statement in statements:
            if isinstance(statement, ast.ImportFrom):
                self._scopes.bind(import_names(statement.names, _TRACKED_NAMES))
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    self._scopes.bind(target_names(target, _TRACKED_NAMES))
        return statements

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        """Visit comprehensions in a non-leaking comprehension scope."""
        visited = self._visit_comprehension(node.generators, [node.elt])
        node.elt = visited[0]
        return self._dispatch(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        """Visit set comprehensions in a non-leaking comprehension scope."""
        visited = self._visit_comprehension(node.generators, [node.elt])
        node.elt = visited[0]
        return self._dispatch(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        """Visit dict comprehensions in a non-leaking comprehension scope."""
        visited = self._visit_comprehension(node.generators, [node.key, node.value])
        node.key = visited[0]
        node.value = visited[1]
        return self._dispatch(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        """Visit generator expressions in a non-leaking comprehension scope."""
        visited = self._visit_comprehension(node.generators, [node.elt])
        node.elt = visited[0]
        return self._dispatch(node)

    def visit_Global(self, node: ast.Global) -> ast.AST:
        """Treat global declarations as conservative local shadows."""
        self._scopes.bind(node.names)
        return self._dispatch(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        """Treat nonlocal declarations as conservative local shadows."""
        self._scopes.bind(node.names)
        return self._dispatch(node)

    def visit_Await(self, node: ast.Await) -> ast.AST:
        """Visit await operands under the in-await flag, then strip if sync."""
        old = self._ctx.in_await_operand
        self._ctx.in_await_operand = True
        try:
            node.value = self.visit(node.value)
        finally:
            self._ctx.in_await_operand = old
        return self._dispatch(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Recurse into ordinary nodes, then dispatch rules post-order."""
        return self._dispatch(super().generic_visit(node))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._scopes.bind({node.name})
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.args = self.visit(node.args)
        node.returns = self.visit(node.returns) if node.returns is not None else None
        self._scopes.push(ScopeKind.FUNCTION, function_locals(node, _TRACKED_NAMES))
        try:
            node.body = self._visit_statement_list(node.body)
        finally:
            self._scopes.pop()
        return self._dispatch(node)

    def _visit_with_items(self, items: list[ast.withitem]) -> None:
        for item in items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
                self._scopes.bind(target_names(item.optional_vars, _TRACKED_NAMES))

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        elements: list[ast.expr],
    ) -> list[ast.expr]:
        if not generators:
            return [cast(ast.expr, self.visit(element)) for element in elements]
        generators[0].iter = self.visit(generators[0].iter)
        targets: set[str] = set()
        for generator in generators:
            targets.update(target_names(generator.target, _TRACKED_NAMES))
        self._scopes.push(ScopeKind.COMPREHENSION, targets)
        try:
            for index, generator in enumerate(generators):
                generator.target = self.visit(generator.target)
                if index > 0:
                    generator.iter = self.visit(generator.iter)
                generator.ifs = [self.visit(condition) for condition in generator.ifs]
            return [cast(ast.expr, self.visit(element)) for element in elements]
        finally:
            self._scopes.pop()

    def _visit_statement_list(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        new_body: list[ast.stmt] = []
        for statement in statements:
            replacement = self.visit(statement)
            if replacement is None:
                continue
            if isinstance(replacement, list):
                new_body.extend(replacement)
            elif isinstance(replacement, ast.Module):
                new_body.extend(replacement.body)
            elif isinstance(replacement, ast.stmt):
                new_body.append(replacement)
        return new_body

    def _dispatch(self, node: ast.AST) -> ast.AST:
        current = node
        for rule in self._rules:
            if not isinstance(current, rule.node_types):
                continue
            replacement = rule.apply(current, self._ctx)
            if replacement is None:
                continue
            current = ast.copy_location(replacement, current)
            self._applied.add(rule.label)
        return current


def _module_body_or_statement(node: ast.AST) -> list[ast.stmt] | None:
    if isinstance(node, ast.Module):
        return node.body
    if isinstance(node, ast.stmt):
        return [node]
    return None


def _module_bound_names(module: ast.Module) -> set[str]:
    collector = _ModuleBindingCollector()
    collector.visit(module)
    return collector.names


class _ModuleBindingCollector(ast.NodeVisitor):
    """Flat module-wide binding scan backing ``datetime_alias_bound_anywhere``.

    Mirrors the old ``_LocalBoundNames`` conservative model: every name bound
    anywhere (Store targets, params, except aliases, def/class names, and
    non-datetime import aliases) makes a datetime module import drop unsafe.
    See ``datetime_alias_bound_anywhere`` for why the drop gate needs this in
    addition to the execution-order scope stack.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Collect non-datetime import aliases that may rebind dropped aliases."""
        self.names.update(
            alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names if alias.name != "datetime"
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Collect non-datetime from-import aliases that may rebind dropped aliases."""
        if node.module != "datetime":
            self.names.update(all_alias_names(node.names))


RULES: tuple[RewriteRule, ...] = cast(
    tuple[RewriteRule, ...],
    (
        DatetimeImportRule(),
        DatetimeFromImportRule(),
        DatetimeAttributeRule(),
        TypeNameRule(),
        NextIterRule(),
        StateSubscriptRule(),
        StateContainsRule(),
        StateLenRule(),
        AwaitStripRule(),
        AwaitInsertRule(),
    ),
)
