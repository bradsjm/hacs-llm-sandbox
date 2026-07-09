"""Scope tracking helpers for shadow-safe AST normalization."""

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto


class ScopeKind(Enum):
    """Kinds of Python scopes tracked by the rewrite engine."""

    MODULE = auto()
    FUNCTION = auto()
    LAMBDA = auto()
    CLASS = auto()
    COMPREHENSION = auto()
    HANDLER = auto()


@dataclass
class Scope:
    """A conservative set of tracked names bound in one lexical region."""

    kind: ScopeKind
    bound: set[str]


class ScopeStack:
    """Track whether sandbox globals and builtin names remain unshadowed."""

    def __init__(self, tracked: frozenset[str]) -> None:
        """Initialize with a module scope and the fixed tracked-name set."""
        self._tracked = tracked
        self._scopes: list[Scope] = [Scope(ScopeKind.MODULE, set())]

    def bind(self, names: Iterable[str]) -> None:
        """Bind tracked names in the current scope."""
        self._scopes[-1].bound.update(name for name in names if name in self._tracked)

    def bind_containing(self, names: Iterable[str]) -> None:
        """Bind tracked names in the nearest enclosing non-comprehension scope.

        Walrus targets inside comprehensions bind in the containing function or
        module scope per PEP 572, not in the transient comprehension scope.
        """
        for scope in reversed(self._scopes):
            if scope.kind is not ScopeKind.COMPREHENSION:
                scope.bound.update(name for name in names if name in self._tracked)
                return

    def push(self, kind: ScopeKind, preset: set[str]) -> None:
        """Push a new scope with an initial tracked-name binding set."""
        self._scopes.append(Scope(kind, {name for name in preset if name in self._tracked}))

    def pop(self) -> None:
        """Pop the current scope."""
        self._scopes.pop()

    def is_intact(self, name: str) -> bool:
        """Return false when a tracked name is bound in any active scope."""
        return all(name not in scope.bound for scope in self._scopes)


def target_names(node: ast.AST, tracked: frozenset[str]) -> set[str]:
    """Return tracked names bound by an assignment-style target."""
    if isinstance(node, ast.Name) and node.id in tracked:
        return {node.id}
    if isinstance(node, ast.Starred):
        return target_names(node.value, tracked)
    if isinstance(node, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in node.elts:
            names.update(target_names(element, tracked))
        return names
    return set()


def argument_names(args: ast.arguments, tracked: frozenset[str]) -> set[str]:
    """Return tracked names bound by function or lambda arguments."""
    names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs] if arg.arg in tracked}
    if args.vararg is not None and args.vararg.arg in tracked:
        names.add(args.vararg.arg)
    if args.kwarg is not None and args.kwarg.arg in tracked:
        names.add(args.kwarg.arg)
    return names


def import_names(aliases: list[ast.alias], tracked: frozenset[str]) -> set[str]:
    """Return tracked names bound by import aliases."""
    names: set[str] = set()
    for alias in aliases:
        name = alias.asname or alias.name.split(".", maxsplit=1)[0]
        if name in tracked:
            names.add(name)
    return names


def all_alias_names(aliases: list[ast.alias]) -> set[str]:
    """Return every name bound by import aliases, regardless of tracking.

    Used by the module-wide binding pre-scan so a non-datetime import that
    rebinds a name (e.g. ``import json as dt``) suppresses dropping a
    ``import datetime as dt`` whose alias it would otherwise shadow.
    """
    return {alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in aliases}


def function_locals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    tracked: frozenset[str],
) -> set[str]:
    """Pre-scan a function body for tracked local bindings."""
    shadowed = argument_names(node.args, tracked)
    collector = _FacadeBindingCollector(tracked)
    for statement in node.body:
        collector.visit(statement)
    shadowed.update(collector.names)
    return shadowed


class _FacadeBindingCollector(ast.NodeVisitor):
    """Collect function-local bindings without entering nested function bodies."""

    def __init__(self, tracked: frozenset[str]) -> None:
        self._tracked = tracked
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.update(target_names(ast.Name(node.name, ast.Store()), self._tracked))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.update(target_names(ast.Name(node.name, ast.Store()), self._tracked))

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.update(target_names(ast.Name(node.name, ast.Store()), self._tracked))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.names.update(target_names(target, self._tracked))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.names.update(target_names(node.target, self._tracked))
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.names.update(target_names(node.target, self._tracked))
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.names.update(target_names(node.target, self._tracked))
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.names.update(target_names(node.target, self._tracked))
        self.visit(node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.names.update(target_names(node.target, self._tracked))
        self.visit(node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(target_names(item.optional_vars, self._tracked))
            self.visit(item.context_expr)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(target_names(item.optional_vars, self._tracked))
            self.visit(item.context_expr)
        for statement in node.body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name in self._tracked:
            self.names.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(import_names(node.names, self._tracked))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(import_names(node.names, self._tracked))
