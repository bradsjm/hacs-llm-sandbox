"""AST normalization that forgives missing or extraneous ``await`` keywords.

The Monty runtime type-checks against stubs. When an LLM forgets the
``await`` (or adds one where none is needed) the type check fails and the
LLM is forced into a retry loop. This module rewrites the source so a known
async method call without an ``await`` is wrapped, and an ``await`` over a
known sync subscript on a facade is stripped — before Monty ever sees the
code.

The classification of async vs sync methods is derived mechanically from the
dataclass registry passed to Monty (see ``async_execute_home_code``). No
hand-maintained allowlist is required.
"""

import ast
import inspect

from ..facade_registry import SYNC_SUBSCRIPT_GLOBALS

AWAITED_ASYNC_CALLS = "awaited_async_calls"
STRIPPED_AWAIT_FROM_SYNC = "stripped_await_from_sync"
REWROTE_SYNC_SUBSCRIPT = "rewrote_sync_subscript"

_SYNC_SUBSCRIPT_GLOBALS = SYNC_SUBSCRIPT_GLOBALS
_VIEW_GLOBALS = _SYNC_SUBSCRIPT_GLOBALS


def normalize_awaits(code: str, view_classes: list[type]) -> tuple[str, list[str]]:
    """Add missing awaits and strip extraneous awaits for known facade methods.

    Returns the (possibly unchanged) code and a sorted list of normalization
    labels applied. Fails open on ``SyntaxError``: bad input is returned
    unchanged so Monty surfaces the natural code error instead.
    """
    async_method_names, sync_method_names = _classify_view_methods(view_classes)
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, []

    subscript_rewriter = _SubscriptRewriter()
    subscript_module = subscript_rewriter.visit(module)

    stripper = _AwaitStripper(sync_method_names)
    stripped_module = stripper.visit(subscript_module)

    wrapper = _AwaitWrapper(async_method_names)
    wrapped_module = wrapper.visit(stripped_module)

    applied: set[str] = set()
    if subscript_rewriter.rewrote:
        applied.add(REWROTE_SYNC_SUBSCRIPT)
    if stripper.stripped:
        applied.add(STRIPPED_AWAIT_FROM_SYNC)
    if wrapper.wrapped:
        applied.add(AWAITED_ASYNC_CALLS)
    if not applied:
        return code, []

    ast.fix_missing_locations(wrapped_module)
    return ast.unparse(wrapped_module), sorted(applied)


def _classify_view_methods(view_classes: list[type]) -> tuple[set[str], set[str]]:
    """Return ``(async method names, sync method names)`` across the facades.

    ``__getitem__`` is included as a sync method so an ``await`` over a facade
    subscript gets stripped. Other dunder methods are skipped because they are
    not invoked as plain attribute calls in user code.
    """
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
    return async_names, sync_names


class _AwaitStripper(ast.NodeTransformer):
    """Strip awaits whose operand is provably non-coroutine."""

    def __init__(self, sync_method_names: set[str]) -> None:
        self._sync_method_names = sync_method_names
        self.stripped = False

    def visit_Await(self, node: ast.Await) -> ast.AST:
        """Recurse first so nested rewrites settle before we inspect."""
        self.generic_visit(node)
        if self._is_sync_operand(node.value):
            self.stripped = True
            return node.value
        return node

    def _is_sync_operand(self, operand: ast.AST) -> bool:
        # A chain of attribute/subscript access rooted at a known sync global
        # (e.g. ``hass.states.get(...)`` or ``states['light.x'].state``) is
        # provably non-coroutine: every step returns a plain value. Sync method
        # calls on such a chain are also safe. We stop at any Call whose method
        # is not in the sync set, because its return value is unknown.
        if isinstance(operand, ast.Name):
            return operand.id in _SYNC_SUBSCRIPT_GLOBALS
        if isinstance(operand, ast.Attribute):
            return self._is_sync_operand(operand.value)
        if isinstance(operand, ast.Subscript):
            return self._is_sync_operand(operand.value)
        if isinstance(operand, ast.Call):
            # A sync method call on a sync chain is safe; any other call
            # returns a value of unknown shape and stops the walk.
            if isinstance(operand.func, ast.Attribute) and operand.func.attr in self._sync_method_names:
                return self._is_sync_operand(operand.func.value)
            return False
        return False


class _SubscriptRewriter(ast.NodeTransformer):
    """Rewrite Monty-visible state-machine operators into public method calls."""

    def __init__(self) -> None:
        self.rewrote = False
        self._shadowed_scopes: list[set[str]] = [set()]

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Visit top-level statements in execution order so assignments shadow later sugar."""
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Normalize function bodies with Python-local bindings treated as shadows."""
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Normalize async function bodies with Python-local bindings treated as shadows."""
        return self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        """Lambda arguments shadow facade globals inside the expression body."""
        node.args = self.visit(node.args)
        self._shadowed_scopes.append(_argument_names(node.args))
        try:
            node.body = self.visit(node.body)
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Class names bind after their definition; class bodies get their own scope."""
        self._shadowed_scopes[-1].update(_target_names(node))
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.bases = [self.visit(base) for base in node.bases]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        self._shadowed_scopes.append(set())
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        """For-loop targets shadow facade globals throughout the loop body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        node.body = [self.visit(statement) for statement in node.body]
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        """Async for-loop targets shadow facade globals throughout the loop body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        node.body = [self.visit(statement) for statement in node.body]
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        """With ``as`` targets shadow facade globals throughout the with body."""
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
                self._shadowed_scopes[-1].update(_target_names(item.optional_vars))
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        """Async with ``as`` targets shadow facade globals throughout the body."""
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
                self._shadowed_scopes[-1].update(_target_names(item.optional_vars))
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        """Exception aliases shadow facade globals inside the handler body."""
        if node.type is not None:
            node.type = self.visit(node.type)
        self._shadowed_scopes.append({node.name} if node.name in {"hass", "states"} else set())
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Assignments shadow facade globals only after their value is evaluated."""
        node.value = self.visit(node.value)
        node.targets = [self.visit(target) for target in node.targets]
        for target in node.targets:
            self._shadowed_scopes[-1].update(_target_names(target))
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Annotated assignments shadow facade globals after evaluating value/annotation."""
        node.annotation = self.visit(node.annotation)
        if node.value is not None:
            node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Augmented assignments read the previous root before shadowing future sugar."""
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        """Walrus targets shadow facade globals after evaluating the value."""
        node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Import aliases can replace facade globals for following statements."""
        self._shadowed_scopes[-1].update(_import_names(node.names))
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Imported names can replace facade globals for following statements."""
        self._shadowed_scopes[-1].update(_import_names(node.names))
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Recurse first so nested facade subscripts are rewritten from inside out."""
        self.generic_visit(node)
        if isinstance(node.slice, ast.Slice) or not self._is_state_machine_root(node.value):
            return node
        self.rewrote = True
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=node.value, attr="get", ctx=ast.Load()),
                args=[node.slice],
                keywords=[],
            ),
            node,
        )

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Rewrite state-machine containment checks to public method calls."""
        self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        if not self._is_state_machine_root(node.comparators[0]):
            return node
        comparator: ast.cmpop
        if isinstance(node.ops[0], ast.In):
            comparator = ast.IsNot()
        elif isinstance(node.ops[0], ast.NotIn):
            comparator = ast.Is()
        else:
            return node
        self.rewrote = True
        return ast.copy_location(
            ast.Compare(
                left=ast.Call(
                    func=ast.Attribute(value=node.comparators[0], attr="get", ctx=ast.Load()),
                    args=[node.left],
                    keywords=[],
                ),
                ops=[comparator],
                comparators=[ast.Constant(value=None)],
            ),
            node,
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Rewrite ``len(states)`` to a public state-machine method call."""
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "len" or len(node.args) != 1:
            return node
        if not self._is_state_machine_root(node.args[0]):
            return node
        self.rewrote = True
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Attribute(value=node.args[0], attr="async_entity_ids", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                ],
                keywords=[],
            ),
            node,
        )

    def _is_state_machine_root(self, node: ast.AST) -> bool:
        # Monty does not dispatch ``[]`` to dataclass ``__getitem__``. Keep the
        # public state-machine sugar by calling the HA-native ``get`` method.
        if isinstance(node, ast.Name):
            return node.id == "states" and not self._is_shadowed("states")
        if isinstance(node, ast.Attribute):
            return (
                node.attr == "states"
                and isinstance(node.value, ast.Name)
                and node.value.id == "hass"
                and not self._is_shadowed("hass")
            )
        return False

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._shadowed_scopes[-1].update(_target_names(node))
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.args = self.visit(node.args)
        node.returns = self.visit(node.returns) if node.returns is not None else None
        self._shadowed_scopes.append(_function_shadowed_names(node))
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node

    def _is_shadowed(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self._shadowed_scopes))


def _function_shadowed_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    shadowed = _argument_names(node.args)
    collector = _FacadeBindingCollector()
    for statement in node.body:
        collector.visit(statement)
    shadowed.update(collector.names)
    return shadowed


def _argument_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs] if arg.arg in {"hass", "states"}}
    if args.vararg is not None and args.vararg.arg in {"hass", "states"}:
        names.add(args.vararg.arg)
    if args.kwarg is not None and args.kwarg.arg in {"hass", "states"}:
        names.add(args.kwarg.arg)
    return names


def _import_names(aliases: list[ast.alias]) -> set[str]:
    names: set[str] = set()
    for alias in aliases:
        name = alias.asname or alias.name.split(".", maxsplit=1)[0]
        if name in {"hass", "states"}:
            names.add(name)
    return names


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name) and node.id in {"hass", "states"}:
        return {node.id}
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    if isinstance(node, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in node.elts:
            names.update(_target_names(element))
        return names
    return set()


class _FacadeBindingCollector(ast.NodeVisitor):
    """Collect function-local bindings that shadow facade root globals."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.update(_target_names(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.update(_target_names(node))

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.update(_target_names(node))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.names.update(_target_names(target))
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.names.update(_target_names(node.target))
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(_target_names(item.optional_vars))
            self.visit(item.context_expr)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self.names.update(_target_names(item.optional_vars))
            self.visit(item.context_expr)
        for statement in node.body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name in {"hass", "states"}:
            self.names.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(_import_names(node.names))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(_import_names(node.names))


class _AwaitWrapper(ast.NodeTransformer):
    """Wrap missing awaits on known async facade method calls."""

    def __init__(self, async_method_names: set[str]) -> None:
        self._async_method_names = async_method_names
        self.wrapped = False
        self._shadowed_scopes: list[set[str]] = [set()]
        # True while visiting the operand of an Await node. Prevents wrapping
        # a Call that is already the direct operand of an Await.
        self._inside_await_operand = False

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Visit top-level statements in execution order so assignments shadow later calls."""
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Normalize function bodies with Python-local bindings treated as shadows."""
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Normalize async function bodies with Python-local bindings treated as shadows."""
        return self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        """Lambda arguments shadow facade globals inside the expression body."""
        node.args = self.visit(node.args)
        self._shadowed_scopes.append(_argument_names(node.args))
        try:
            node.body = self.visit(node.body)
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Class names bind after their definition; class bodies get their own scope."""
        self._shadowed_scopes[-1].update(_target_names(node))
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.bases = [self.visit(base) for base in node.bases]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        self._shadowed_scopes.append(set())
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        """For-loop targets shadow facade globals throughout the loop body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        node.body = [self.visit(statement) for statement in node.body]
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        """Async for-loop targets shadow facade globals throughout the loop body."""
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        node.body = [self.visit(statement) for statement in node.body]
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_With(self, node: ast.With) -> ast.AST:
        """With ``as`` targets shadow facade globals throughout the with body."""
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
                self._shadowed_scopes[-1].update(_target_names(item.optional_vars))
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        """Async with ``as`` targets shadow facade globals throughout the body."""
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
                self._shadowed_scopes[-1].update(_target_names(item.optional_vars))
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        """Exception aliases shadow facade globals inside the handler body."""
        if node.type is not None:
            node.type = self.visit(node.type)
        self._shadowed_scopes.append({node.name} if node.name in {"hass", "states"} else set())
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Assignments shadow facade globals only after their value is evaluated."""
        node.value = self.visit(node.value)
        node.targets = [self.visit(target) for target in node.targets]
        for target in node.targets:
            self._shadowed_scopes[-1].update(_target_names(target))
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Annotated assignments shadow facade globals after evaluating value/annotation."""
        node.annotation = self.visit(node.annotation)
        if node.value is not None:
            node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Augmented assignments read the previous root before shadowing future calls."""
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        """Walrus targets shadow facade globals after evaluating the value."""
        node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        self._shadowed_scopes[-1].update(_target_names(node.target))
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST:
        """Import aliases can replace facade globals for following statements."""
        self._shadowed_scopes[-1].update(_import_names(node.names))
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Imported names can replace facade globals for following statements."""
        self._shadowed_scopes[-1].update(_import_names(node.names))
        return node

    def visit_Await(self, node: ast.Await) -> ast.AST:
        old = self._inside_await_operand
        self._inside_await_operand = True
        self.generic_visit(node)
        self._inside_await_operand = old
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Recurse first so nested async calls are also wrapped."""
        self.generic_visit(node)
        if self._inside_await_operand:
            return node
        if not isinstance(node.func, ast.Attribute):
            return node
        if node.func.attr not in self._async_method_names:
            return node
        if not self._is_view_rooted(node.func.value):
            return node
        self.wrapped = True
        return ast.Await(value=node)

    def _is_view_rooted(self, node: ast.AST) -> bool:
        # A chain rooted at a known facade global (e.g. hass.services.async_call)
        # is an API call we may auto-await. Anything else (a local variable, a
        # call result) is out of scope: do not rewrite it.
        if isinstance(node, ast.Name):
            if node.id == "hass":
                return not self._is_shadowed("hass")
            return node.id in _VIEW_GLOBALS and not self._is_shadowed(node.id)
        if isinstance(node, ast.Attribute):
            return self._is_view_rooted(node.value)
        if isinstance(node, ast.Subscript):
            return self._is_view_rooted(node.value)
        return False

    def _is_shadowed(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self._shadowed_scopes))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._shadowed_scopes[-1].update(_target_names(node))
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.args = self.visit(node.args)
        node.returns = self.visit(node.returns) if node.returns is not None else None
        self._shadowed_scopes.append(_function_shadowed_names(node))
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._shadowed_scopes.pop()
        return node
