"""AST normalization for safe builtin reflection conveniences.

This pass statically resolves ``getattr``/``hasattr`` calls with literal names
and ``type(x).__name__`` for bare facade-global receivers. It is intentionally
scoped to known globals so the sandbox can forgive common Python discovery
patterns without enabling dunder walking or dynamic type resolution.
"""

import ast
import inspect
from dataclasses import fields

from ..snapshot.models import (
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfigEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SafeState,
)
from .facade_views import (
    SafeAreaRegistry,
    SafeCategoryRegistry,
    SafeConfigEntries,
    SafeDate,
    SafeDateFacade,
    SafeDateTime,
    SafeDateTimeFacade,
    SafeDeviceRegistry,
    SafeEntityRegistry,
    SafeFloorRegistry,
    SafeHass,
    SafeIssueRegistry,
    SafeLabelRegistry,
    SafeLLMContext,
    SafeNotificationRegistry,
    SafeServiceRegistry,
    SafeStateMachine,
)

GLOBAL_TYPE_MAP: dict[str, type] = {
    "hass": SafeHass,
    "states": SafeStateMachine,
    "er": SafeEntityRegistry,
    "dr": SafeDeviceRegistry,
    "ar": SafeAreaRegistry,
    "fr": SafeFloorRegistry,
    "lr": SafeLabelRegistry,
    "cr": SafeCategoryRegistry,
    "entity_registry": SafeEntityRegistry,
    "device_registry": SafeDeviceRegistry,
    "area_registry": SafeAreaRegistry,
    "floor_registry": SafeFloorRegistry,
    "label_registry": SafeLabelRegistry,
    "category_registry": SafeCategoryRegistry,
    "repairs": SafeIssueRegistry,
    "persistent_notifications": SafeNotificationRegistry,
    "config_entries": SafeConfigEntries,
    "date": SafeDateFacade,
    "datetime": SafeDateTimeFacade,
    "llm_context": SafeLLMContext,
}

_SUPPORTED_OPERATOR_DUNDERS = frozenset({"__getitem__", "__contains__", "__len__", "__iter__"})
HASATTR_RESOLVED = "hasattr_resolved"
GETATTR_RESOLVED = "getattr_resolved"
TYPE_NAME_RESOLVED = "type_name_resolved"
REWROTE_MAP_FILTER = "rewrote_map_filter"


def public_surface(cls: type) -> frozenset[str]:
    """Return public dataclass fields and supported methods for ``cls``."""
    field_names = {field.name for field in fields(cls)}
    method_names: set[str] = set()
    for name, _member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_") and name not in _SUPPORTED_OPERATOR_DUNDERS:
            continue
        method_names.add(name)
    return frozenset(field_names | method_names)


def surface_for_class_name(name: str) -> frozenset[str] | None:
    """Return the public surface for a facade or snapshot record class name."""
    classes_by_name = {
        cls.__name__: cls
        for cls in (
            *GLOBAL_TYPE_MAP.values(),
            SafeServiceRegistry,
            SafeContext,
            SafeState,
            SafeRegistryEntry,
            SafeDeviceEntry,
            SafeAreaEntry,
            SafeFloorEntry,
            SafeLabelEntry,
            SafeCategoryEntry,
            SafeIssueEntry,
            SafeNotificationEntry,
            SafeConfigEntry,
            SafeDate,
            SafeDateTime,
        )
    }
    if (cls := classes_by_name.get(name)) is None:
        return None
    return public_surface(cls)


def normalize_builtins(code: str) -> tuple[str, list[str]]:
    """Resolve safe builtin reflection and rewrite ``map``/``filter`` over known globals."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, []

    resolver = _BuiltinResolver()
    module = resolver.visit(module)
    map_filter = _MapFilterRewriter()
    module = map_filter.visit(module)

    applied = resolver.applied | map_filter.applied
    if not applied:
        return code, []

    ast.fix_missing_locations(module)
    return ast.unparse(module), sorted(applied)


class _BuiltinResolver(ast.NodeTransformer):
    """Rewrite literal builtin reflection when the receiver is a known global."""

    def __init__(self) -> None:
        self.applied: set[str] = set()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Recurse first so nested builtin rewrites settle before this call."""
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            return self._resolve_hasattr(node)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            return self._resolve_getattr(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Resolve ``type(<global>).__name__`` without rewriting bare ``type``."""
        self.generic_visit(node)
        if node.attr != "__name__":
            return node
        if not isinstance(node.value, ast.Call):
            return node
        if not isinstance(node.value.func, ast.Name) or node.value.func.id != "type":
            return node
        if len(node.value.args) != 1:
            return node
        if (cls := _qualifying_global_type(node.value.args[0])) is None:
            return node
        # Expose only the safe facade class name, not the type object itself.
        self.applied.add(TYPE_NAME_RESOLVED)
        return ast.copy_location(ast.Constant(value=cls.__name__), node)

    def _resolve_hasattr(self, node: ast.Call) -> ast.AST:
        # Only literal-name checks on bare known globals are statically safe.
        if len(node.args) != 2:
            return node
        if (cls := _qualifying_global_type(node.args[0])) is None:
            return node
        if not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
            return node
        self.applied.add(HASATTR_RESOLVED)
        return ast.copy_location(ast.Constant(value=node.args[1].value in public_surface(cls)), node)

    def _resolve_getattr(self, node: ast.Call) -> ast.AST:
        # Only literal-name lookups on bare known globals are statically safe.
        if len(node.args) not in {2, 3}:
            return node
        if (cls := _qualifying_global_type(node.args[0])) is None:
            return node
        if not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
            return node
        attr = node.args[1].value
        if attr in public_surface(cls):
            self.applied.add(GETATTR_RESOLVED)
            return ast.copy_location(ast.Attribute(value=node.args[0], attr=attr, ctx=ast.Load()), node)
        if len(node.args) == 3:
            self.applied.add(GETATTR_RESOLVED)
            return ast.copy_location(node.args[2], node)
        # Missing attributes without defaults keep their natural runtime error.
        return node


def _qualifying_global_type(node: ast.AST) -> type | None:
    """Return the facade type only for bare Monty global names."""
    if not isinstance(node, ast.Name):
        return None
    return GLOBAL_TYPE_MAP.get(node.id)


class _MapFilterRewriter(ast.NodeTransformer):
    """Rewrite ``map``/``filter`` calls into equivalent list comprehensions.

    Monty does not provide ``map``/``filter`` as builtins, but they are common
    in LLM-generated code. Only shapes whose evaluation semantics are preserved
    are rewritten; anything ambiguous is left untouched so Monty surfaces the
    natural error. The callable must be a plain name or lambda (so call timing
    and arity are unchanged); multi-iterable ``map`` is driven through ``zip``.
    """

    def __init__(self) -> None:
        self.applied: set[str] = set()
        self._counter = 0

    def _target(self) -> str:
        name = f"_lsbx_{self._counter}"
        self._counter += 1
        return name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Recurse first so nested map/filter settle before this call.
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        name = node.func.id
        if name == "map" and len(node.args) >= 2:
            return self._rewrite_map(node)
        if name == "filter" and len(node.args) == 2:
            return self._rewrite_filter(node)
        return node

    def _rewrite_map(self, node: ast.Call) -> ast.AST:
        func = node.args[0]
        iterables = node.args[1:]
        # Reject callable expressions whose call timing/side effects could differ.
        if not isinstance(func, ast.Name | ast.Lambda):
            return node
        targets = [self._target() for _ in iterables]
        if len(iterables) == 1:
            iter_expr: ast.expr = iterables[0]
        else:
            # Multi-iterable map pairs elements like Python's eager map via zip.
            iter_expr = ast.Call(
                func=ast.Name(id="zip", ctx=ast.Load()),
                args=list(iterables),
                keywords=[],
            )
        elt = ast.Call(
            func=func,
            args=[ast.Name(id=target, ctx=ast.Load()) for target in targets],
            keywords=[],
        )
        target_node = self._comprehension_target(targets)
        self.applied.add(REWROTE_MAP_FILTER)
        return ast.copy_location(
            ast.ListComp(
                elt=elt,
                generators=[ast.comprehension(target=target_node, iter=iter_expr, ifs=[], is_async=0)],
            ),
            node,
        )

    def _rewrite_filter(self, node: ast.Call) -> ast.AST:
        func = node.args[0]
        iterable = node.args[1]
        target = self._target()
        if isinstance(func, ast.Constant) and func.value is None:
            # filter(None, xs) keeps truthy elements.
            condition: ast.expr = ast.Name(id=target, ctx=ast.Load())
        elif isinstance(func, ast.Name | ast.Lambda):
            condition = ast.Call(
                func=func,
                args=[ast.Name(id=target, ctx=ast.Load())],
                keywords=[],
            )
        else:
            return node
        self.applied.add(REWROTE_MAP_FILTER)
        return ast.copy_location(
            ast.ListComp(
                elt=ast.Name(id=target, ctx=ast.Load()),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id=target, ctx=ast.Store()),
                        iter=iterable,
                        ifs=[condition],
                        is_async=0,
                    )
                ],
            ),
            node,
        )

    @staticmethod
    def _comprehension_target(targets: list[str]) -> ast.expr:
        if len(targets) == 1:
            return ast.Name(id=targets[0], ctx=ast.Store())
        return ast.Tuple(
            elts=[ast.Name(id=target, ctx=ast.Store()) for target in targets],
            ctx=ast.Store(),
        )
