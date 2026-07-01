"""AST normalization for safe builtin reflection conveniences.

This pass statically resolves ``type(x).__name__`` for bare facade-global
receivers and wraps the first argument of ``next()`` in ``iter()``. It is
intentionally scoped so the sandbox can forgive common Python discovery
patterns while leaving Monty-native builtins to run directly.
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
TYPE_NAME_RESOLVED = "type_name_resolved"
WRAPPED_NEXT_ITER = "wrapped_next_iter"


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
    """Resolve ``type(<global>).__name__`` and wrap ``next()`` args in ``iter()``."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, []

    resolver = _BuiltinResolver()
    module = resolver.visit(module)
    next_wrapper = _NextIterWrapper()
    module = next_wrapper.visit(module)

    applied = resolver.applied | next_wrapper.applied
    if not applied:
        return code, []

    ast.fix_missing_locations(module)
    return ast.unparse(module), sorted(applied)


class _BuiltinResolver(ast.NodeTransformer):
    """Rewrite type-name reflection when the receiver is a known global."""

    def __init__(self) -> None:
        self.applied: set[str] = set()

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


def _qualifying_global_type(node: ast.AST) -> type | None:
    """Return the facade type only for bare Monty global names."""
    if not isinstance(node, ast.Name):
        return None
    return GLOBAL_TYPE_MAP.get(node.id)


class _NextIterWrapper(ast.NodeTransformer):
    """Wrap the first argument of ``next(...)`` in ``iter(...)``.

    Monty generator expressions and comprehension results are lists, not
    iterators, so ``next(xs)``/``next(genexpr)`` fail at runtime. Wrapping the
    first argument in ``iter()`` makes any iterable a valid iterator
    (``iter`` is idempotent on real iterators, so multi-``next`` patterns are
    preserved). ``next(iter(x))`` written by the LLM is left untouched.
    """

    def __init__(self) -> None:
        self.applied: set[str] = set()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Recurse first so nested calls settle before this one.
        self.generic_visit(node)
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "next"
            or node.keywords
            or len(node.args) not in {1, 2}
        ):
            return node
        first = node.args[0]
        # Leave an explicit next(iter(...)) untouched.
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name) and first.func.id == "iter":
            return node
        self.applied.add(WRAPPED_NEXT_ITER)
        node.args[0] = ast.copy_location(
            ast.Call(func=ast.Name(id="iter", ctx=ast.Load()), args=[first], keywords=[]),
            first,
        )
        return node
