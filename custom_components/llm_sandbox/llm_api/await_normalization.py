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

AWAITED_ASYNC_CALLS = "awaited_async_calls"
STRIPPED_AWAIT_FROM_SYNC = "stripped_await_from_sync"
REWROTE_SYNC_SUBSCRIPT = "rewrote_sync_subscript"

# Top-level globals exposed to Monty whose attribute access returns a plain
# (non-coroutine) value. These cover the snapshot-backed facades: state machine,
# unified registry facades, date/time facades, context, and the hass root.
_SYNC_SUBSCRIPT_GLOBALS = frozenset(
    {
        "hass",
        "states",
        "er",
        "dr",
        "ar",
        "fr",
        "lr",
        "cr",
        "entity_registry",
        "device_registry",
        "area_registry",
        "floor_registry",
        "label_registry",
        "category_registry",
        "date",
        "datetime",
        "llm_context",
    }
)
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
            return node.id == "states"
        if isinstance(node, ast.Attribute):
            return node.attr == "states" and isinstance(node.value, ast.Name) and node.value.id == "hass"
        return False


class _AwaitWrapper(ast.NodeTransformer):
    """Wrap missing awaits on known async facade method calls."""

    def __init__(self, async_method_names: set[str]) -> None:
        self._async_method_names = async_method_names
        self.wrapped = False
        # True while visiting the operand of an Await node. Prevents wrapping
        # a Call that is already the direct operand of an Await.
        self._inside_await_operand = False

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
            return node.id in _VIEW_GLOBALS
        if isinstance(node, ast.Attribute):
            return self._is_view_rooted(node.value)
        if isinstance(node, ast.Subscript):
            return self._is_view_rooted(node.value)
        return False
