"""AST normalization that resolves datetime/date imports to sandbox facades.

The sandbox exposes ``date`` and ``datetime`` as frozen facade globals. LLMs
commonly write ``from datetime import datetime`` or ``import datetime as dt``
which would shadow or mismatch those globals. This pass rewrites supported
datetime imports and aliases so they resolve to the sandbox facades, before
builtin/await normalization runs.

``from datetime import <name> [as alias]`` becomes a normal Python name binding
(``alias = <name>``), so Python's own scoping rules apply. ``import datetime``
and ``import datetime as dt`` use attribute rewriting that is conservative: when
the alias name is locally bound anywhere in the module (e.g. a function
parameter named ``dt``), rewriting is skipped and the real import is kept so
Monty surfaces a natural import error instead of silently misrewriting code.

Unsupported imports (e.g. ``timedelta``) are left in place so Monty surfaces
the natural error.
"""

import ast

DATETIME_IMPORTS_RESOLVED = "datetime_imports_resolved"

# Names that map directly to sandbox facade globals.
_SUPPORTED_CLASS_NAMES = frozenset({"datetime", "date"})


def normalize_datetime_imports(code: str) -> tuple[str, list[str]]:
    """Rewrite supported datetime imports to sandbox facade globals."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, []

    collector = _ImportCollector()
    collector.visit(module)

    if not collector.has_supported_imports:
        return code, []

    bound_names = _LocalBoundNames.collect(module)

    rewriter = _ImportRewriter(
        module_aliases=collector.module_aliases,
        locally_bound_names=bound_names,
    )
    rewritten_module = rewriter.visit(module)

    if not rewriter.applied:
        return code, []

    ast.fix_missing_locations(rewritten_module)
    return ast.unparse(rewritten_module), [DATETIME_IMPORTS_RESOLVED]


class _ImportCollector(ast.NodeVisitor):
    """Collect supported datetime import aliases for later rewriting."""

    def __init__(self) -> None:
        # Maps alias name → source name for module-level imports
        # e.g. {"dt": "datetime"} for ``import datetime as dt``
        # e.g. {"datetime": "datetime"} for ``import datetime``
        self.module_aliases: dict[str, str] = {}
        self.has_supported_imports = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "datetime":
                bound_name = alias.asname if alias.asname else alias.name
                self.module_aliases[bound_name] = "datetime"
                self.has_supported_imports = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "datetime":
            return
        for alias in node.names:
            if alias.name in _SUPPORTED_CLASS_NAMES:
                self.has_supported_imports = True
                # Aliased class imports (e.g. ``from datetime import datetime as dt``)
                # are handled by the rewriter generating assignment statements.
                # Unaliased class imports (e.g. ``from datetime import datetime``)
                # are handled by simply dropping the name — the bare global exists.


class _LocalBoundNames(ast.NodeVisitor):
    """Collect names bound locally anywhere in the module.

    Covers function/lambda parameters, Store-context name targets (assignments,
    for/with/comprehension/walrus targets), except handler names, and
    function/class definition names. Used to decide when a module-import alias
    is shadowed and must not be blindly rewritten.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    @classmethod
    def collect(cls, module: ast.Module) -> set[str]:
        """Return every locally-bound name in ``module``."""
        visitor = cls()
        visitor.visit(module)
        return visitor.names

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    visit_FunctionDef = _visit_def  # noqa: N815
    visit_AsyncFunctionDef = _visit_def  # noqa: N815

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)


class _ImportRewriter(ast.NodeTransformer):
    """Rewrite datetime imports and module-alias attribute access."""

    def __init__(
        self,
        module_aliases: dict[str, str],
        locally_bound_names: set[str],
    ) -> None:
        self._module_aliases = module_aliases
        self._locally_bound_names = locally_bound_names
        self.applied = False

    def _is_shadowed(self, bound_name: str) -> bool:
        return bound_name in self._locally_bound_names

    def visit_Import(self, node: ast.Import) -> ast.AST | list[ast.stmt] | None:
        """Drop supported datetime aliases, unless the alias is locally shadowed."""
        kept: list[ast.alias] = []
        for alias in node.names:
            if alias.name != "datetime":
                kept.append(alias)
                continue
            bound_name = alias.asname if alias.asname else alias.name
            # If the alias name is locally shadowed (e.g. a ``dt`` parameter),
            # keep the real import so attribute access resolves to the stdlib
            # module and Monty surfaces a natural import error.
            if self._is_shadowed(bound_name):
                kept.append(alias)
                continue
            self.applied = True
        if len(kept) == len(node.names):
            return node  # Nothing to remove.
        if not kept:
            return None  # Remove the entire statement.
        node.names = kept
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | list[ast.stmt] | None:
        """Rewrite ``from datetime import ...`` for supported names."""
        if node.module != "datetime":
            return node

        assignments: list[ast.stmt] = []
        kept_aliases: list[ast.alias] = []

        for alias in node.names:
            if alias.name in _SUPPORTED_CLASS_NAMES:
                self.applied = True
                if alias.asname is not None:
                    # ``from datetime import datetime as dt`` → ``dt = datetime``
                    assignments.append(
                        ast.Assign(
                            targets=[ast.Name(id=alias.asname, ctx=ast.Store())],
                            value=ast.Name(id=alias.name, ctx=ast.Load()),
                        )
                    )
                # Unaliased: just drop (bare global already exists).
            else:
                kept_aliases.append(alias)

        if not self.applied:
            return node

        if kept_aliases:
            node.names = kept_aliases
            if assignments:
                return [*assignments, node]
            return node
        # All names were supported → drop the import entirely.
        if assignments:
            return assignments
        return None

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Rewrite module_alias.datetime / .date to bare globals, when unshadowed."""
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        bound_name = node.value.id
        if bound_name not in self._module_aliases:
            return node
        if self._is_shadowed(bound_name):
            return node
        if node.attr not in _SUPPORTED_CLASS_NAMES:
            return node
        self.applied = True
        return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
