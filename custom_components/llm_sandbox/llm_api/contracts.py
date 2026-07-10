"""Runtime contracts for the Monty-backed LLM Sandbox facade runtime.

Holds mechanically derived artifacts that must track the facade dataclasses:

- ``AVAILABLE_GLOBALS``: the global names exposed to Monty.
- ``MONTY_TYPE_STUBS``: code-generated from the facade dataclasses via
  ``get_type_hints``; consumed by the Monty type-checker.

Static LLM-facing prose lives in ``prompts.py``.
"""

from typing import Any

from ..snapshot.models import _JsonSafeRecord
from .facade_registry import ATTRIBUTE_REACHABLE_RECORDS, AVAILABLE_GLOBALS, FACADE_CLASSES

# Builtin callables Monty runs natively but does not auto-declare to its
# type-checker. Declaring them lets common LLM discovery patterns
# (hasattr/getattr/next/iter/map/filter) type-check AND run. Runtime
# getattr/hasattr cannot walk dunders, so this opens no new escape surface.
MONTY_BUILTIN_STUBS = """\
def hasattr(obj: Any, name: str) -> bool: ...
def getattr(obj: Any, name: str, default: Any = ...) -> Any: ...
def next(iterator: Any, default: Any = ...) -> Any: ...
def iter(obj: Any) -> Any: ...
def map(func: Any, *iterables: Any) -> list[Any]: ...
def filter(func: Any, iterable: Any) -> list[Any]: ...
"""

_RECORD_MAPPING_READ_STUBS = (
    "    def get(self, key: str, default: object = None) -> object | None: ...",
    "    def keys(self) -> list[str]: ...",
    "    def items(self) -> list[tuple[str, object]]: ...",
    "    def values(self) -> list[object]: ...",
)


def _format_type(annotation: object) -> str:
    import inspect
    from collections.abc import Mapping as AbcMapping
    from types import UnionType
    from typing import get_args, get_origin

    if annotation is Any:
        return "Any"
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    if annotation is inspect.Parameter.empty:
        return "Any"

    origin = get_origin(annotation)
    if origin is None:
        return str(annotation).replace("typing.", "")

    args = get_args(annotation)
    if origin is UnionType:
        return " | ".join(_format_type(arg) for arg in args)
    if origin is AbcMapping:
        return f"Mapping[{_format_type(args[0])}, {_format_type(args[1])}]"
    if origin is list:
        return f"list[{_format_type(args[0])}]"
    if origin is dict:
        return f"dict[{_format_type(args[0])}, {_format_type(args[1])}]"
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_format_type(args[0])}, ...]"
        return f"tuple[{', '.join(_format_type(arg) for arg in args)}]"
    if origin is set:
        return f"set[{_format_type(args[0])}]"
    if getattr(origin, "__module__", "") == "typing" and getattr(origin, "__qualname__", "") == "Union":
        return " | ".join(_format_type(arg) for arg in args)
    return str(annotation).replace("typing.", "").replace("collections.abc.", "")


def _render_fields(cls: type[Any]) -> list[str]:
    from dataclasses import fields
    from typing import get_type_hints

    hints = get_type_hints(cls, include_extras=True)
    public_fields = [f for f in fields(cls) if not f.name.startswith("_")]
    return [f"    {f.name}: {_format_type(hints[f.name])}" for f in public_fields]


def _render_methods(cls: type[Any]) -> list[str]:
    import inspect

    lines: list[str] = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_") and name not in {"__getitem__", "__contains__", "__len__", "__iter__"}:
            continue
        try:
            sig = inspect.signature(member)
        except TypeError, ValueError:
            continue
        is_async = inspect.iscoroutinefunction(member)
        prefix = "async def" if is_async else "def"
        params: list[str] = ["self"]
        returns = ""
        for pname, param in sig.parameters.items():
            if pname == "self" or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            default = " = None" if param.default is not inspect.Parameter.empty else ""
            params.append(f"{pname}: {_format_type(param.annotation)}{default}")
        if sig.return_annotation is not inspect.Signature.empty:
            returns = f" -> {_format_type(sig.return_annotation)}"
        lines.append(f"    {prefix} {name}({', '.join(params)}){returns}: ...")
    if issubclass(cls, _JsonSafeRecord):
        # Monty resolves these record reads through _JsonSafeRecord.__getattr__
        # so its field-discovery surface remains limited to actual record data.
        lines.extend(_RECORD_MAPPING_READ_STUBS)
    return lines


def _render_class(cls: type[Any]) -> list[str]:
    body = _render_fields(cls) + _render_methods(cls)
    if not body:
        return [f"class {cls.__name__}:", "    pass"]
    return [f"class {cls.__name__}:", *body]


def _build_monty_type_stubs() -> str:
    """Render Monty stubs from the facade dataclasses and their methods.

    For each facade class, renders public fields (non-underscore) via
    ``get_type_hints``, then appends method signatures reflected from the
    class via ``inspect.signature`` so the stub stays in sync with the real API.
    """
    all_classes = [*ATTRIBUTE_REACHABLE_RECORDS, *FACADE_CLASSES]

    sections: list[str] = ["from typing import Any, Mapping", ""]
    for cls in all_classes:
        sections.append("")
        sections.extend(_render_class(cls))

    # Global declarations so Monty knows the root object types.
    sections.append("")
    sections.extend(f"{name}: Any" for name in AVAILABLE_GLOBALS)

    sections.append("")
    sections.append(MONTY_BUILTIN_STUBS)

    return "\n".join(sections)


MONTY_TYPE_STUBS = _build_monty_type_stubs()
