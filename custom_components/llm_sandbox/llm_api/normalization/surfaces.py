"""Public surface descriptions for facade and snapshot-record classes.

Consumed by error refinement and tests; kept independent of the rewrite
engine so the surface description survives normalization-layer changes.
"""

from dataclasses import fields
import inspect

from ..facade_registry import ATTRIBUTE_REACHABLE_RECORDS, FACADE_CLASSES

_SUPPORTED_OPERATOR_DUNDERS = frozenset({"__getitem__", "__contains__", "__len__", "__iter__"})


def public_surface(cls: type) -> frozenset[str]:
    """Return public dataclass fields and supported methods for ``cls``."""
    field_names = {field.name for field in fields(cls) if not field.name.startswith("_")}
    method_names: set[str] = set()
    for name, _member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_") and name not in _SUPPORTED_OPERATOR_DUNDERS:
            continue
        method_names.add(name)
    return frozenset(field_names | method_names)


def surface_for_class_name(name: str) -> frozenset[str] | None:
    """Return the public surface for a facade or snapshot record class name."""
    classes_by_name = {cls.__name__: cls for cls in (*FACADE_CLASSES, *ATTRIBUTE_REACHABLE_RECORDS)}
    if (cls := classes_by_name.get(name)) is None:
        return None
    return public_surface(cls)
