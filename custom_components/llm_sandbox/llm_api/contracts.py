"""Runtime contracts for the Monty-backed LLM Sandbox facade runtime.

Holds mechanically derived artifacts and programmatic data lists that must
track the facade dataclasses:

- ``AVAILABLE_GLOBALS``: the global names exposed to Monty.
- ``suggested_methods()``: programmatic data surfaced in code-error payloads.
- ``MONTY_TYPE_STUBS``: code-generated from the facade dataclasses via
  ``get_type_hints``; consumed by the Monty type-checker.

Static LLM-facing prose lives in ``prompts.py``.
"""

from typing import Any

AVAILABLE_GLOBALS = [
    "hass",
    "states",
    "er",
    "dr",
    "ar",
    "fr",
    "entity_registry",
    "device_registry",
    "area_registry",
    "floor_registry",
    "now",
    "llm_context",
]


def suggested_methods() -> list[str]:
    """Return suggested methods for code-error payloads."""
    return [
        "hass.states.get('light.bedroom')",
        "hass.states.async_all('light')",
        "er.async_entries_for_area(er.async_get(hass), '<area_id>')",
        "er.async_entries_for_device(er.async_get(hass), '<device_id>')",
        "list(device_registry.devices.values())",
        "area_registry.async_get_area_by_name('Bedroom')",
        "[(area.name, area.id) for area in area_registry.async_list_areas()]",
        "[(f.name, f.floor_id) for f in floor_registry.async_list_floors()]",
        "device_registry.async_get('<device_id>')",
        "hass.services.has_service('light', 'turn_on')",
        "hass.services.async_services_for_domain('light')",
        "hass.services.supports_response('light', 'turn_on')",
        "await hass.services.async_call('light', 'turn_on', {'brightness_pct': 80}, target={'entity_id': 'light.bedroom'})  # performs the call",
    ]


def _build_monty_type_stubs() -> str:
    """Render Monty stubs from the facade dataclasses and their methods.

    For each facade class, renders public fields (non-underscore) via
    ``get_type_hints``, then appends method signatures reflected from the
    class via ``inspect.signature`` so the stub stays in sync with the real API.
    """
    import inspect
    from collections.abc import Mapping as AbcMapping
    from dataclasses import fields
    from types import UnionType
    from typing import get_args, get_origin, get_type_hints

    from ..snapshot.models import (
        SafeAreaEntry,
        SafeContext,
        SafeDeviceEntry,
        SafeFloorEntry,
        SafeRegistryEntry,
        SafeState,
    )
    from .facade_views import (
        SafeAreaModule,
        SafeAreaRegistry,
        SafeDeviceModule,
        SafeDeviceRegistry,
        SafeEntityModule,
        SafeEntityRegistry,
        SafeFloorModule,
        SafeFloorRegistry,
        SafeHass,
        SafeLLMContext,
        SafeServiceRegistry,
        SafeStateMachine,
    )

    def _format_type(annotation: object) -> str:
        if annotation is Any:
            return "Any"
        if annotation is type(None):
            return "None"
        if isinstance(annotation, type):
            return annotation.__name__
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
        hints = get_type_hints(cls, include_extras=True)
        public_fields = [f for f in fields(cls) if not f.name.startswith("_")]
        return [f"    {f.name}: {_format_type(hints[f.name])}" for f in public_fields] if public_fields else []

    def _render_methods(cls: type[Any]) -> list[str]:
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
                if pname == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                ptype = _format_type(param.annotation) if param.annotation is not inspect.Parameter.empty else "Any"
                default = ""
                if param.default is not inspect.Parameter.empty:
                    default = " = None"
                params.append(f"{pname}: {ptype}{default}")
            if sig.return_annotation is not inspect.Signature.empty:
                returns = f" -> {_format_type(sig.return_annotation)}"
            lines.append(f"    {prefix} {name}({', '.join(params)}){returns}: ...")
        return lines

    def _render_class(cls: type[Any]) -> list[str]:
        body = _render_fields(cls) + _render_methods(cls)
        if not body:
            return [f"class {cls.__name__}:", "    pass"]
        return [f"class {cls.__name__}:", *body]

    all_classes = [
        SafeContext,
        SafeState,
        SafeRegistryEntry,
        SafeDeviceEntry,
        SafeAreaEntry,
        SafeFloorEntry,
        SafeStateMachine,
        SafeServiceRegistry,
        SafeEntityRegistry,
        SafeDeviceRegistry,
        SafeAreaRegistry,
        SafeFloorRegistry,
        SafeEntityModule,
        SafeDeviceModule,
        SafeAreaModule,
        SafeFloorModule,
        SafeHass,
        SafeLLMContext,
    ]

    sections: list[str] = ["from typing import Any, Mapping", ""]
    for cls in all_classes:
        sections.append("")
        sections.extend(_render_class(cls))

    # Global declarations so Monty knows the root object types.
    sections.append("")
    for name in AVAILABLE_GLOBALS:
        sections.append(f"{name}: Any")

    return "\n".join(sections)


MONTY_TYPE_STUBS = _build_monty_type_stubs()
