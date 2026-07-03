"""Derived registry for the Monty-visible facade surface."""

from dataclasses import dataclass
from typing import Any

from ..snapshot.models import (
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfig,
    SafeConfigEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
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


@dataclass(frozen=True, slots=True)
class FacadeBinding:
    """One root facade class and the bare global exposing it to Monty."""

    cls: type[Any]
    names: tuple[str, ...] = ()
    subscript_sync: bool = True


FACADE_BINDINGS: tuple[FacadeBinding, ...] = (
    FacadeBinding(SafeHass, ("hass",)),
    FacadeBinding(SafeStateMachine, ("states",)),
    FacadeBinding(SafeServiceRegistry),
    FacadeBinding(SafeEntityRegistry, ("er",)),
    FacadeBinding(SafeDeviceRegistry, ("dr",)),
    FacadeBinding(SafeAreaRegistry, ("ar",)),
    FacadeBinding(SafeFloorRegistry, ("fr",)),
    FacadeBinding(SafeLabelRegistry, ("lr",)),
    FacadeBinding(SafeCategoryRegistry, ("cr",)),
    FacadeBinding(SafeEntityRegistry, ("entity_registry",)),
    FacadeBinding(SafeDeviceRegistry, ("device_registry",)),
    FacadeBinding(SafeAreaRegistry, ("area_registry",)),
    FacadeBinding(SafeFloorRegistry, ("floor_registry",)),
    FacadeBinding(SafeLabelRegistry, ("label_registry",)),
    FacadeBinding(SafeCategoryRegistry, ("category_registry",)),
    FacadeBinding(SafeIssueRegistry, ("repairs",), subscript_sync=False),
    FacadeBinding(SafeNotificationRegistry, ("persistent_notifications",), subscript_sync=False),
    FacadeBinding(SafeConfigEntries, ("config_entries",), subscript_sync=False),
    FacadeBinding(SafeDateFacade, ("date",)),
    FacadeBinding(SafeDateTimeFacade, ("datetime",)),
    FacadeBinding(SafeLLMContext, ("llm_context",)),
)


def _unique_classes(bindings: tuple[FacadeBinding, ...]) -> tuple[type[Any], ...]:
    """Return registry classes once, preserving their first declaration order."""
    classes: list[type[Any]] = []
    for binding in bindings:
        if binding.cls not in classes:
            classes.append(binding.cls)
    return tuple(classes)


FACADE_CLASSES = _unique_classes(FACADE_BINDINGS)

# ``now`` is a raw ISO string global, not a facade object, so it deliberately
# remains outside ``GLOBAL_TYPE_MAP`` and the dataclass registry.
RAW_GLOBALS: tuple[str, ...] = ("now",)
_FACADE_GLOBALS = [name for binding in FACADE_BINDINGS for name in binding.names]
AVAILABLE_GLOBALS: list[str] = [*_FACADE_GLOBALS[:-1], *RAW_GLOBALS, _FACADE_GLOBALS[-1]]
GLOBAL_TYPE_MAP: dict[str, type[Any]] = {name: binding.cls for binding in FACADE_BINDINGS for name in binding.names}
SYNC_SUBSCRIPT_GLOBALS = frozenset(
    name for binding in FACADE_BINDINGS if binding.subscript_sync for name in binding.names
)

# Record/value dataclasses reached from facade methods or fields. These are not
# bare globals, but Monty must know them for type-checking returned objects.
ATTRIBUTE_REACHABLE_RECORDS: tuple[type[Any], ...] = (
    SafeDate,
    SafeDateTime,
    SafeContext,
    SafeState,
    SafeConfig,
    SafeUnitSystem,
    SafeRegistryEntry,
    SafeDeviceEntry,
    SafeAreaEntry,
    SafeFloorEntry,
    SafeLabelEntry,
    SafeCategoryEntry,
    SafeIssueEntry,
    SafeNotificationEntry,
    SafeConfigEntry,
)

MONTY_DATACLASS_REGISTRY: list[type[Any]] = [*FACADE_CLASSES, *ATTRIBUTE_REACHABLE_RECORDS]
