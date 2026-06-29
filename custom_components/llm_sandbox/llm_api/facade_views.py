"""Monty-facing Home Assistant-native facade views.

Each facade is a frozen dataclass that reads from a frozen ``HomeSnapshot``.
Reads mirror Home Assistant's synchronous registry/state-machine callbacks
and cost zero helper calls. The only async method is
``hass.services.async_call``, which records a proposed action (Option A) and
costs one helper call.

The facades intentionally expose only the documented public surface. They
never leak the live ``HomeAssistant`` object, mutable registries, the event
bus, the config, or auth into the Monty sandbox.
"""


# ruff: noqa: D105, ANN401

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from homeassistant.core import SupportsResponse
from homeassistant.util.json import JsonValueType

from ..snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeRegistryEntry,
    SafeState,
    SnapshotIndexes,
)
from ..types import ProposedAction
from .executor_support import helper_response, json_safe
from .runtime import RuntimeContext, require_runtime

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeStateMachine:
    """Read-only Home Assistant StateMachine facade.

    Mirrors HA's ``hass.states`` read API. All methods are synchronous
    callbacks (the ``async_`` prefix denotes loop-safe, not coroutine).
    Optional subscript sugar (``states["light.x"]``, ``"light.x" in states``)
    is provided in addition to the strict ``get``/``async_all`` methods.
    """

    states: Mapping[str, SafeState]
    type: str = "states"

    def get(self, entity_id: str) -> SafeState | None:
        """Return the state for ``entity_id``, or None if it does not exist."""
        return self.states.get(entity_id)

    def async_all(self, domain_filter: str | None = None) -> list[SafeState]:
        """Return all states, optionally filtered by domain."""
        if domain_filter is None:
            return list(self.states.values())
        return [s for s in self.states.values() if s.domain == domain_filter]

    def is_state(self, entity_id: str, state: str) -> bool:
        """Return True if ``entity_id`` exists and its state equals ``state``."""
        st = self.states.get(entity_id)
        return st is not None and st.state == state

    def async_entity_ids(self, domain_filter: str | None = None) -> list[str]:
        """Return all entity IDs, optionally filtered by domain."""
        if domain_filter is None:
            return list(self.states.keys())
        return [eid for eid, st in self.states.items() if st.domain == domain_filter]

    def entity_ids(self, domain_filter: str | None = None) -> list[str]:
        """Sync alias for async_entity_ids (HA parity)."""
        return self.async_entity_ids(domain_filter)

    # --- Optional subscript/containment sugar (additive; strict API still works) ---

    def __getitem__(self, entity_id: str) -> SafeState:
        st = self.states.get(entity_id)
        if st is None:
            raise KeyError(entity_id)
        return st

    def __contains__(self, entity_id: object) -> bool:
        return isinstance(entity_id, str) and entity_id in self.states

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self) -> Any:
        return iter(self.states.values())

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, {"type": self.type, "entity_count": len(self.states)})


# ---------------------------------------------------------------------------
# Entity registry (instance facade: entity_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeEntityRegistry:
    """Read-only entity registry facade mirroring ``EntityRegistry`` instance methods."""

    entities: Mapping[str, SafeRegistryEntry]

    def async_get(self, entity_id: str) -> SafeRegistryEntry | None:
        """Return the registry entry for ``entity_id``, or None."""
        return self.entities.get(entity_id)

    def async_get_entity_id(self, domain: str, platform: str, unique_id: str) -> str | None:
        """Return the entity_id matching (domain, platform, unique_id), or None."""
        for entry in self.entities.values():
            if (
                entry.entity_id.split(".", 1)[0] == domain
                and entry.platform == platform
                and entry.unique_id == unique_id
            ):
                return entry.entity_id
        return None


# ---------------------------------------------------------------------------
# Entity registry (module facade: er global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeEntityModule:
    """Module-level entity registry facade mirroring ``er`` (entity_registry)."""

    registry: SafeEntityRegistry
    entities: Mapping[str, SafeRegistryEntry]
    indexes: SnapshotIndexes

    def async_get(self, _hass: object) -> SafeEntityRegistry:
        """Return the entity registry instance (HA parity: ``er.async_get(hass)``)."""
        return self.registry

    def async_entries_for_area(
        self,
        registry: SafeEntityRegistry,
        area_id: str,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries whose effective area is ``area_id``.

        Effective area = ``entity.area_id or device.area_id`` (entity override wins).
        """
        del registry  # HA parity: the registry argument is the first parameter.
        entity_ids = self.indexes.entity_ids_by_area_id.get(area_id, ())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def async_entries_for_device(
        self,
        registry: SafeEntityRegistry,
        device_id: str,
        include_disabled_entities: bool = False,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries linked to ``device_id``."""
        del registry
        entity_ids = self.indexes.entity_ids_by_device_id.get(device_id, ())
        results: list[SafeRegistryEntry] = []
        for eid in entity_ids:
            entry = self.entities.get(eid)
            if entry is None:
                continue
            if entry.disabled_by is not None and not include_disabled_entities:
                continue
            results.append(entry)
        return results

    def async_entries_for_config_entry(
        self,
        registry: SafeEntityRegistry,
        config_entry_id: str,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries created by ``config_entry_id``."""
        del registry
        entity_ids = self.indexes.entity_ids_by_config_entry_id.get(config_entry_id, ())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def async_entries_for_label(
        self,
        registry: SafeEntityRegistry,
        label_id: str,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries carrying ``label_id``."""
        del registry
        entity_ids = self.indexes.entity_ids_by_label.get(label_id, ())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]


# ---------------------------------------------------------------------------
# Device registry (instance facade: device_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeDeviceRegistry:
    """Read-only device registry facade mirroring ``DeviceRegistry`` instance methods."""

    devices: Mapping[str, SafeDeviceEntry]

    def async_get(self, device_id: str) -> SafeDeviceEntry | None:
        """Return the device entry for ``device_id``, or None."""
        return self.devices.get(device_id)

    def async_get_device(
        self,
        identifiers: set[tuple[str, ...]],
        connections: set[tuple[str, ...]] | None = None,
    ) -> SafeDeviceEntry | None:
        """Return the device matching the given identifiers or connections."""
        ident_set = {tuple(i) for i in (identifiers or set())}
        conn_set = {tuple(c) for c in (connections or set())}
        for device in self.devices.values():
            if ident_set and ident_set.intersection(device.identifiers):
                return device
            if conn_set and conn_set.intersection(device.connections):
                return device
        return None


@dataclass(frozen=True, slots=True)
class SafeDeviceModule:
    """Module-level device registry facade mirroring ``dr`` (device_registry)."""

    registry: SafeDeviceRegistry
    devices: Mapping[str, SafeDeviceEntry]
    indexes: SnapshotIndexes

    def async_get(self, _hass: object) -> SafeDeviceRegistry:
        """Return the device registry instance (HA parity: ``dr.async_get(hass)``)."""
        return self.registry

    def async_entries_for_area(
        self,
        registry: SafeDeviceRegistry,
        area_id: str,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries assigned to ``area_id``."""
        del registry
        device_ids = self.indexes.device_ids_by_area_id.get(area_id, ())
        return [self.devices[did] for did in device_ids if did in self.devices]

    def async_entries_for_config_entry(
        self,
        registry: SafeDeviceRegistry,
        config_entry_id: str,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries linked to ``config_entry_id``."""
        del registry
        return [d for d in self.devices.values() if config_entry_id in d.config_entries]


# ---------------------------------------------------------------------------
# Area registry (instance facade: area_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeAreaRegistry:
    """Read-only area registry facade mirroring ``AreaRegistry`` instance methods."""

    areas: Mapping[str, SafeAreaEntry]

    def async_get_area(self, area_id: str) -> SafeAreaEntry | None:
        """Return the area entry for ``area_id``, or None."""
        return self.areas.get(area_id)

    def async_get_area_by_name(self, name: str) -> SafeAreaEntry | None:
        """Return the area whose name or alias matches ``name`` (case-insensitive)."""
        lowered = name.lower()
        for area in self.areas.values():
            if area.name.lower() == lowered:
                return area
            if any(alias.lower() == lowered for alias in area.aliases):
                return area
        return None

    def async_list_areas(self) -> list[SafeAreaEntry]:
        """Return all area entries."""
        return list(self.areas.values())


@dataclass(frozen=True, slots=True)
class SafeAreaModule:
    """Module-level area registry facade mirroring ``ar`` (area_registry)."""

    registry: SafeAreaRegistry

    def async_get(self, _hass: object) -> SafeAreaRegistry:
        """Return the area registry instance (HA parity: ``ar.async_get(hass)``)."""
        return self.registry


# ---------------------------------------------------------------------------
# Floor registry (instance facade: floor_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeFloorRegistry:
    """Read-only floor registry facade mirroring ``FloorRegistry`` instance methods."""

    floors: Mapping[str, SafeFloorEntry]

    def async_get_floor(self, floor_id: str) -> SafeFloorEntry | None:
        """Return the floor entry for ``floor_id``, or None."""
        return self.floors.get(floor_id)

    def async_get_floor_by_name(self, name: str) -> SafeFloorEntry | None:
        """Return the floor whose name or alias matches ``name`` (case-insensitive)."""
        lowered = name.lower()
        for floor in self.floors.values():
            if floor.name.lower() == lowered:
                return floor
            if any(alias.lower() == lowered for alias in floor.aliases):
                return floor
        return None

    def async_list_floors(self) -> list[SafeFloorEntry]:
        """Return all floor entries."""
        return list(self.floors.values())


@dataclass(frozen=True, slots=True)
class SafeFloorModule:
    """Module-level floor registry facade mirroring ``fr`` (floor_registry)."""

    registry: SafeFloorRegistry

    def async_get(self, _hass: object) -> SafeFloorRegistry:
        """Return the floor registry instance (HA parity: ``fr.async_get(hass)``)."""
        return self.registry


# ---------------------------------------------------------------------------
# Service registry (read catalog + propose-only async_call)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeServiceRegistry:
    """Read-only service catalog + propose-only ``async_call``.

    ``has_service`` and ``async_services`` reflect the frozen service catalog.
    ``async_call`` validates the service exists against the catalog, records a
    proposed action into ``ExecutionState.proposed_actions``, and returns what
    real Home Assistant would return (``None``, or ``{}`` when
    ``return_response=True``). No real service is executed in the MVP.
    """

    services: Mapping[str, tuple[str, ...]]
    services_supports_response: Mapping[str, Mapping[str, str]]

    def has_service(self, domain: str, service: str) -> bool:
        """Return True if ``domain.service`` exists in the service catalog."""
        return service in self.services.get(domain, ())

    def async_services(self) -> dict[str, dict[str, dict[str, object]]]:
        """Return the service catalog as a nested dict mirroring HA's shape."""
        result: dict[str, dict[str, dict[str, object]]] = {}
        for domain, services in self.services.items():
            response_values = self.services_supports_response.get(domain, {})
            result[domain] = {service: {"supports_response": response_values[service]} for service in services}
        return result

    async def async_call(
        self,
        domain: str,
        service: str,
        service_data: Mapping[str, object] | None = None,
        *,
        blocking: bool = False,
        context: object | None = None,
        target: Mapping[str, object] | None = None,
        return_response: bool = False,
    ) -> JsonValueType:
        """Record a proposed service call without executing it (Option A).

        Validates the service exists and the response flags match HA's service
        call contract against the frozen catalog. Returns what real Home
        Assistant would return: ``None`` normally, or ``{}`` when
        ``return_response=True``. Costs one helper call.
        """
        del context  # Reserved for future live execution boundary.

        def _record() -> dict[str, object] | None:
            runtime = require_runtime(None)
            if not self.has_service(domain, service):
                from ..types import TranslationPlaceholders
                from .executor_support import validation_error

                raise validation_error(
                    "service_not_found",
                    cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                )
            supports_response = self.services_supports_response[domain][service]
            if return_response and not blocking:
                from ..types import TranslationPlaceholders
                from .executor_support import validation_error

                raise validation_error(
                    "service_response_requires_blocking",
                    cast(TranslationPlaceholders, {"blocking": "blocking=True"}),
                )
            if supports_response == SupportsResponse.NONE.value and return_response:
                from ..types import TranslationPlaceholders
                from .executor_support import validation_error

                raise validation_error(
                    "service_response_not_supported",
                    cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                )
            if supports_response == SupportsResponse.ONLY.value and not return_response:
                from ..types import TranslationPlaceholders
                from .executor_support import validation_error

                raise validation_error(
                    "service_lacks_response_request",
                    cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                )
            action: ProposedAction = {
                "domain": domain,
                "service": service,
                "service_data": cast(dict[str, object], json_safe(service_data)) if service_data else {},
                "target": cast(dict[str, object], json_safe(target)) if target else None,
                "blocking": blocking,
                "return_response": return_response,
            }
            runtime.state.proposed_actions.append(action)
            return {} if return_response else None

        return await helper_response(self._require_state(), "services.async_call", _record)

    def _require_state(self) -> Any:
        """Return the active runtime's execution state for helper-call budgeting."""
        return require_runtime(None).state

    def __llm_sandbox_json__(self) -> JsonValueType:
        domain_count = len(self.services)
        service_count = sum(len(s) for s in self.services.values())
        return cast(
            JsonValueType,
            {"type": "services", "domain_count": domain_count, "service_count": service_count},
        )


# ---------------------------------------------------------------------------
# Hass root facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeHass:
    """Root Home Assistant facade exposed to Monty.

    Exposes only ``states`` and ``services``. The live ``hass`` object's
    ``bus``, ``config``, ``config_entries``, ``auth``, ``loop``, ``helpers``,
    and ``data`` are intentionally absent — they are never reachable from
    the sandbox.
    """

    states: SafeStateMachine
    services: SafeServiceRegistry
    type: str = "hass"

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {"type": self.type, "states": self.states, "services": self.services},
        )


# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeLLMContext:
    """Bounded view of the Home Assistant LLM request context.

    Carries the initiating device id (when the request came from a device)
    so Monty code can resolve it through ``device_registry.async_get``.
    """

    platform: str
    context: SafeContext
    language: str | None
    assistant: str | None
    device_id: str | None
    type: str = "llm_context"

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {
                "type": self.type,
                "platform": self.platform,
                "context": self.context,
                "language": self.language,
                "assistant": self.assistant,
                "device_id": self.device_id,
            },
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_facades(
    snapshot: HomeSnapshot,
    *,
    runtime: RuntimeContext | None = None,
) -> dict[str, object]:
    """Build all Monty-visible facade globals from a snapshot.

    Returns the input dict keyed by global name: ``hass``, ``states``,
    ``er``, ``dr``, ``ar``, ``fr``, ``entity_registry``, ``device_registry``,
    ``area_registry``, ``floor_registry``, and ``llm_context``. ``llm_context``
    is added separately by the tool caller (it depends on the live request).
    """
    _ = runtime  # Reads need no runtime; services.async_call resolves it via ContextVar.
    entity_registry = SafeEntityRegistry(entities=snapshot.entities)
    device_registry = SafeDeviceRegistry(devices=snapshot.devices)
    area_registry = SafeAreaRegistry(areas=snapshot.areas)
    floor_registry = SafeFloorRegistry(floors=snapshot.floors)

    state_machine = SafeStateMachine(states=snapshot.states)
    service_registry = SafeServiceRegistry(
        services=snapshot.services,
        services_supports_response=snapshot.services_supports_response,
    )
    hass = SafeHass(states=state_machine, services=service_registry)

    return {
        "hass": hass,
        "states": state_machine,
        "er": SafeEntityModule(
            registry=entity_registry,
            entities=snapshot.entities,
            indexes=snapshot.indexes,
        ),
        "dr": SafeDeviceModule(
            registry=device_registry,
            devices=snapshot.devices,
            indexes=snapshot.indexes,
        ),
        "ar": SafeAreaModule(registry=area_registry),
        "fr": SafeFloorModule(registry=floor_registry),
        "entity_registry": entity_registry,
        "device_registry": device_registry,
        "area_registry": area_registry,
        "floor_registry": floor_registry,
    }


def build_llm_context(
    platform: str,
    context_id: str | None,
    parent_id: str | None,
    user_id: str | None,
    language: str | None,
    assistant: str | None,
    device_id: str | None,
) -> SafeLLMContext:
    """Build the bounded LLM context view from live request metadata."""
    return SafeLLMContext(
        platform=platform,
        context=SafeContext(id=context_id, parent_id=parent_id, user_id=user_id),
        language=language,
        assistant=assistant,
        device_id=device_id,
    )
