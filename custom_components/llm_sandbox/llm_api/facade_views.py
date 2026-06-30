"""Monty-facing Home Assistant-native facade views.

Each facade is a frozen dataclass that reads from a frozen ``HomeSnapshot``.
Reads mirror Home Assistant's synchronous registry/state-machine callbacks
    and cost zero helper calls. The only async method is
    ``hass.services.async_call``, which validates against the snapshot and
    executes through a private runtime invoker.

The facades intentionally expose only the documented public surface. They
never leak the live ``HomeAssistant`` object, mutable registries, the event
bus, the config, or auth into the Monty sandbox.
"""


# ruff: noqa: D105, ANN401

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import voluptuous as vol
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.util.json import JsonValueType

from ..snapshot.models import (
    HomeSnapshot,
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
    ServiceSchemaBrief,
    SnapshotIndexes,
)
from ..types import ActionRecord, ProposedAction, TranslationPlaceholders
from .errors import HelperExecutionError
from .executor_support import helper_response, json_safe
from .runtime import require_runtime, require_snapshot

_TARGET_SELECTOR_KEYS = frozenset(("entity_id", "device_id", "area_id", "label_id", "label", "floor_id"))

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

    def async_get_entity(
        self,
        registry: SafeEntityRegistry,
        domain: str,
        platform: str,
        unique_id: str,
    ) -> str | None:
        """Return the entity_id matching (domain, platform, unique_id), or None."""
        return registry.async_get_entity_id(domain, platform, unique_id)

    def async_entries(
        self,
        registry: SafeEntityRegistry,
    ) -> list[SafeRegistryEntry]:
        """Return all entity registry entries."""
        del registry
        return list(self.entities.values())


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

    def async_entries_for_label(
        self,
        registry: SafeDeviceRegistry,
        label_id: str,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries carrying ``label_id``."""
        del registry
        device_ids = self.indexes.device_ids_by_label.get(label_id, ())
        return [self.devices[did] for did in device_ids if did in self.devices]


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
# Label registry (instance facade: label_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeLabelRegistry:
    """Read-only label registry facade mirroring ``LabelRegistry`` instance methods."""

    labels: Mapping[str, SafeLabelEntry]

    def async_get_label(self, label_id: str) -> SafeLabelEntry | None:
        """Return the label entry for ``label_id``, or None."""
        return self.labels.get(label_id)

    def async_get_label_by_name(self, name: str) -> SafeLabelEntry | None:
        """Return the label whose normalized name matches ``name``."""
        normalized = name.casefold().replace(" ", "")
        for label in self.labels.values():
            if label.normalized_name == normalized:
                return label
        return None

    def async_list_labels(self) -> list[SafeLabelEntry]:
        """Return all label entries."""
        return list(self.labels.values())


@dataclass(frozen=True, slots=True)
class SafeLabelModule:
    """Module-level label registry facade mirroring ``lr`` (label_registry)."""

    registry: SafeLabelRegistry

    def async_get(self, _hass: object) -> SafeLabelRegistry:
        """Return the label registry instance (HA parity: ``lr.async_get(hass)``)."""
        return self.registry


# ---------------------------------------------------------------------------
# Category registry (instance facade: category_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeCategoryRegistry:
    """Read-only category registry facade mirroring ``CategoryRegistry`` instance methods."""

    categories: Mapping[str, Mapping[str, SafeCategoryEntry]]

    def async_get_category(self, *, scope: str, category_id: str) -> SafeCategoryEntry | None:
        """Return the category entry for ``scope``/``category_id``, or None."""
        return self.categories.get(scope, {}).get(category_id)

    def async_list_categories(self, *, scope: str) -> list[SafeCategoryEntry]:
        """Return all category entries within ``scope``."""
        return list(self.categories.get(scope, {}).values())


@dataclass(frozen=True, slots=True)
class SafeCategoryModule:
    """Module-level category registry facade mirroring ``cr`` (category_registry)."""

    registry: SafeCategoryRegistry

    def async_get(self, _hass: object) -> SafeCategoryRegistry:
        """Return the category registry instance (HA parity: ``cr.async_get(hass)``)."""
        return self.registry


# ---------------------------------------------------------------------------
# Repairs issue registry (instance facade: repairs global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeIssueRegistry:
    """Read-only repairs facade over the frozen issue registry snapshot."""

    issues: list[SafeIssueEntry]

    def async_issues(self) -> list[SafeIssueEntry]:
        """Return all repairs issues."""
        return list(self.issues)

    def async_get_issue(self, domain: str, issue_id: str) -> SafeIssueEntry | None:
        """Return the issue for ``domain``/``issue_id``, or None."""
        for issue in self.issues:
            if issue.domain == domain and issue.issue_id == issue_id:
                return issue
        return None

    def async_issues_for_domain(self, domain: str) -> list[SafeIssueEntry]:
        """Return all issues raised by ``domain``."""
        return [issue for issue in self.issues if issue.domain == domain]

    def async_issues_by_severity(self, severity: str) -> list[SafeIssueEntry]:
        """Return issues whose severity value equals ``severity``."""
        return [issue for issue in self.issues if issue.severity == severity]

    def async_active_issues(self) -> list[SafeIssueEntry]:
        """Return issues that are currently active."""
        return [issue for issue in self.issues if issue.active]

    def async_dismissed_issues(self) -> list[SafeIssueEntry]:
        """Return issues the user has dismissed."""
        return [issue for issue in self.issues if issue.dismissed_version is not None]


# ---------------------------------------------------------------------------
# Persistent notifications (instance facade: persistent_notifications global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeNotificationRegistry:
    """Read-only facade over the frozen persistent-notification snapshot."""

    notifications: list[SafeNotificationEntry]

    def async_get_notifications(self) -> list[SafeNotificationEntry]:
        """Return all persistent notifications."""
        return list(self.notifications)

    def async_get_notification(self, notification_id: str) -> SafeNotificationEntry | None:
        """Return the persistent notification for ``notification_id``, or None."""
        for notification in self.notifications:
            if notification.notification_id == notification_id:
                return notification
        return None


# ---------------------------------------------------------------------------
# Config entries (instance facade: config_entries global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeConfigEntries:
    """Read-only config-entries facade mirroring HA list/get methods."""

    entries: list[SafeConfigEntry]

    def async_entries(self, domain: str | None = None) -> list[SafeConfigEntry]:
        """Return all entries, optionally filtered by ``domain`` (HA parity)."""
        if domain is None:
            return list(self.entries)
        return [entry for entry in self.entries if entry.domain == domain]

    def async_get_entry(self, entry_id: str) -> SafeConfigEntry | None:
        """Return the entry for ``entry_id``, or None (HA parity)."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None


# ---------------------------------------------------------------------------
# Service registry (read catalog + live async_call boundary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafeServiceRegistry:
    """Read-only service catalog + live ``async_call``.

    ``has_service`` and ``async_services`` reflect the frozen service catalog.
    ``async_call`` validates the service against the catalog and visible target
    indexes, records an action outcome, and invokes live Home Assistant through
    the private runtime callable.
    """

    services: Mapping[str, tuple[str, ...]]
    services_supports_response: Mapping[str, Mapping[str, str]]
    services_schema: Mapping[str, Mapping[str, ServiceSchemaBrief]]

    def __init__(
        self,
        *,
        services: Mapping[str, tuple[str, ...]],
        services_supports_response: Mapping[str, Mapping[str, str]],
        services_schema: Mapping[str, Mapping[str, ServiceSchemaBrief]],
        snapshot: HomeSnapshot | None = None,
    ) -> None:
        """Initialize with snapshot data kept outside dataclass fields.

        ``services_schema`` is a declared field rather than read from the
        private ``_snapshot`` stash because the synchronous catalog reads
        (``async_services`` / ``async_services_for_domain``) run inside the
        Monty VM, where neither the runtime snapshot contextvar nor the
        ``_snapshot_value`` stash (dropped when Monty reconstructs inputs from
        dataclass fields) is available. The async ``async_call`` path still
        uses ``_snapshot`` because coroutines execute on the host event loop
        where the runtime contextvar is active.
        """
        if snapshot is not None:
            object.__setattr__(self, "_snapshot_value", snapshot)
        object.__setattr__(self, "services", services)
        object.__setattr__(self, "services_supports_response", services_supports_response)
        object.__setattr__(self, "services_schema", services_schema)

    @property
    def _snapshot(self) -> HomeSnapshot:
        """Return the private source snapshot for validation helpers."""
        if "_snapshot_value" in self.__dict__:
            return cast(HomeSnapshot, self.__dict__["_snapshot_value"])
        return require_snapshot()

    def has_service(self, domain: str, service: str) -> bool:
        """Return True if ``domain.service`` exists in the service catalog."""
        return service in self.services.get(domain, ())

    def async_services(self) -> dict[str, dict[str, dict[str, object]]]:
        """Return the service catalog as a nested dict mirroring HA's shape."""
        return {domain: self.async_services_for_domain(domain) for domain in self.services}

    def async_services_for_domain(self, domain: str) -> dict[str, dict[str, object]]:
        """Return JSON-safe service metadata for one domain from the snapshot."""
        response_values = self.services_supports_response.get(domain, {})
        briefs = self.services_schema.get(domain, {})
        return {
            service: {
                "supports_response": response_values[service],
                **briefs.get(service, {"fields": [], "dynamic": False}),
            }
            for service in self.services.get(domain, ())
        }

    def supports_response(self, domain: str, service: str) -> str:
        """Return the response mode value for ``domain.service`` from the snapshot."""
        return self.services_supports_response.get(domain.lower(), {}).get(
            service.lower(), SupportsResponse.NONE.value
        )

    async def async_call(
        self,
        domain: str,
        service: str,
        service_data: Mapping[str, object] | None = None,
        blocking: bool = False,
        context: object | None = None,  # noqa: ARG002
        target: Mapping[str, object] | None = None,
        return_response: bool = False,
    ) -> JsonValueType:
        """Validate, execute, and record one service call outcome.

        The sandbox-supplied ``context`` is intentionally ignored. The private
        invoker supplies the real Home Assistant context for attribution.
        """

        async def _call() -> object:
            runtime = require_runtime(None)
            settings = runtime.settings
            cleaned_service_data, merged_target = _extract_target_selectors(service_data, target)
            raw_target = cast(dict[str, object], json_safe(merged_target)) if merged_target is not None else None

            def _request_action(action_target: dict[str, object] | None) -> ProposedAction:
                return {
                    "domain": domain,
                    "service": service,
                    "service_data": cleaned_service_data,
                    "target": action_target,
                    "blocking": blocking,
                    "return_response": return_response,
                }

            def _raise_validation(
                key: str,
                placeholders: TranslationPlaceholders,
                *,
                hints: Mapping[str, object] | None = None,
            ) -> None:
                action = _request_action(raw_target)
                runtime.state.actions.append(
                    _action_record(
                        action,
                        status="error",
                        response=None,
                        error=_action_error(key, placeholders, key),
                    )
                )
                raise HelperExecutionError("services.async_call", key, placeholders, hints=hints)

            # Action master switch: refuse all service calls when disabled.
            if not settings.actions_enabled:
                _raise_validation("actions_disabled", {})
            # Domain allowlist: empty means all domains allowed.
            if settings.action_domains and domain not in settings.action_domains:
                _raise_validation(
                    "action_domain_not_allowed",
                    cast(TranslationPlaceholders, {"domain": domain}),
                )
            if not self.has_service(domain, service):
                _raise_validation(
                    "service_not_found",
                    cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                    hints={"available_services": self._snapshot.services_schema.get(domain, {})},
                )
            supports_response = self.services_supports_response[domain][service]
            if return_response and not blocking:
                _raise_validation(
                    "service_response_requires_blocking",
                    cast(TranslationPlaceholders, {"blocking": "blocking=True"}),
                )
            if supports_response == SupportsResponse.NONE.value and return_response:
                _raise_validation(
                    "service_response_not_supported",
                    cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                )
            if supports_response == SupportsResponse.ONLY.value and not return_response:
                _raise_validation(
                    "service_lacks_response_request",
                    cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                )
            try:
                resolved_target = self._visible_target(merged_target)
            except HelperExecutionError as err:
                runtime.state.actions.append(
                    _action_record(
                        _request_action(raw_target),
                        status="error",
                        response=None,
                        error=_action_error(err.key, err.placeholders, err.key),
                    )
                )
                raise
            action = _request_action(resolved_target)
            record = _action_record(action, status="ok", response=None, error=None)
            runtime.state.actions.append(record)
            remaining = runtime.deadline - time.monotonic()
            # Mutate the just-recorded action before raising when no per-call budget remains.
            if remaining <= 0:
                error = _action_error(
                    "service_call_timeout",
                    {"domain": domain, "service": service},
                    "Service call timed out before execution",
                )
                record["status"] = "error"
                record["error"] = error
                raise HelperExecutionError(
                    "services.async_call",
                    "service_call_timeout",
                    {"domain": domain, "service": service},
                )
            try:
                result = await asyncio.wait_for(runtime.invoke(action), timeout=remaining)
            except TimeoutError as err:
                error = _action_error("service_call_timeout", {"domain": domain, "service": service}, str(err))
                record["status"] = "error"
                record["error"] = error
                raise HelperExecutionError(
                    "services.async_call",
                    "service_call_timeout",
                    {"domain": domain, "service": service},
                ) from err
            except ServiceValidationError as err:
                helper_err = self._service_call_error(err, domain, service)
                record["status"] = "error"
                record["error"] = _action_error(helper_err.key, helper_err.placeholders, str(err))
                raise helper_err from err
            except vol.Invalid as err:
                helper_err = self._service_call_error(err, domain, service)
                record["status"] = "error"
                record["error"] = _action_error(helper_err.key, helper_err.placeholders, str(err))
                raise helper_err from err
            except HomeAssistantError as err:
                helper_err = self._service_call_error(err, domain, service)
                record["status"] = "error"
                record["error"] = _action_error(helper_err.key, helper_err.placeholders, str(err))
                raise helper_err from err
            except Exception as err:
                helper_err = self._service_call_error(err, domain, service)
                record["status"] = "error"
                record["error"] = _action_error(helper_err.key, helper_err.placeholders, str(err))
                raise helper_err from err
            record["response"] = json_safe(result) if return_response else None
            return result

        return await helper_response(self._require_state(), "services.async_call", _call)

    def _visible_target(self, target: Mapping[str, object] | None) -> dict[str, object] | None:
        """Resolve supported HA target selectors to visible entity IDs."""
        if not target:
            return cast(dict[str, object] | None, json_safe(target))

        indexes = self._snapshot.indexes
        entity_ids: set[str] = set()
        supported_values: list[str] = []
        supported_keys: list[str] = []

        if "entity_id" in target:
            supported_keys.append("entity_id")
            for entity_id in _target_values(target["entity_id"]):
                supported_values.append(entity_id)
                if entity_id not in self._snapshot.states:
                    raise HelperExecutionError(
                        "services.async_call",
                        "service_target_not_visible",
                        {"entity_id": entity_id},
                    )
                entity_ids.add(entity_id)
        if "device_id" in target:
            supported_keys.append("device_id")
            for device_id in _target_values(target["device_id"]):
                supported_values.append(device_id)
                entity_ids.update(indexes.entity_ids_by_device_id.get(device_id, ()))
        if "area_id" in target:
            supported_keys.append("area_id")
            for area_id in _target_values(target["area_id"]):
                supported_values.append(area_id)
                entity_ids.update(indexes.entity_ids_by_area_id.get(area_id, ()))
        if "label_id" in target:
            supported_keys.append("label_id")
            for label_id in _target_values(target["label_id"]):
                supported_values.append(label_id)
                entity_ids.update(indexes.entity_ids_by_label.get(label_id, ()))
        if "label" in target:
            supported_keys.append("label")
            for label_id in _target_values(target["label"]):
                supported_values.append(label_id)
                entity_ids.update(indexes.entity_ids_by_label.get(label_id, ()))
        if "floor_id" in target:
            supported_keys.append("floor_id")
            for floor_id in _target_values(target["floor_id"]):
                supported_values.append(floor_id)
                for area_id in indexes.area_ids_by_floor_id.get(floor_id, ()):
                    entity_ids.update(indexes.entity_ids_by_area_id.get(area_id, ()))

        if entity_ids:
            return {"entity_id": sorted(entity_ids)}
        if supported_values:
            raise HelperExecutionError(
                "services.async_call",
                "service_target_not_visible",
                {"entity_id": supported_values[0]},
            )
        if supported_keys:
            raise HelperExecutionError(
                "services.async_call",
                "service_target_not_visible",
                {"entity_id": supported_keys[0]},
            )
        return cast(dict[str, object], json_safe(target))

    def _service_call_error(
        self,
        err: Exception,
        domain: str,
        service: str,
    ) -> HelperExecutionError:
        """Classify live Home Assistant service-call and schema failures."""
        translation_key = getattr(err, "translation_key", None)
        if translation_key is None:
            key = "service_call_failed"
            placeholders: TranslationPlaceholders = {
                "domain": domain,
                "service": service,
                "reason": err.__class__.__name__,
            }
        else:
            key = str(translation_key)
            raw_placeholders = getattr(err, "translation_placeholders", None)
            if isinstance(raw_placeholders, Mapping):
                placeholders = {str(item_key): str(value) for item_key, value in raw_placeholders.items()}
            else:
                placeholders = {"domain": domain, "service": service, "reason": key}
        hints = None
        if (brief := self._snapshot.services_schema.get(domain, {}).get(service)) is not None:
            hints = {"available_services": {service: brief}}
        return HelperExecutionError("services.async_call", key, placeholders, hints=hints)

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

    Exposes only frozen ``states``, ``services``, and ``config`` snapshots. The
    live ``hass`` object's ``bus``, ``config_entries``, ``auth``, ``loop``, ``helpers``,
    and ``data`` are intentionally absent — they are never reachable from
    the sandbox.
    """

    states: SafeStateMachine
    services: SafeServiceRegistry
    config: SafeConfig
    type: str = "hass"

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(
            JsonValueType,
            {"type": self.type, "states": self.states, "services": self.services, "config": self.config},
        )


# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeLLMContext:
    """Bounded view of the Home Assistant LLM request context.

    Carries the initiating device id and derived location ids (when the
    request came from a device assigned to an area/floor) so Monty code can
    scope ambiguous local requests without touching live registries.
    """

    platform: str
    context: SafeContext
    language: str | None
    assistant: str | None
    device_id: str | None
    area_id: str | None
    area_name: str | None
    floor_id: str | None
    floor_name: str | None
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
                "area_id": self.area_id,
                "area_name": self.area_name,
                "floor_id": self.floor_id,
                "floor_name": self.floor_name,
            },
        )


# ---------------------------------------------------------------------------
# Date/datetime value objects and facades
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeDate:
    """Frozen date value returned by the ``date`` facade.

    Stores parsed calendar components from a single datetime. All fields are
    JSON-safe primitives.
    """

    iso: str
    year: int
    month: int
    day: int
    weekday: int

    def isoformat(self) -> str:
        """Return the date as an ISO 8601 string (YYYY-MM-DD)."""
        return self.iso

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, self.iso)


@dataclass(frozen=True, slots=True)
class SafeDateTime:
    """Frozen datetime value returned by the ``datetime`` facade.

    Stores parsed datetime components from a single datetime. All fields are
    JSON-safe primitives.
    """

    iso: str
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    weekday: int

    def date(self) -> SafeDate:
        """Return the calendar-date portion as a SafeDate."""
        return SafeDate(
            iso=self.iso[:10],
            year=self.year,
            month=self.month,
            day=self.day,
            weekday=self.weekday,
        )

    def isoformat(self) -> str:
        """Return the datetime as an ISO 8601 string."""
        return self.iso

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, self.iso)


@dataclass(frozen=True, slots=True)
class SafeDateFacade:
    """Frozen date class facade exposed as the ``date`` Monty global.

    ``today()`` returns the frozen snapshot date. ``fromisoformat()`` parses
    a caller-supplied ISO date string. No live wall-clock access.
    """

    today_value: SafeDate

    def today(self) -> SafeDate:
        """Return the frozen snapshot date in the configured HA timezone."""
        return self.today_value

    def fromisoformat(self, date_string: str) -> SafeDate:
        """Parse an ISO 8601 date string into a SafeDate.

        Mirrors stdlib date.fromisoformat: a datetime string (containing a time
        component) is rejected rather than silently truncated.
        """
        parsed = _date.fromisoformat(date_string)
        return SafeDate(
            iso=parsed.isoformat(),
            year=parsed.year,
            month=parsed.month,
            day=parsed.day,
            weekday=parsed.weekday(),
        )


@dataclass(frozen=True, slots=True)
class SafeDateTimeFacade:
    """Frozen datetime class facade exposed as the ``datetime`` Monty global.

    ``now()`` returns the frozen snapshot datetime in the HA timezone.
    ``utcnow()`` returns the UTC snapshot datetime. ``fromisoformat()`` parses
    a caller-supplied ISO datetime string. No live wall-clock access.
    """

    now_value: SafeDateTime
    utcnow_value: SafeDateTime

    def now(self, tz: object = None) -> SafeDateTime:
        """Return the frozen snapshot datetime in the configured HA timezone."""
        del tz  # API parity; frozen time cannot honor a caller-supplied timezone.
        return self.now_value

    def utcnow(self) -> SafeDateTime:
        """Return the frozen snapshot datetime in UTC."""
        return self.utcnow_value

    def fromisoformat(self, date_string: str) -> SafeDateTime:
        """Parse an ISO 8601 datetime string into a SafeDateTime."""
        return _datetime_from_dt(_datetime.fromisoformat(date_string))


def _date_from_datetime(dt: _datetime) -> SafeDate:
    """Build a SafeDate from a parsed datetime, preserving the calendar date."""
    return SafeDate(
        iso=dt.strftime("%Y-%m-%d"),
        year=dt.year,
        month=dt.month,
        day=dt.day,
        weekday=dt.weekday(),
    )


def _datetime_from_dt(dt: _datetime) -> SafeDateTime:
    """Build a SafeDateTime from a parsed datetime, preserving all components."""
    return SafeDateTime(
        iso=dt.isoformat(),
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
        microsecond=dt.microsecond,
        weekday=dt.weekday(),
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_facades(
    snapshot: HomeSnapshot,
) -> dict[str, object]:
    """Build all Monty-visible facade globals from a snapshot.

    Returns the input dict keyed by global name: ``hass``, ``states``,
    registry/module facades, ``repairs``, ``persistent_notifications``,
    ``config_entries``, date/time facades, and ``now``. ``llm_context`` is
    added separately by the tool caller (it depends on the live request).
    """
    entity_registry = SafeEntityRegistry(entities=snapshot.entities)
    device_registry = SafeDeviceRegistry(devices=snapshot.devices)
    area_registry = SafeAreaRegistry(areas=snapshot.areas)
    floor_registry = SafeFloorRegistry(floors=snapshot.floors)
    label_registry = SafeLabelRegistry(labels=snapshot.labels)
    category_registry = SafeCategoryRegistry(categories=snapshot.categories)
    repairs = SafeIssueRegistry(issues=list(snapshot.issues))
    persistent_notifications = SafeNotificationRegistry(notifications=list(snapshot.notifications))
    config_entries = SafeConfigEntries(entries=list(snapshot.config_entries))

    state_machine = SafeStateMachine(states=snapshot.states)
    service_registry = SafeServiceRegistry(
        snapshot=snapshot,
        services=snapshot.services,
        services_supports_response=snapshot.services_supports_response,
        services_schema=snapshot.services_schema,
    )
    hass = SafeHass(states=state_machine, services=service_registry, config=snapshot.config)

    created = _datetime.fromisoformat(snapshot.created_at)
    # hass.config.time_zone is validated by Home Assistant; trust it directly so
    # an invalid timezone surfaces as an error instead of silently falling back.
    local = created.astimezone(ZoneInfo(snapshot.config.time_zone))

    date_facade = SafeDateFacade(today_value=_date_from_datetime(local))
    datetime_facade = SafeDateTimeFacade(
        now_value=_datetime_from_dt(local),
        utcnow_value=_datetime_from_dt(created),
    )

    return {
        "hass": hass,
        "states": state_machine,
        "date": date_facade,
        "datetime": datetime_facade,
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
        "lr": SafeLabelModule(registry=label_registry),
        "cr": SafeCategoryModule(registry=category_registry),
        "entity_registry": entity_registry,
        "device_registry": device_registry,
        "area_registry": area_registry,
        "floor_registry": floor_registry,
        "label_registry": label_registry,
        "category_registry": category_registry,
        "repairs": repairs,
        "persistent_notifications": persistent_notifications,
        "config_entries": config_entries,
        "now": snapshot.created_at,
    }


def build_llm_context(
    platform: str,
    context_id: str | None,
    parent_id: str | None,
    user_id: str | None,
    language: str | None,
    assistant: str | None,
    device_id: str | None,
    area_id: str | None,
    area_name: str | None,
    floor_id: str | None,
    floor_name: str | None,
) -> SafeLLMContext:
    """Build the bounded LLM context view from live request metadata."""
    return SafeLLMContext(
        platform=platform,
        context=SafeContext(id=context_id, parent_id=parent_id, user_id=user_id),
        language=language,
        assistant=assistant,
        device_id=device_id,
        area_id=area_id,
        area_name=area_name,
        floor_id=floor_id,
        floor_name=floor_name,
    )


def _target_values(value: object) -> list[str]:
    """Return HA target selector values as strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _extract_target_selectors(
    service_data: Mapping[str, object] | None,
    target: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Move HA target selector keys from service data into the target mapping."""
    raw_service_data = dict(service_data) if service_data is not None else {}
    extracted_target = {
        key: raw_service_data.pop(key) for key in tuple(raw_service_data) if key in _TARGET_SELECTOR_KEYS
    }
    raw_target = dict(target) if target is not None else {}

    # Explicit target values win over selector values supplied inside service data.
    merged_target = extracted_target | raw_target
    cleaned_service_data = cast(dict[str, object], json_safe(raw_service_data)) if raw_service_data else None
    return cleaned_service_data, cast(dict[str, object], json_safe(merged_target)) if merged_target else None


def _action_error(key: str, placeholders: TranslationPlaceholders, message: str) -> dict[str, object]:
    """Build the JSON-safe action error shape."""
    return {"key": key, "placeholders": placeholders, "message": message}


def _action_record(
    action: ProposedAction,
    *,
    status: str,
    response: object,
    error: dict[str, object] | None,
) -> ActionRecord:
    """Build one mutable service action record."""
    return {
        "domain": action["domain"],
        "service": action["service"],
        "service_data": action["service_data"],
        "target": action["target"],
        "blocking": action["blocking"],
        "return_response": action["return_response"],
        "status": status,
        "response": response,
        "error": error,
    }
