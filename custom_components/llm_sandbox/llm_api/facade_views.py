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

from homeassistant.core import SupportsResponse
from homeassistant.util.json import JsonValueType

from ..runtime import SandboxSettings
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
from .resolution import _DISCOVERY_LIMIT, CandidateTarget, candidates_for_domain, resolve_target_entity
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
# Entity registry (er/entity_registry globals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeEntityRegistry:
    """Read-only entity registry facade mirroring HA module and instance methods."""

    entities: Mapping[str, SafeRegistryEntry]
    indexes: SnapshotIndexes

    def async_get(self, key: object = None) -> Any:
        """Return an entry for string IDs, otherwise return this registry.

        This accepts both HA idioms: ``er.async_get(hass)`` resolves the
        registry, while ``entity_registry.async_get('<entity_id>')`` resolves a
        single entry. Treating every non-string as registry resolution avoids
        leaking hash/type errors when LLMs pass the HA ``hass`` facade.
        """
        if isinstance(key, str):
            # String arguments mean the instance lookup idiom.
            return self.entities.get(key)
        # Non-string arguments include hass/module ceremony; return the registry.
        return self

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

    def async_entries_for_area(
        self,
        registry_or_area_id: Any,
        area_id: str | None = None,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries whose effective area is ``area_id``.

        Effective area = ``entity.area_id or device.area_id`` (entity override wins).
        The HA-native two-arg form passes the registry first; the one-arg form
        omits it. Either is accepted.
        """
        if area_id is None:
            area_id = registry_or_area_id
        entity_ids = self.indexes.entity_ids_by_area_id.get(area_id, ())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def async_entries_for_device(
        self,
        registry_or_device_id: Any,
        device_id: str | None = None,
        include_disabled_entities: bool = False,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries linked to ``device_id``."""
        if device_id is None:
            device_id = registry_or_device_id
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
        registry_or_config_entry_id: Any,
        config_entry_id: str | None = None,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries created by ``config_entry_id``."""
        if config_entry_id is None:
            config_entry_id = registry_or_config_entry_id
        entity_ids = self.indexes.entity_ids_by_config_entry_id.get(config_entry_id, ())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def async_entries_for_label(
        self,
        registry_or_label_id: Any,
        label_id: str | None = None,
    ) -> list[SafeRegistryEntry]:
        """Return all entity entries carrying ``label_id``."""
        if label_id is None:
            label_id = registry_or_label_id
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
        del registry
        return self.async_get_entity_id(domain, platform, unique_id)

    def async_entries(self, registry: Any = None) -> list[SafeRegistryEntry]:
        """Return all entity registry entries."""
        del registry
        return list(self.entities.values())


# ---------------------------------------------------------------------------
# Device registry (dr/device_registry globals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeDeviceRegistry:
    """Read-only device registry facade mirroring HA module and instance methods."""

    devices: Mapping[str, SafeDeviceEntry]
    indexes: SnapshotIndexes

    def async_get(self, key: object = None) -> Any:
        """Return an entry for string IDs, otherwise return this registry."""
        if isinstance(key, str):
            # String arguments mean the instance lookup idiom.
            return self.devices.get(key)
        # Non-string arguments include hass/module ceremony; return the registry.
        return self

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

    def async_entries_for_area(
        self,
        registry_or_area_id: Any,
        area_id: str | None = None,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries assigned to ``area_id``."""
        if area_id is None:
            area_id = registry_or_area_id
        device_ids = self.indexes.device_ids_by_area_id.get(area_id, ())
        return [self.devices[did] for did in device_ids if did in self.devices]

    def async_entries_for_config_entry(
        self,
        registry_or_config_entry_id: Any,
        config_entry_id: str | None = None,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries linked to ``config_entry_id``."""
        if config_entry_id is None:
            config_entry_id = registry_or_config_entry_id
        return [d for d in self.devices.values() if config_entry_id in d.config_entries]

    def async_entries_for_label(
        self,
        registry_or_label_id: Any,
        label_id: str | None = None,
    ) -> list[SafeDeviceEntry]:
        """Return all device entries carrying ``label_id``."""
        if label_id is None:
            label_id = registry_or_label_id
        device_ids = self.indexes.device_ids_by_label.get(label_id, ())
        return [self.devices[did] for did in device_ids if did in self.devices]


# ---------------------------------------------------------------------------
# Area registry (instance facade: area_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeAreaRegistry:
    """Read-only area registry facade mirroring ``AreaRegistry`` instance methods."""

    areas: Mapping[str, SafeAreaEntry]

    def async_get(self, _key: object = None) -> Any:
        """Return this registry (HA parity: ``ar.async_get(hass)``)."""
        return self

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


# ---------------------------------------------------------------------------
# Floor registry (instance facade: floor_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeFloorRegistry:
    """Read-only floor registry facade mirroring ``FloorRegistry`` instance methods."""

    floors: Mapping[str, SafeFloorEntry]

    def async_get(self, _key: object = None) -> Any:
        """Return this registry (HA parity: ``fr.async_get(hass)``)."""
        return self

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


# ---------------------------------------------------------------------------
# Label registry (instance facade: label_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeLabelRegistry:
    """Read-only label registry facade mirroring ``LabelRegistry`` instance methods."""

    labels: Mapping[str, SafeLabelEntry]

    def async_get(self, _key: object = None) -> Any:
        """Return this registry (HA parity: ``lr.async_get(hass)``)."""
        return self

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


# ---------------------------------------------------------------------------
# Category registry (instance facade: category_registry global)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafeCategoryRegistry:
    """Read-only category registry facade mirroring ``CategoryRegistry`` instance methods."""

    categories: Mapping[str, Mapping[str, SafeCategoryEntry]]

    def async_get(self, _key: object = None) -> Any:
        """Return this registry (HA parity: ``cr.async_get(hass)``)."""
        return self

    def async_get_category(self, *, scope: str, category_id: str) -> SafeCategoryEntry | None:
        """Return the category entry for ``scope``/``category_id``, or None."""
        return self.categories.get(scope, {}).get(category_id)

    def async_list_categories(self, *, scope: str) -> list[SafeCategoryEntry]:
        """Return all category entries within ``scope``."""
        return list(self.categories.get(scope, {}).values())


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


@dataclass(frozen=True, slots=True)
class SafeServiceRegistry:
    """Read-only service catalog + live ``async_call``.

    Sync catalog reads use the declared frozen catalog fields. ``async_call``
    resolves the active runtime snapshot for target validation, records an
    action outcome, and invokes live Home Assistant through the private runtime
    callable.
    """

    services: Mapping[str, tuple[str, ...]]
    services_supports_response: Mapping[str, Mapping[str, str]]
    services_schema: Mapping[str, Mapping[str, ServiceSchemaBrief]]

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

    def _policy_block(
        self,
        settings: SandboxSettings,
        domain: str,
        service: str,
        blocking: bool,
        return_response: bool,
    ) -> _PolicyBlock | None:
        """Evaluate snapshot-policy gates without raising; None means the call may proceed."""
        if not settings.actions_enabled:
            return _PolicyBlock("actions_disabled", {}, message="Service calls are disabled for this sandbox.")
        if settings.action_domains and domain not in settings.action_domains:
            valid_domains = _bounded_strings(sorted(settings.action_domains))
            return _PolicyBlock(
                "action_domain_not_allowed",
                cast(TranslationPlaceholders, {"domain": domain}),
                message=_valid_domains_message(domain, valid_domains),
                fix=valid_domains,
            )
        if not self.has_service(domain, service):
            if not self.services.get(domain):
                valid_domains = _bounded_strings(sorted(self.services))
                return _PolicyBlock(
                    "service_not_found",
                    cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                    message=_valid_domains_message(domain, valid_domains),
                    fix=valid_domains,
                )
            valid_services = _bounded_strings(sorted(self.services[domain]))
            return _PolicyBlock(
                "service_not_found",
                cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                message=_valid_services_message(domain, service, valid_services),
                fix=valid_services,
            )
        supports_response = self.services_supports_response[domain][service]
        if return_response and not blocking:
            return _PolicyBlock(
                "service_response_requires_blocking",
                cast(TranslationPlaceholders, {"blocking": "blocking=True"}),
                message="Set blocking=True when requesting a service response.",
            )
        if supports_response == SupportsResponse.NONE.value and return_response:
            return _PolicyBlock(
                "service_response_not_supported",
                cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                message=f"Service '{domain}.{service}' does not support return_response=True.",
            )
        if supports_response == SupportsResponse.ONLY.value and not return_response:
            return _PolicyBlock(
                "service_lacks_response_request",
                cast(TranslationPlaceholders, {"return_response": "return_response=True"}),
                message=f"Service '{domain}.{service}' requires return_response=True.",
            )
        return None

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
            cleaned_service_data, merged_target, selector_adjustments = _extract_target_selectors(service_data, target)
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

            def _block(
                key: str,
                _placeholders: TranslationPlaceholders,
                *,
                message: str,
                fix: list[str] | None = None,
            ) -> None:
                # Policy blocks are non-raising: record an errored action and let
                # the call return None so execution stays status="ok" with a
                # recorded errored action (Decision 3: live failures keep raising).
                runtime.state.actions.append(
                    _action_record(
                        _request_action(raw_target),
                        status="error",
                        response=None,
                        error=_action_error(key, message, fix=fix),
                    )
                )

            # Policy gate (non-raising).
            if (block := self._policy_block(settings, domain, service, blocking, return_response)) is not None:
                _block(block.key, block.placeholders, message=block.message, fix=block.fix)
                return None

            # Target visibility resolution with auto-resolve.
            target_outcome = self._visible_target(merged_target, domain)
            if isinstance(target_outcome, _UnresolvedTarget):
                _block(
                    "service_target_not_visible",
                    cast(TranslationPlaceholders, {"entity_id": target_outcome.requested}),
                    message=target_outcome.hint,
                    fix=target_outcome.fix,
                )
                return None
            resolved_target = target_outcome.target

            action = _request_action(resolved_target)
            record = _action_record(
                action,
                status="ok",
                response=None,
                error=None,
                adjustments=[*selector_adjustments, *target_outcome.adjustments],
            )
            runtime.state.actions.append(record)
            remaining = runtime.deadline - time.monotonic()
            # Mutate the just-recorded action before raising when no per-call budget remains.
            if remaining <= 0:
                error = _action_error(
                    "service_call_timeout",
                    f"Service '{domain}.{service}' timed out before execution.",
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
                error = _action_error(
                    "service_call_timeout",
                    f"Service '{domain}.{service}' timed out during execution.",
                )
                record["status"] = "error"
                record["error"] = error
                raise HelperExecutionError(
                    "services.async_call",
                    "service_call_timeout",
                    {"domain": domain, "service": service},
                ) from err
            except Exception as err:
                helper_err = self._service_call_error(err, domain, service)
                record["status"] = "error"
                record["error"] = _action_error(
                    helper_err.key,
                    f"Service '{domain}.{service}' failed validation or execution: {err.__class__.__name__}.",
                    fix=_service_field_names(self.services_schema.get(domain, {}).get(service)),
                )
                raise helper_err from err
            if return_response:
                record["response"] = json_safe(result)
            return result

        return await helper_response(self._require_state(), "services.async_call", _call)

    def _visible_target(
        self,
        target: Mapping[str, object] | None,
        domain: str,
    ) -> _ResolvedTarget | _UnresolvedTarget:
        """Resolve supported HA target selectors to visible entity IDs."""
        snapshot = require_snapshot()
        if not target:
            return _ResolvedTarget(cast(dict[str, object] | None, json_safe(target)))

        indexes = snapshot.indexes
        entity_ids: set[str] = set()
        supported_values: list[str] = []
        supported_keys: list[str] = []
        adjustments: list[dict[str, object]] = []

        if "entity_id" in target:
            supported_keys.append("entity_id")
            for entity_id in _target_values(target["entity_id"]):
                supported_values.append(entity_id)
                if entity_id in snapshot.states:
                    entity_ids.add(entity_id)
                    continue
                resolve_domain = entity_id.split(".", 1)[0] if "." in entity_id else domain
                outcome = resolve_target_entity(snapshot, entity_id, resolve_domain)
                if outcome.is_resolved:
                    resolved_entity_id = cast(str, outcome.resolved)
                    entity_ids.add(resolved_entity_id)
                    adjustments.append(_target_entity_resolved_adjustment(entity_id, resolved_entity_id))
                else:
                    candidates: tuple[CandidateTarget, ...] = outcome.candidates or candidates_for_domain(
                        snapshot, resolve_domain, limit=_DISCOVERY_LIMIT + 1
                    )
                    fix = _candidate_ids(candidates)
                    return _UnresolvedTarget(
                        requested=entity_id,
                        hint=_target_not_found_message(entity_id, resolve_domain, fix),
                        fix=fix,
                    )
        if "device_id" in target:
            supported_keys.append("device_id")
            for device_id in _target_values(target["device_id"]):
                supported_values.append(device_id)
                resolved = indexes.entity_ids_by_device_id.get(device_id, ())
                entity_ids.update(resolved)
                if resolved:
                    adjustments.append(_target_selector_expanded_adjustment("device_id", device_id, resolved))
        if "area_id" in target:
            supported_keys.append("area_id")
            for area_id in _target_values(target["area_id"]):
                supported_values.append(area_id)
                resolved = indexes.entity_ids_by_area_id.get(area_id, ())
                entity_ids.update(resolved)
                if resolved:
                    adjustments.append(_target_selector_expanded_adjustment("area_id", area_id, resolved))
        if "label_id" in target:
            supported_keys.append("label_id")
            for label_id in _target_values(target["label_id"]):
                supported_values.append(label_id)
                resolved = indexes.entity_ids_by_label.get(label_id, ())
                entity_ids.update(resolved)
                if resolved:
                    adjustments.append(_target_selector_expanded_adjustment("label_id", label_id, resolved))
        if "label" in target:
            supported_keys.append("label")
            for label_id in _target_values(target["label"]):
                supported_values.append(label_id)
                resolved = indexes.entity_ids_by_label.get(label_id, ())
                entity_ids.update(resolved)
                if resolved:
                    adjustments.append(_target_selector_expanded_adjustment("label", label_id, resolved))
        if "floor_id" in target:
            supported_keys.append("floor_id")
            for floor_id in _target_values(target["floor_id"]):
                supported_values.append(floor_id)
                floor_entity_ids: list[str] = []
                for area_id in indexes.area_ids_by_floor_id.get(floor_id, ()):
                    floor_entity_ids.extend(indexes.entity_ids_by_area_id.get(area_id, ()))
                entity_ids.update(floor_entity_ids)
                if floor_entity_ids:
                    adjustments.append(_target_selector_expanded_adjustment("floor_id", floor_id, floor_entity_ids))

        if entity_ids:
            return _ResolvedTarget({"entity_id": sorted(entity_ids)}, tuple(adjustments))
        if supported_values:
            fallback_fix = _candidate_ids(candidates_for_domain(snapshot, domain, limit=_DISCOVERY_LIMIT + 1))
            return _UnresolvedTarget(
                requested=supported_values[0],
                hint=_target_not_found_message(supported_values[0], domain, fallback_fix),
                fix=fallback_fix,
            )
        if supported_keys:
            return _UnresolvedTarget(
                requested=supported_keys[0],
                hint="No visible entities resolved for the requested target.",
            )
        return _ResolvedTarget(cast(dict[str, object], json_safe(target)))

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
        if (brief := self.services_schema.get(domain, {}).get(service)) is not None:
            hints = {"fields": _service_field_names(brief)}
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

    def _mapping(self) -> dict[str, object | None]:
        """Return the bounded context fields exposed to Monty."""
        return {
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
        }

    def get(self, key: str, default: object | None = None) -> object | None:
        """Return a context field by name, or ``default`` when absent."""
        return self._mapping().get(key, default)

    def keys(self) -> list[str]:
        """Return the available context field names as a concrete list."""
        return list(self._mapping())

    def items(self) -> list[tuple[str, object | None]]:
        """Return the available context fields as concrete key/value tuples."""
        return list(self._mapping().items())

    def __llm_sandbox_json__(self) -> JsonValueType:
        return cast(JsonValueType, self._mapping())


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
    registry facades, ``repairs``, ``persistent_notifications``,
    ``config_entries``, date/time facades, and ``now``. ``llm_context`` is
    added separately by the tool caller (it depends on the live request).
    """
    entity_registry = SafeEntityRegistry(entities=snapshot.entities, indexes=snapshot.indexes)
    device_registry = SafeDeviceRegistry(devices=snapshot.devices, indexes=snapshot.indexes)
    area_registry = SafeAreaRegistry(areas=snapshot.areas)
    floor_registry = SafeFloorRegistry(floors=snapshot.floors)
    label_registry = SafeLabelRegistry(labels=snapshot.labels)
    category_registry = SafeCategoryRegistry(categories=snapshot.categories)
    repairs = SafeIssueRegistry(issues=list(snapshot.issues))
    persistent_notifications = SafeNotificationRegistry(notifications=list(snapshot.notifications))
    config_entries = SafeConfigEntries(entries=list(snapshot.config_entries))

    state_machine = SafeStateMachine(states=snapshot.states)
    service_registry = SafeServiceRegistry(
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
        "er": entity_registry,
        "dr": device_registry,
        "ar": area_registry,
        "fr": floor_registry,
        "lr": label_registry,
        "cr": category_registry,
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


def _bounded_strings(values: list[str]) -> list[str]:
    """Bound deterministic repair lists to the discovery limit plus overflow marker."""
    if len(values) > _DISCOVERY_LIMIT:
        return [*values[: _DISCOVERY_LIMIT - 1], "..."]
    return values


def _candidate_ids(candidates: tuple[CandidateTarget, ...]) -> list[str]:
    """Return bounded candidate entity ids for model repair."""
    return _bounded_strings([candidate.entity_id for candidate in candidates])


def _valid_domains_message(domain: str, valid_domains: list[str]) -> str:
    """Return the compact domain-layer repair message."""
    if valid_domains:
        return f"Domain '{domain}' is not available. Valid domains: {', '.join(valid_domains)}."
    return f"Domain '{domain}' is not available."


def _valid_services_message(domain: str, service: str, valid_services: list[str]) -> str:
    """Return the compact service-layer repair message."""
    if valid_services:
        return f"No service '{service}' on '{domain}'. Valid services: {', '.join(valid_services)}."
    return f"No service '{service}' on '{domain}'."


def _target_not_found_message(requested: str, domain: str, fix: list[str]) -> str:
    """Return the compact target-layer repair message."""
    if fix:
        return f"Target '{requested}' not found in '{domain}'. Did you mean: {', '.join(fix)}."
    return f"Target '{requested}' not found in '{domain}'."


def _service_field_names(brief: ServiceSchemaBrief | None) -> list[str] | None:
    """Return bounded field names from a service schema brief."""
    if brief is None:
        return None
    fields = brief.get("fields", [])
    if not isinstance(fields, list):
        return None
    names = sorted(str(field["name"]) for field in fields if isinstance(field, dict) and "name" in field)
    return _bounded_strings(names) if names else None


def _resolved_from_adjustments(adjustments: list[dict[str, object]]) -> str | None:
    """Return the requested entity id when an entity-id rewrite was applied."""
    for adjustment in adjustments:
        if adjustment.get("key") != "target_entity_resolved":
            continue
        requested = adjustment.get("requested")
        if isinstance(requested, Mapping):
            entity_id = requested.get("entity_id")
            if isinstance(entity_id, str):
                return entity_id
    return None


def _extract_target_selectors(
    service_data: Mapping[str, object] | None,
    target: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, tuple[dict[str, object], ...]]:
    """Move HA target selector keys from service data into the target mapping."""
    raw_service_data = dict(service_data) if service_data is not None else {}
    extracted_target = {
        key: raw_service_data.pop(key) for key in tuple(raw_service_data) if key in _TARGET_SELECTOR_KEYS
    }
    raw_target = dict(target) if target is not None else {}

    # Explicit target values win over selector values supplied inside service data.
    merged_target = extracted_target | raw_target
    cleaned_service_data = cast(dict[str, object], json_safe(raw_service_data)) if raw_service_data else None
    applied_keys = tuple(key for key in extracted_target if key not in raw_target)
    adjustments = (_target_selector_moved_adjustment(applied_keys),) if applied_keys else ()
    return (
        cleaned_service_data,
        cast(dict[str, object], json_safe(merged_target)) if merged_target else None,
        adjustments,
    )


@dataclass(frozen=True, slots=True)
class _PolicyBlock:
    """A snapshot-policy gate that prevents a service call from executing."""

    key: str
    placeholders: TranslationPlaceholders
    message: str
    fix: list[str] | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    """A visibility-resolved service target (entity_id list or empty)."""

    target: dict[str, object] | None
    adjustments: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _UnresolvedTarget:
    """A target that could not resolve to visible entities; carries discovery hints."""

    requested: str
    hint: str
    fix: list[str] | None = None


def _action_error(
    key: str,
    message: str,
    *,
    fix: list[str] | None = None,
) -> dict[str, object]:
    """Build the JSON-safe action error shape."""
    clean = " ".join(message.split())
    error: dict[str, object] = {
        "key": key,
        "message": clean if clean and clean != key else f"Resolve '{key}' before retrying.",
    }
    if fix:
        error["fix"] = fix
    return error


def _action_record(
    action: ProposedAction,
    *,
    status: str,
    response: object,
    error: dict[str, object] | None,
    adjustments: list[dict[str, object]] | None = None,
) -> ActionRecord:
    """Build one mutable service action record."""
    record: ActionRecord = {
        "service": f"{action['domain']}.{action['service']}",
        "target": action["target"],
        "status": status,
    }
    if response is not None:
        record["response"] = response
    if error is not None:
        record["error"] = error
    if adjustments:
        if (resolved_from := _resolved_from_adjustments(adjustments)) is not None:
            record["resolved_from"] = resolved_from
        else:
            record["adjustments"] = adjustments
    return record


def _applied_adjustment(key: str, message: str, **extra: object) -> dict[str, object]:
    """Build a concise model-facing note for a rewrite already applied."""
    return {"key": key, "status": "applied", "retry_needed": False, "message": message, **extra}


def _target_selector_moved_adjustment(selectors: tuple[str, ...]) -> dict[str, object]:
    """Explain target selectors moved out of service_data."""
    selector_list = sorted(selectors)
    return _applied_adjustment(
        "target_selector_moved",
        "Moved target selector(s) from service_data into target before execution; no retry needed.",
        selectors=selector_list,
    )


def _target_entity_resolved_adjustment(requested_entity_id: str, resolved_entity_id: str) -> dict[str, object]:
    """Explain fuzzy entity-id resolution for one requested target."""
    return _applied_adjustment(
        "target_entity_resolved",
        (
            f"Resolved requested target entity_id {requested_entity_id} to visible entity {resolved_entity_id} "
            "before execution; no retry needed."
        ),
        requested={"entity_id": requested_entity_id},
        applied={"entity_id": [resolved_entity_id]},
    )


def _target_selector_expanded_adjustment(
    selector: str,
    requested: str,
    resolved_entity_ids: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Explain selector expansion to concrete visible entity IDs."""
    entity_ids = sorted(set(resolved_entity_ids))
    return _applied_adjustment(
        "target_selector_expanded",
        f"Expanded target {selector} {requested} to visible entity target(s) before execution; no retry needed.",
        selector=selector,
        requested={selector: requested},
        applied={"entity_id": entity_ids},
    )
