"""Registry facades for frozen Home Assistant snapshot records."""

# ruff: noqa: ANN401

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...snapshot.models import (
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfigEntry,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SnapshotIndexes,
)


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
