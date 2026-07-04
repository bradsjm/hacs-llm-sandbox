"""Snapshot package: frozen Home Assistant state and registry records."""

from .builder import build_recorder_snapshot, build_snapshot, build_vision_snapshot, finalize_snapshot
from .models import (
    DEFAULT_SCOPE,
    HomeSnapshot,
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfigEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SafeState,
    SnapshotIndexes,
    SnapshotScope,
)

__all__ = [
    "DEFAULT_SCOPE",
    "HomeSnapshot",
    "SafeAreaEntry",
    "SafeCategoryEntry",
    "SafeConfigEntry",
    "SafeContext",
    "SafeDeviceEntry",
    "SafeFloorEntry",
    "SafeIssueEntry",
    "SafeLabelEntry",
    "SafeNotificationEntry",
    "SafeRegistryEntry",
    "SafeState",
    "SnapshotIndexes",
    "SnapshotScope",
    "build_recorder_snapshot",
    "build_snapshot",
    "build_vision_snapshot",
    "finalize_snapshot",
]
