"""Snapshot package: frozen Home Assistant state and registry records."""

from .builder import build_snapshot
from .models import (
    DEFAULT_SCOPE,
    HomeSnapshot,
    SafeAreaEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeRegistryEntry,
    SafeState,
    SnapshotIndexes,
    SnapshotScope,
)

__all__ = [
    "DEFAULT_SCOPE",
    "HomeSnapshot",
    "SafeAreaEntry",
    "SafeContext",
    "SafeDeviceEntry",
    "SafeFloorEntry",
    "SafeRegistryEntry",
    "SafeState",
    "SnapshotIndexes",
    "SnapshotScope",
    "build_snapshot",
]
