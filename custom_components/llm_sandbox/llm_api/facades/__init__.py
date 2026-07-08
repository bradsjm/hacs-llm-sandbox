"""Narrow public surface for Monty-facing Home Assistant facades."""

from .context import SafeLLMContext, build_facades, build_llm_context
from .registries import (
    SafeAreaRegistry,
    SafeCategoryRegistry,
    SafeConfigEntries,
    SafeDeviceRegistry,
    SafeEntityRegistry,
    SafeFloorRegistry,
    SafeIssueRegistry,
    SafeLabelRegistry,
    SafeNotificationRegistry,
)
from .services import SafeServiceRegistry
from .state import SafeDate, SafeDateFacade, SafeDateTime, SafeDateTimeFacade, SafeHass, SafeStateMachine

__all__ = (
    "SafeAreaRegistry",
    "SafeCategoryRegistry",
    "SafeConfigEntries",
    "SafeDate",
    "SafeDateFacade",
    "SafeDateTime",
    "SafeDateTimeFacade",
    "SafeDeviceRegistry",
    "SafeEntityRegistry",
    "SafeFloorRegistry",
    "SafeHass",
    "SafeIssueRegistry",
    "SafeLLMContext",
    "SafeLabelRegistry",
    "SafeNotificationRegistry",
    "SafeServiceRegistry",
    "SafeStateMachine",
    "build_facades",
    "build_llm_context",
)
