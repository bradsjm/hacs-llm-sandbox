"""LLM context facade and facade builders."""

# ruff: noqa: D105

from dataclasses import dataclass
from datetime import datetime as _datetime
from typing import cast
from zoneinfo import ZoneInfo

from homeassistant.util.json import JsonValueType

from ...snapshot.models import HomeSnapshot, SafeContext
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
from .services import SafeServiceRegistry, service_discovery_facts
from .state import (
    SafeDateFacade,
    SafeDateTimeFacade,
    SafeHass,
    SafeStateMachine,
    _date_from_datetime,
    _datetime_from_dt,
)


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
        _discovery=service_discovery_facts(snapshot),
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
