"""Direct, Home Assistant-authorized automation reads."""

import asyncio
import base64
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import functools
import json
import time
from typing import cast, final, override

from homeassistant.auth.permissions.const import POLICY_READ
from homeassistant.components import automation
from homeassistant.components.automation.logbook import EVENT_AUTOMATION_TRIGGERED
from homeassistant.components.logbook import DOMAIN as LOGBOOK_DOMAIN
from homeassistant.components.logbook.processor import EventProcessor
from homeassistant.components.recorder import get_instance
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import category_registry as cr
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import llm, selector
from homeassistant.helpers.recorder import DATA_INSTANCE
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
import voluptuous as vol

from ...const import (
    DEFAULT_LOGBOOK_WINDOW_HOURS,
    MAX_RECORDER_ENTITY_IDS,
    MAX_RECORDER_LOOKBACK_HOURS,
    MAX_RECORDER_PAGE_BYTES,
    TOOL_GET_AUTOMATION,
)
from ..data.recorder_scope import _clamp_window
from ..errors import RecoverableToolError, tool_error_envelope, tool_error_from_exception
from ..executor_support import json_safe
from ..prompts import build_get_automation_description
from ._recorder_runtime import _sync_recorder_for_query
from ._support import (
    _bounded_list,
    _omit_empty_optional_args,
    _require_loaded_entry_error,
    _require_sandbox_runtime,
)

_NULL_KEYS = frozenset({"query", "entity_ids", "include", "hours", "start", "end", "limit", "cursor"})
_EMPTY_STRINGS = frozenset({"query", "start", "end", "cursor"})
_EMPTY_LISTS = frozenset({"entity_ids", "include"})
_INCLUDE = ("content", "runs")

type AutomationRunsFetcher = Callable[
    [list[str], datetime, datetime], Awaitable[Mapping[str, list[dict[str, object]]]]
]


@dataclass(frozen=True, slots=True)
class AutomationRecord:
    """Copied, hass-free automation data used by the query core."""

    entity_id: str
    summary: Mapping[str, object]
    search_terms: tuple[str, ...]
    content: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class AutomationSource:
    """The host-owned, copied data and fetch seam for one automation query."""

    now: datetime
    available: bool
    content_authorized: bool
    records: tuple[AutomationRecord, ...]
    fetch_runs: AutomationRunsFetcher


def _iso_datetime(value: object) -> datetime:
    """Validate an ISO datetime and normalize it to UTC."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str) and (parsed := dt_util.parse_datetime(value)) is not None:
        return dt_util.as_utc(parsed)
    raise vol.Invalid("expected an ISO datetime")


def _query_value(value: object) -> str:
    """Validate and canonicalize a text query."""
    if not isinstance(value, str):
        raise vol.Invalid("query must be a string")
    value = " ".join(value.split())
    if not value:
        raise vol.Invalid("query must not be empty")
    if len(value) > 256:
        raise vol.Invalid("query must not exceed 256 characters")
    return value


def _automation_entity_id(value: object) -> str:
    """Validate one automation entity ID."""
    entity_id = cv.entity_id(value)
    if not entity_id.startswith("automation."):
        raise vol.Invalid("entity_ids must contain automation entities")
    return entity_id


def _automation_entity_ids(value: object) -> list[str]:
    """Validate a whole list of automation entity IDs."""
    if not isinstance(value, list):
        raise vol.Invalid("entity_ids must be a list")
    return [_automation_entity_id(item) for item in value]


def _canonical_include(value: object) -> list[str]:
    """Validate and canonicalize requested projections."""
    if not isinstance(value, list):
        raise vol.Invalid("include must be a list")
    if any(item not in _INCLUDE for item in value):
        raise vol.Invalid("include contains an unsupported projection")
    return [item for item in _INCLUDE if item in value]


def _compact_json(payload: Mapping[str, object]) -> bytes:
    """Encode a response using the established compact UTF-8 policy."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _encode_cursor(data: Mapping[str, object]) -> str:
    """Encode a compact opaque cursor."""
    return base64.urlsafe_b64encode(_compact_json(data)).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> dict[str, object]:
    """Decode and structurally validate an automation cursor."""
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        result = json.loads(decoded)
    except ValueError, UnicodeDecodeError, json.JSONDecodeError:
        raise RecoverableToolError("invalid_cursor", {}) from None
    if not isinstance(result, dict) or result.get("v") != 1 or result.get("k") != "automation":
        raise RecoverableToolError("invalid_cursor", {})
    if (
        not isinstance(result.get("q"), str)
        or len(result["q"]) > 256
        or " ".join(result["q"].split()) != result["q"]
        or not isinstance(result.get("e"), list)
        or len(result["e"]) > MAX_RECORDER_ENTITY_IDS
    ):
        raise RecoverableToolError("invalid_cursor", {})
    projections = result.get("p")
    if not isinstance(projections, list) or projections != [item for item in _INCLUDE if item in projections]:
        raise RecoverableToolError("invalid_cursor", {})
    if len(set(projections)) != len(projections) or any(item not in _INCLUDE for item in projections):
        raise RecoverableToolError("invalid_cursor", {})
    entity_ids = result["e"]
    if not isinstance(result.get("after"), str):
        raise RecoverableToolError("invalid_cursor", {})
    try:
        validated_entity_ids = [_automation_entity_id(item) for item in entity_ids]
        validated_after = _automation_entity_id(result["after"])
    except TypeError, vol.Invalid:
        raise RecoverableToolError("invalid_cursor", {}) from None
    if validated_entity_ids != entity_ids or validated_after != result["after"]:
        raise RecoverableToolError("invalid_cursor", {})
    if entity_ids != sorted(set(entity_ids)):
        raise RecoverableToolError("invalid_cursor", {})
    limit = result.get("l")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_RECORDER_ENTITY_IDS:
        raise RecoverableToolError("invalid_cursor", {})
    window = result.get("w")
    if "runs" in projections:
        if (
            not isinstance(window, dict)
            or not isinstance(window.get("s"), str)
            or not isinstance(window.get("e"), str)
        ):
            raise RecoverableToolError("invalid_cursor", {})
        start, end = dt_util.parse_datetime(window["s"]), dt_util.parse_datetime(window["e"])
        try:
            invalid_window = (
                start is None
                or end is None
                or start > end
                or end - start > timedelta(hours=MAX_RECORDER_LOOKBACK_HOURS)
            )
        except TypeError:
            invalid_window = True
        if invalid_window:
            raise RecoverableToolError("invalid_cursor", {})
    elif window is not None:
        raise RecoverableToolError("invalid_cursor", {})
    return result


def _fit_automation_page(
    records: list[dict[str, object]], limit: int, cursor_data: dict[str, object], window: dict[str, str] | None
) -> tuple[list[dict[str, object]], str | None]:
    """Fit whole automation records to the compact response budget."""
    selected: list[dict[str, object]] = []
    for record in records[:limit]:
        candidate = [*selected, record]
        payload: dict[str, object] = {"automations": candidate, "returned": len(candidate), "limit": limit}
        if window is not None:
            payload["window"] = window
        if len(candidate) < len(records):
            payload["next_cursor"] = _encode_cursor({**cursor_data, "after": record["entity_id"]})
        if len(_compact_json(payload)) > MAX_RECORDER_PAGE_BYTES and selected:
            break
        selected = candidate
    if not selected or selected[-1]["entity_id"] == records[-1]["entity_id"]:
        return selected, None
    return selected, _encode_cursor({**cursor_data, "after": selected[-1]["entity_id"]})


@final
class GetAutomationTool(llm.Tool):
    """Return Home Assistant-authorized automation records."""

    name = TOOL_GET_AUTOMATION
    description = build_get_automation_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Optional("query"): vol.All(selector.TextSelector(), _query_value),
            vol.Optional("entity_ids"): vol.All(
                cv.ensure_list,
                selector.EntitySelector(selector.EntitySelectorConfig(domain="automation", multiple=True)),
                _automation_entity_ids,
                _bounded_list(
                    "entity_ids",
                    min_items=1,
                    max_items=MAX_RECORDER_ENTITY_IDS,
                ),
            ),
            vol.Optional("include"): vol.All(
                selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=list(_INCLUDE),
                        multiple=True,
                    )
                ),
                _canonical_include,
                _bounded_list("include", min_items=1, max_items=2),
            ),
            vol.Optional("hours"): vol.All(vol.Coerce(float), vol.Range(min=0)),
            vol.Optional("start"): vol.All(selector.DateTimeSelector(), _iso_datetime),
            vol.Optional("end"): vol.All(selector.DateTimeSelector(), _iso_datetime),
            vol.Optional("limit", default=10): vol.All(vol.Coerce(int), vol.Range(min=1, max=MAX_RECORDER_ENTITY_IDS)),
            vol.Optional("cursor"): str,
        }
    )

    def __init__(self, entry_id: str) -> None:
        """Initialize the tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        """Validate, authorize, and return automation records."""
        try:
            normalized_args = self._normalize_args(tool_input.tool_args)
            if "cursor" in normalized_args and len(normalized_args) != 1:
                raise vol.Invalid("cursor must be the only non-empty argument")
            data = cast(dict[str, object], self.parameters(normalized_args))
            self._validate_query_data(data)
        except Exception as err:
            mapped = (
                (err.key, err.placeholders)
                if isinstance(err, RecoverableToolError)
                else tool_error_from_exception(err)
            )
            if mapped is None:
                raise
            return tool_error_envelope(*mapped)

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            return tool_error_envelope(*setup_error)
        cursor = _decode_cursor(cast(str, data["cursor"])) if "cursor" in data else None
        include = tuple(cast(list[str], cursor["p"] if cursor else data.get("include", [])))
        user_id = llm_context.context.user_id if llm_context.context is not None else None
        user = await hass.auth.async_get_user(user_id) if user_id else None
        if user is None or not user.is_active or ("content" in include and not user.is_admin):
            return tool_error_envelope("authorization_denied", {})
        component = hass.data.get(automation.DATA_COMPONENT)
        if component is None:
            return tool_error_envelope("automation_unavailable", {})
        entities = [
            entity for entity in component.entities if user.permissions.check_entity(entity.entity_id, POLICY_READ)
        ]
        cursor = _decode_cursor(cast(str, data["cursor"])) if "cursor" in data else None
        include = tuple(cast(list[str], cursor["p"] if cursor else data.get("include", [])))
        if cursor:
            explicit_ids = cast(list[str], cursor["e"])
        else:
            explicit_ids = sorted(set(cast(list[str], data.get("entity_ids", []))))
        if explicit_ids:
            entities = [entity for entity in entities if entity.entity_id in explicit_ids]
        if "runs" in include:
            if DATA_INSTANCE not in hass.data:
                return tool_error_envelope("recorder_unavailable", {})
            if LOGBOOK_DOMAIN not in hass.data:
                return tool_error_envelope("logbook_unavailable", {})
        try:
            candidates = tuple(self._record(entity, hass, user.is_admin, "content" in include) for entity in entities)
        except RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)

        settings = _require_sandbox_runtime(hass, self.entry_id).settings

        async def fetch_runs(
            entity_ids: list[str], start: datetime, end: datetime
        ) -> Mapping[str, list[dict[str, object]]]:
            run_records: list[dict[str, object]] = [{"entity_id": entity_id} for entity_id in entity_ids]
            await self._add_runs(hass, run_records, start, end, settings.execution_timeout_seconds)
            return {
                entity_id: cast(list[dict[str, object]], record.get("runs", []))
                for entity_id, record in zip(entity_ids, run_records, strict=True)
            }

        source = AutomationSource(
            now=dt_util.utcnow(),
            available=True,
            content_authorized=True,
            records=candidates,
            fetch_runs=fetch_runs,
        )
        return await self.run_query(data, source)

    @staticmethod
    def _validate_query_data(data: Mapping[str, object]) -> None:
        """Validate cross-field query rules shared by live and eval callers."""
        cursor = _decode_cursor(cast(str, data["cursor"])) if "cursor" in data else None
        include = tuple(cast(list[str], cursor["p"] if cursor else data.get("include", [])))
        if ("hours" in data or "start" in data or "end" in data) and "runs" not in include:
            raise vol.Invalid("hours, start, and end require include=runs")

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]:
        """Normalize optional values before schema validation."""
        return _omit_empty_optional_args(
            args, null_keys=_NULL_KEYS, empty_string_keys=_EMPTY_STRINGS, empty_list_keys=_EMPTY_LISTS
        )

    async def run_query(self, data: dict[str, object], source: AutomationSource) -> JsonObjectType:
        """Run the hass-free automation query and preserve stable envelopes."""
        try:
            return await self._query(data, source)
        except RecoverableToolError as err:
            return tool_error_envelope(err.key, err.placeholders)
        except Exception as err:  # noqa: BLE001 - direct query failures use the stable query envelope
            mapped = tool_error_from_exception(err)
            return tool_error_envelope(*(mapped or ("query_failed", {"error": type(err).__name__})))

    async def _query(self, data: dict[str, object], source: AutomationSource) -> JsonObjectType:
        """Apply search, projection, runs, and pagination to copied source records."""
        cursor = _decode_cursor(cast(str, data["cursor"])) if "cursor" in data else None
        include = tuple(cast(list[str], cursor["p"] if cursor else data.get("include", [])))
        if not source.available:
            raise RecoverableToolError("automation_unavailable", {})
        if "content" in include and not source.content_authorized:
            raise RecoverableToolError("authorization_denied", {})
        query = cast(str, cursor["q"] if cursor else data.get("query", ""))
        explicit_ids = cast(
            list[str], cursor["e"] if cursor else sorted(set(cast(list[str], data.get("entity_ids", []))))
        )
        limit = cast(int, cursor["l"] if cursor else data["limit"])
        after = cast(str, cursor["after"] if cursor else "")
        records = [record for record in source.records if not explicit_ids or record.entity_id in explicit_ids]
        records = [
            record
            for record in records
            if not query
            or all(token.casefold() in " ".join(record.search_terms).casefold() for token in query.split())
        ]
        records = sorted((record for record in records if record.entity_id > after), key=lambda item: item.entity_id)
        window: dict[str, str] | None = None
        runs: Mapping[str, list[dict[str, object]]] = {}
        if "runs" in include:
            if cursor:
                raw_window = cast(dict[str, str], cursor["w"])
                start, end = _iso_datetime(raw_window["s"]), _iso_datetime(raw_window["e"])
            else:
                start, end = _clamp_window(
                    source.now,
                    cast(datetime | None, data.get("start")),
                    cast(datetime | None, data.get("end")),
                    hours=cast(float | None, data.get("hours")),
                    default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
                    max_hours=MAX_RECORDER_LOOKBACK_HOURS,
                )
            window = {"start": start.isoformat(), "end": end.isoformat()}
            runs = await source.fetch_runs([record.entity_id for record in records[:limit]], start, end)
        output: list[dict[str, object]] = []
        for record in records:
            value = dict(record.summary)
            if "content" in include:
                if record.content is None:
                    raise RecoverableToolError("automation_content_unavailable", {})
                value["content"] = json_safe(dict(record.content))
            if "runs" in include:
                value["runs"] = json_safe(
                    sorted(runs.get(record.entity_id, []), key=lambda entry: str(entry.get("when", "")), reverse=True)
                )
            output.append(cast(dict[str, object], json_safe(value)))
        cursor_data: dict[str, object] = {
            "v": 1,
            "k": "automation",
            "q": query,
            "e": explicit_ids,
            "p": list(include),
            "l": limit,
        }
        if window is not None:
            cursor_data["w"] = {"s": window["start"], "e": window["end"]}
        selected, next_cursor = _fit_automation_page(output, limit, cursor_data, window)
        payload: dict[str, object] = {"automations": selected, "returned": len(selected), "limit": limit}
        if window is not None:
            payload["window"] = window
        if next_cursor:
            payload["next_cursor"] = next_cursor
        return cast(JsonObjectType, json_safe(payload))

    @staticmethod
    def _record(  # noqa: C901 - one flat record projection keeps the direct-tool boundary local
        entity: automation.BaseAutomationEntity,
        hass: HomeAssistant,
        is_admin: bool,
        include_content: bool,
    ) -> AutomationRecord:
        """Build one copied summary and apply normalized metadata search."""
        automation_entity = entity
        entity_id = automation_entity.entity_id
        state = hass.states.get(entity_id)
        registry_entry = er.async_get(hass).async_get(entity_id)
        raw_config = getattr(automation_entity, "raw_config", None)
        description = raw_config.get("description") if is_admin and isinstance(raw_config, Mapping) else None
        title = state.name if state is not None else entity_id
        search_terms = [title, entity_id]
        record: dict[str, object] = {"entity_id": entity_id, "title": title}
        if state is not None:
            record.update(
                {"state": state.state, "is_on": automation_entity.is_on, "available": state.state != "unavailable"}
            )
            for key in ("last_triggered", "mode", "current"):
                if key in state.attributes:
                    record[key] = state.attributes[key]
        if automation_entity.unique_id is not None:
            record["config_id"] = automation_entity.unique_id
            search_terms.append(automation_entity.unique_id)
        if description is not None:
            record["description"] = description
            search_terms.append(str(description))
        if registry_entry is not None:
            area_registry, category_registry, label_registry = (
                ar.async_get(hass),
                cr.async_get(hass),
                lr.async_get(hass),
            )
            if registry_entry.area_id and (area := area_registry.async_get_area(registry_entry.area_id)):
                record["area"] = {"id": area.id, "name": area.name}
                search_terms.extend((area.id, area.name))
            search_terms.extend(alias for alias in registry_entry.aliases if isinstance(alias, str))
            record["labels"] = [
                {
                    "id": label_id,
                    **(
                        {"name": label.name} if (label := label_registry.async_get_label(label_id)) is not None else {}
                    ),
                }
                for label_id in sorted(registry_entry.labels)
            ]
            for label_id in sorted(registry_entry.labels):
                search_terms.append(label_id)
                if label := label_registry.async_get_label(label_id):
                    search_terms.append(label.name)
            record["categories"] = [
                {
                    "scope": scope,
                    "id": category_id,
                    **(
                        {"name": category.name}
                        if (category := category_registry.async_get_category(scope=scope, category_id=category_id))
                        else {}
                    ),
                }
                for scope, category_id in sorted(registry_entry.categories.items())
            ]
            for scope, category_id in sorted(registry_entry.categories.items()):
                search_terms.extend((scope, category_id))
                if category := category_registry.async_get_category(scope=scope, category_id=category_id):
                    search_terms.append(category.name)
        references: dict[str, list[dict[str, object]]] = {}
        for ref_name, ref_ids in (
            ("entities", automation_entity.referenced_entities),
            ("devices", automation_entity.referenced_devices),
            ("areas", automation_entity.referenced_areas),
            ("floors", automation_entity.referenced_floors),
            ("labels", automation_entity.referenced_labels),
        ):
            values: list[dict[str, object]] = []
            for ref_id in sorted(ref_ids):
                ref: object | None
                if ref_name == "entities":
                    ref = er.async_get(hass).async_get(ref_id)
                    ref_state = hass.states.get(ref_id)
                    ref_name_value = (
                        ref_state.name
                        if ref_state is not None
                        else (ref.name or ref.original_name)
                        if ref is not None
                        else None
                    )
                elif ref_name == "devices":
                    ref = dr.async_get(hass).async_get(ref_id)
                    ref_name_value = (ref.name_by_user or ref.name) if ref is not None else None
                elif ref_name == "areas":
                    ref = ar.async_get(hass).async_get_area(ref_id)
                    ref_name_value = getattr(ref, "name", None) if ref is not None else None
                elif ref_name == "floors":
                    ref = fr.async_get(hass).async_get_floor(ref_id)
                    ref_name_value = ref.name if ref is not None else None
                else:
                    ref = lr.async_get(hass).async_get_label(ref_id)
                    ref_name_value = ref.name if ref is not None else None
                values.append({"id": ref_id, **({"name": ref_name_value} if ref_name_value else {})})
                search_terms.append(ref_id)
                if ref_name_value:
                    search_terms.append(ref_name_value)
            if values:
                references[ref_name] = values
        if references:
            record["references"] = references
        return AutomationRecord(
            entity_id=entity_id,
            summary=cast(dict[str, object], json_safe(record)),
            search_terms=tuple(str(term) for term in search_terms),
            content=cast(dict[str, object] | None, json_safe(dict(raw_config)))
            if include_content and is_admin and isinstance(raw_config, Mapping)
            else None,
        )

    @staticmethod
    async def _add_runs(
        hass: HomeAssistant, records: list[dict[str, object]], start: datetime, end: datetime, budget_seconds: float
    ) -> None:
        """Fetch only automation-triggered Logbook entries and group them by entity."""
        entity_ids = [cast(str, record["entity_id"]) for record in records]
        if not entity_ids:
            return
        deadline = time.monotonic() + budget_seconds
        await _sync_recorder_for_query(hass, get_instance(hass), deadline)
        processor = EventProcessor(
            hass,
            {EVENT_AUTOMATION_TRIGGERED},
            entity_ids,
            None,
            None,
            timestamp=False,
            include_entity_name=True,
        )
        entries = await asyncio.wait_for(
            hass.async_add_executor_job(functools.partial(processor.get_events, start_day=start, end_day=end)),
            max(0, deadline - time.monotonic()),
        )
        grouped: dict[str, list[dict[str, object]]] = {entity_id: [] for entity_id in entity_ids}
        for entry in entries:
            entity_id = entry.get("entity_id")
            if entity_id in grouped:
                grouped[entity_id].append(dict(entry))
        for record in records:
            runs = grouped[cast(str, record["entity_id"])]
            runs.sort(key=lambda entry: str(entry.get("when", "")), reverse=True)
            record["runs"] = cast(list[object], json_safe(runs))
