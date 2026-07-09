"""Snapshot Home Assistant service catalog metadata."""

from collections.abc import Mapping

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service import async_get_cached_service_description

from .models import (
    ServiceFieldBrief,
    ServiceFieldFilter,
    ServiceSchemaBrief,
    ServiceTargetBrief,
    ServiceTargetFilter,
)

SERVICE_SCHEMA_FIELD_LIMIT = 12


def _safe_services(
    hass: HomeAssistant,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, dict[str, str]],
    dict[str, dict[str, ServiceSchemaBrief]],
    dict[str, dict[str, ServiceTargetBrief]],
]:
    """Snapshot the service catalog as domain -> service names.

    Returns ``(services, services_supports_response, services_schema, services_target)``
    where response support preserves each service's enum value as a JSON-safe string,
    service schemas are eager JSON-safe parameter briefs, and service targets carry
    the entity-filter metadata HA's automation matching uses
    (``websocket_api/automation.py``). Services without a declared target are absent
    from ``services_target``; the matcher treats absence as "not entity-targeting".
    """
    catalog = hass.services.async_services()
    services: dict[str, tuple[str, ...]] = {}
    supports_response: dict[str, dict[str, str]] = {}
    services_schema: dict[str, dict[str, ServiceSchemaBrief]] = {}
    services_target: dict[str, dict[str, ServiceTargetBrief]] = {}
    for domain, domain_services in catalog.items():
        names: list[str] = []
        response_values: dict[str, str] = {}
        schema_values: dict[str, ServiceSchemaBrief] = {}
        target_values: dict[str, ServiceTargetBrief] = {}
        for service_name, service in domain_services.items():
            description = async_get_cached_service_description(hass, domain, service_name)
            names.append(service_name)
            response_values[service_name] = service.supports_response.value
            schema_values[service_name] = _service_schema_brief(description, service.schema)
            target_brief = _service_target_brief(description)
            if target_brief is not None:
                target_values[service_name] = target_brief
        services[domain] = tuple(sorted(names))
        supports_response[domain] = {name: response_values[name] for name in sorted(response_values)}
        services_schema[domain] = {name: schema_values[name] for name in sorted(schema_values)}
        if target_values:
            services_target[domain] = {name: target_values[name] for name in sorted(target_values)}
    return services, supports_response, services_schema, services_target


def _service_schema_brief(
    description: object,
    schema: object,
) -> ServiceSchemaBrief:
    """Build a JSON-safe brief for one service schema."""
    if schema is None:
        return {"fields": (), "dynamic": False}

    raw_schema = (
        schema.schema
        if isinstance(schema, vol.Schema)
        else vol.Schema(schema).schema
        if isinstance(schema, dict)
        else None
    )
    if not isinstance(raw_schema, dict):
        return {"fields": (), "dynamic": True}

    description_fields = _service_description_fields(description)
    fields: list[ServiceFieldBrief] = []
    dynamic = False
    for key, validator in raw_schema.items():
        name, required = _service_field_name_and_required(key)
        if name is None:
            dynamic = True
            continue

        field_description = description_fields.get(name, {})
        # Selector metadata and non-primitive validators often encode dynamic
        # value spaces; keep the brief coarse and mark the schema as dynamic.
        dynamic = dynamic or "selector" in field_description or not _is_plain_service_validator(validator)
        description = field_description.get("description")
        field_brief: ServiceFieldBrief = {
            "name": name,
            "required": required,
            "type_hint": _service_type_hint(validator),
            "description": description if isinstance(description, str) else None,
        }
        field_filter = _service_field_filter(field_description)
        if field_filter is not None:
            field_brief["filter"] = field_filter
        fields.append(field_brief)

    fields = sorted(fields, key=lambda field: str(field["name"]))
    if len(fields) > SERVICE_SCHEMA_FIELD_LIMIT:
        dynamic = True
        fields = fields[:SERVICE_SCHEMA_FIELD_LIMIT]
    return {"fields": tuple(fields), "dynamic": dynamic}


def _service_description_fields(description: object) -> dict[str, dict[str, object]]:
    """Return cached services.yaml fields for a service, if HA has them."""
    if not isinstance(description, dict):
        return {}
    fields = description.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {str(name): field for name, field in fields.items() if isinstance(field, dict)}


def _service_field_filter(field_description: Mapping[str, object]) -> ServiceFieldFilter | None:
    """Extract a JSON-safe capability filter for one service field, if declared.

    HA validates ``services.yaml`` field ``filter`` blocks via ``_FIELD_SCHEMA``,
    resolving ``supported_features`` / attribute-option names to their enum values.
    A field without a ``filter`` (or with an empty one) is universally usable and
    returns ``None`` so the matcher treats it as unconditional.
    """
    raw_filter = field_description.get("filter")
    if not isinstance(raw_filter, Mapping):
        return None
    field_filter: ServiceFieldFilter = {}
    raw_features = raw_filter.get("supported_features")
    if isinstance(raw_features, list):
        features = tuple(value for value in raw_features if isinstance(value, int))
        if features:
            field_filter["supported_features"] = features
    raw_attribute = raw_filter.get("attribute")
    if isinstance(raw_attribute, Mapping):
        attribute: dict[str, tuple[int | str, ...] | list[int | str]] = {}
        for attr_name, allowed in raw_attribute.items():
            if not isinstance(allowed, list):
                continue
            values = tuple(value for value in allowed if isinstance(value, int | str))
            if values:
                attribute[str(attr_name)] = values
        if attribute:
            field_filter["attribute"] = attribute
    return field_filter or None


def _service_target_brief(
    description: object,
) -> ServiceTargetBrief | None:
    """Build a JSON-safe target brief mirroring HA's automation matching input.

    Captures the service's declared ``entity`` target filters (each may constrain
    ``domain``, ``device_class``, ``integration``, ``supported_features``) from the
    cached service description. A service with a ``target`` but no entity filters
    is captured as ``{"entity": []}`` (matches any entity); a service without any
    ``target`` returns ``None`` (not entity-targeting, excluded from discovery).
    """
    if not isinstance(description, dict):
        return None
    target = description.get("target")
    if not isinstance(target, Mapping):
        return None
    raw_entity_filters = target.get("entity")
    if raw_entity_filters is None:
        # Target present but entity filters absent: HA treats this as matching
        # any entity, so capture an empty filter list to preserve that signal.
        raw_entity_filters = []
    if not isinstance(raw_entity_filters, list):
        return None
    filters: list[ServiceTargetFilter] = []
    for raw_filter in raw_entity_filters:
        if not isinstance(raw_filter, Mapping):
            continue
        filters.append(
            {
                "domain": tuple(str(value) for value in raw_filter.get("domain", []) if isinstance(value, str)),
                "device_class": tuple(
                    str(value) for value in raw_filter.get("device_class", []) if isinstance(value, str)
                ),
                "integration": value if isinstance(value := raw_filter.get("integration"), str) else None,
                "supported_features": tuple(
                    value for value in raw_filter.get("supported_features", []) if isinstance(value, int)
                ),
            }
        )
    return {"entity": tuple(filters)}


def _service_field_name_and_required(key: object) -> tuple[str | None, bool]:
    """Extract a service field name and required flag from a voluptuous key."""
    if isinstance(key, vol.Required):
        name = key.schema
        required = True
    elif isinstance(key, vol.Optional):
        name = key.schema
        required = False
    else:
        # Voluptuous treats bare (non-Marker) schema keys as optional by
        # default; only ``vol.Required`` (or schema-level ``required=True``)
        # makes a field required. Mark plain keys optional to match the
        # validation behavior Home Assistant services actually enforce.
        name = key
        required = False
    return (name, required) if isinstance(name, str) else (None, required)


def _service_type_hint(validator: object, depth: int = 0) -> str | None:
    """Derive a coarse JSON-safe type hint from a voluptuous validator."""
    if depth > 4:
        return None
    primitive_hint = _primitive_type_hint(validator)
    if primitive_hint is not None:
        return primitive_hint

    coerced_type = getattr(validator, "type", None)
    primitive_hint = _primitive_type_hint(coerced_type)
    if primitive_hint is not None:
        return primitive_hint

    child_validators = getattr(validator, "validators", None)
    if isinstance(child_validators, tuple):
        hints = {_service_type_hint(child, depth + 1) for child in child_validators}
        hints.discard(None)
        if hints == {"integer", "number"}:
            return "number"
        if len(hints) == 1:
            return hints.pop()
    return None


def _primitive_type_hint(validator: object) -> str | None:
    """Map simple Python validators and container schemas to coarse types."""
    if validator is str:
        return "string"
    if validator is bool:
        return "boolean"
    if validator is int:
        return "integer"
    if validator is float:
        return "number"
    if isinstance(validator, list | tuple):
        return "array"
    if isinstance(validator, dict):
        return "object"
    return None


def _is_plain_service_validator(validator: object) -> bool:
    """Whether a validator is simple enough that values are not dynamic."""
    if _primitive_type_hint(validator) is not None:
        return True
    coerced_type = getattr(validator, "type", None)
    return _primitive_type_hint(coerced_type) is not None
