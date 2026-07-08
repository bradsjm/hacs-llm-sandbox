"""Service catalog and gated live service-call facade."""

# ruff: noqa: D105, ANN401

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from homeassistant.core import SupportsResponse
from homeassistant.util.json import JsonValueType

from ...runtime import SandboxSettings
from ...snapshot.models import HomeSnapshot, ServiceSchemaBrief
from ...types import ActionRecord, ProposedAction, TranslationPlaceholders
from ..data.selectors import AGGREGATE_SELECTOR_KEYS, expand_aggregate_selectors
from ..errors import HelperExecutionError
from ..executor_support import helper_response, json_safe
from ..guidance import FailureContext, Intent, advise
from ..resolution import _DISCOVERY_LIMIT, bounded_strings, resolve_target_entity
from ..sandbox_context import require_runtime, require_snapshot
from ..target_matching import raw_service_field_names, service_accepts_domain, services_for_entity

_TARGET_SELECTOR_KEYS = frozenset(("entity_id", "device_id", "area_id", "label_id", "label", "floor_id"))


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

    def async_services_for_target(
        self, target: Mapping[str, object] | None
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Return compact, snapshot-derived service metadata for a target.

        Resolves ``entity_id``/``device_id``/``area_id``/``label_id``/``floor_id``
        selectors against the snapshot and reports, per resolved entity, the
        services whose declared target accepts that entity (mirroring HA's
        ``async_get_services_for_target``). Computed on demand from the frozen
        snapshot; no live Home Assistant call. Output is bounded to
        ``_DISCOVERY_LIMIT`` entities; each service lists its field names.
        """
        snapshot = require_snapshot()
        entity_ids = sorted(_expand_target_entities(snapshot, target))
        if not entity_ids:
            return {}
        result: dict[str, dict[str, dict[str, object]]] = {}
        for entity_id in entity_ids[:_DISCOVERY_LIMIT]:
            matched = services_for_entity(snapshot, entity_id)
            if not matched:
                continue
            per_entity: dict[str, dict[str, object]] = {}
            for service_id in matched:
                domain, _, service = service_id.partition(".")
                fields = _service_field_names(self.services_schema.get(domain, {}).get(service)) or []
                per_entity.setdefault(domain, {})[service] = {
                    "supports_response": self.supports_response(domain, service),
                    "fields": fields,
                }
            result[entity_id] = per_entity
        return result

    def _policy_block(
        self,
        settings: SandboxSettings,
        snapshot: HomeSnapshot,
        domain: str,
        service: str,
        service_data: Mapping[str, object] | None,
        target: Mapping[str, object] | None,
        blocking: bool,
        return_response: bool,
    ) -> _PolicyBlock | None:
        """Evaluate snapshot-policy gates without raising; None means the call may proceed."""
        if not settings.actions_enabled:
            return _PolicyBlock("actions_disabled", {}, message="Service calls are disabled for this sandbox.")
        if settings.action_domains and domain not in settings.action_domains:
            valid_domains = bounded_strings(sorted(settings.action_domains))
            guidance: Mapping[str, object] = advise(
                snapshot,
                FailureContext(intent=Intent.CALL_SERVICE, requested=service, domain=domain, service=service),
            ).to_payload()
            return _PolicyBlock(
                "action_domain_not_allowed",
                cast(TranslationPlaceholders, {"domain": domain}),
                message=_valid_domains_message(domain, valid_domains),
                guidance=guidance,
            )
        if not self.has_service(domain, service):
            guidance = _service_not_found_guidance(snapshot, domain, service, service_data, target)
            if not self.services.get(domain):
                valid_domains = bounded_strings(sorted(self.services))
                return _PolicyBlock(
                    "service_not_found",
                    cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                    message=_valid_domains_message(domain, valid_domains),
                    guidance=guidance,
                )
            valid_services = bounded_strings(sorted(self.services[domain]))
            return _PolicyBlock(
                "service_not_found",
                cast(TranslationPlaceholders, {"domain": domain, "service": service}),
                message=_valid_services_message(domain, service, valid_services),
                guidance=guidance,
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
        return None

    def _target_capability_block(
        self,
        snapshot: HomeSnapshot,
        domain: str,
        service: str,
        resolved_target: dict[str, object] | None,
    ) -> _PolicyBlock | None:
        """Conservative stable-fact pre-block for an exclusive domain mismatch.

        Pre-blocks only when the service declares exclusive target domains and none
        of the resolved target entities match — a stable snapshot fact Home Assistant
        would reject live anyway. Field/capability support (e.g. color temperature)
        is never pre-blocked; it informs ranking and discovery, and HA decides live.
        Returns ``None`` when the service has no target metadata or the mismatch is
        not a definitive domain exclusion.
        """
        target_brief = snapshot.services_target.get(domain, {}).get(service)
        if not isinstance(target_brief, Mapping):
            return None
        resolved_ids = resolved_target.get("entity_id") if isinstance(resolved_target, Mapping) else None
        if not isinstance(resolved_ids, list) or not resolved_ids:
            return None
        excluded = sorted({eid.split(".", 1)[0] for eid in resolved_ids if isinstance(eid, str) and "." in eid})
        if not excluded:
            return None
        # Block only when every resolved domain is definitively excluded. A brief
        # that does not constrain domains returns None (not False) and skips blocking.
        if any(service_accepts_domain(target_brief, entity_domain) is not False for entity_domain in excluded):
            return None
        guidance = advise(
            snapshot,
            FailureContext(intent=Intent.RESOLVE_SELECTOR, requested=service, domain=domain, service=service),
        ).to_payload()
        return _PolicyBlock(
            "service_target_not_supported",
            cast(TranslationPlaceholders, {"domain": domain, "service": service}),
            message=_service_target_not_supported_message(domain, service, excluded, guidance),
            guidance=guidance,
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
            nonlocal blocking, return_response

            runtime = require_runtime(None)
            settings = runtime.settings
            # Response-mode accommodation: any service that can return a response
            # runs blocking with return_response=True, so the LLM never has to
            # remember the flag. NONE services are untouched and still reject an
            # erroneous return_response=True in the policy gate below.
            if self.supports_response(domain, service) in (
                SupportsResponse.ONLY.value,
                SupportsResponse.OPTIONAL.value,
            ):
                blocking = True
                return_response = True
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
                guidance: Mapping[str, object] | None = None,
            ) -> None:
                # Policy blocks are non-raising: record an errored action and let
                # the call return None so execution stays status="ok" with a
                # recorded errored action (Decision 3: live failures keep raising).
                runtime.state.actions.append(
                    _action_record(
                        _request_action(raw_target),
                        status="error",
                        response=None,
                        error=_action_error(key, message, guidance=guidance),
                    )
                )

            # Policy gate (non-raising).
            snapshot = require_snapshot()
            if (
                block := self._policy_block(
                    settings,
                    snapshot,
                    domain,
                    service,
                    cleaned_service_data,
                    raw_target,
                    blocking,
                    return_response,
                )
            ) is not None:
                _block(block.key, block.placeholders, message=block.message, guidance=block.guidance)
                return None

            # Target visibility resolution with auto-resolve.
            target_outcome = self._visible_target(merged_target, domain, service)
            if isinstance(target_outcome, _UnresolvedTarget):
                _block(
                    "service_target_not_visible",
                    cast(TranslationPlaceholders, {"entity_id": target_outcome.requested}),
                    message=_target_not_found_message(
                        target_outcome.requested,
                        target_outcome.selector,
                        target_outcome.scope_domain,
                        target_outcome.guidance,
                    ),
                    guidance=target_outcome.guidance,
                )
                return None
            resolved_target = target_outcome.target

            # Conservative stable-fact pre-block: a service whose declared target
            # excludes every resolved entity's domain is blocked with guidance
            # instead of forwarding a call Home Assistant would reject live.
            if (cap_block := self._target_capability_block(snapshot, domain, service, resolved_target)) is not None:
                _block(cap_block.key, cap_block.placeholders, message=cap_block.message, guidance=cap_block.guidance)
                return None

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
                # Mark that this run dispatched a live write so later recorder-backed
                # reads know to synchronize before reading (read-after-write). Set
                # before invoking so a partial write (or a failure) still counts.
                runtime.state.live_write_dispatched = True
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
                    guidance=advise(
                        require_snapshot(),
                        FailureContext(
                            intent=Intent.CALL_SERVICE,
                            requested=service,
                            domain=domain,
                            service=service,
                            service_data=cleaned_service_data or {},
                        ),
                    ).to_payload(),
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
        service: str | None = None,
    ) -> _ResolvedTarget | _UnresolvedTarget:
        """Resolve supported HA target selectors to visible entity IDs."""
        snapshot = require_snapshot()
        if not target:
            return _ResolvedTarget(cast(dict[str, object] | None, json_safe(target)))

        entity_ids: set[str] = set()
        supported_values: list[str] = []
        supported_keys: list[str] = []
        adjustments: list[dict[str, object]] = []
        memory = require_runtime(None).memory

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
                    if memory is not None and resolved_entity_id != entity_id:
                        # Persist only after the fresh snapshot resolver chose a
                        # visible entity id for this requested target literal.
                        memory.record(entity_id, resolved_entity_id)
                    adjustments.append(_target_entity_resolved_adjustment(entity_id, resolved_entity_id))
                else:
                    guidance = advise(
                        snapshot,
                        FailureContext(
                            intent=Intent.RESOLVE_SELECTOR,
                            requested=entity_id,
                            domain=resolve_domain,
                            service=service or "",
                            selector="entity_id",
                        ),
                    ).to_payload()
                    return _UnresolvedTarget(
                        requested=entity_id,
                        selector="entity_id",
                        scope_domain=resolve_domain,
                        guidance=guidance,
                    )
        first_supported: tuple[str, str] | None = None
        first_existing_selector: tuple[str, str] | None = None
        for selector in AGGREGATE_SELECTOR_KEYS:
            if selector not in target:
                continue
            supported_keys.append(selector)
            for requested in _target_values(target[selector]):
                supported_values.append(requested)
                if first_supported is None:
                    first_supported = (selector, requested)
                if first_existing_selector is None and _selector_exists(snapshot, selector, requested):
                    first_existing_selector = (selector, requested)
        for selector, requested_expansions in expand_aggregate_selectors(snapshot, target).items():
            for requested, resolved in requested_expansions:
                domain_resolved = _domain_filtered_entity_ids(snapshot, resolved, domain)
                entity_ids.update(domain_resolved)
                adjustments.append(_target_selector_expanded_adjustment(selector, requested, domain_resolved))

        if entity_ids:
            return _ResolvedTarget({"entity_id": sorted(entity_ids)}, tuple(adjustments))
        if first_existing_selector is not None:
            selector, requested = first_existing_selector
            guidance = advise(
                snapshot,
                FailureContext(
                    intent=Intent.RESOLVE_SELECTOR,
                    requested=requested,
                    domain=domain,
                    service=service or "",
                    selector=selector,
                ),
            ).to_payload()
            return _UnresolvedTarget(
                requested=requested,
                selector=selector,
                scope_domain=domain,
                guidance=guidance,
            )
        if supported_values:
            selector, requested = first_supported or (supported_keys[0], supported_values[0])
            guidance = advise(
                snapshot,
                FailureContext(
                    intent=Intent.RESOLVE_SELECTOR,
                    requested=requested,
                    domain=domain,
                    service=service or "",
                    selector=selector,
                ),
            ).to_payload()
            return _UnresolvedTarget(
                requested=requested,
                selector=selector,
                scope_domain=domain,
                guidance=guidance,
            )
        if supported_keys:
            return _UnresolvedTarget(
                requested=supported_keys[0],
                selector=supported_keys[0],
                scope_domain=domain,
                guidance=None,
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
        return HelperExecutionError("services.async_call", key, placeholders)

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


def _target_values(value: object) -> list[str]:
    """Return HA target selector values as strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _expand_target_entities(snapshot: HomeSnapshot, target: Mapping[str, object] | None) -> set[str]:
    """Resolve HA target selectors to visible entity ids for read-only discovery.

    Unlike ``_visible_target`` this performs no fuzzy auto-resolution and records
    no action: it is a pure selector expansion used by ``async_services_for_target``.
    """
    if not isinstance(target, Mapping):
        return set()
    entity_ids: set[str] = set()
    if "entity_id" in target:
        for entity_id in _target_values(target["entity_id"]):
            if entity_id in snapshot.states:
                entity_ids.add(entity_id)
    for requested_expansions in expand_aggregate_selectors(snapshot, target).values():
        for _requested, resolved in requested_expansions:
            entity_ids.update(resolved)
    return entity_ids


def _selector_exists(snapshot: HomeSnapshot, selector: str, requested: str) -> bool:
    """Return whether an aggregate selector value exists in the frozen snapshot."""
    if selector == "area_id":
        return requested in snapshot.areas
    if selector == "device_id":
        return requested in snapshot.devices
    if selector == "floor_id":
        return requested in snapshot.floors
    if selector in {"label_id", "label"}:
        return requested in snapshot.labels
    return False


def _domain_filtered_entity_ids(snapshot: HomeSnapshot, entity_ids: tuple[str, ...], domain: str) -> tuple[str, ...]:
    """Return selector-expanded entity ids scoped to the requested service domain."""
    if not domain:
        return tuple(sorted(entity_ids))
    return tuple(sorted(entity_id for entity_id in entity_ids if snapshot.states[entity_id].domain == domain))


def _service_not_found_guidance(
    snapshot: HomeSnapshot,
    domain: str,
    service: str,
    service_data: Mapping[str, object] | None,
    target: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Return service-name guidance, annotated when the requested target is not visible."""
    guidance = advise(
        snapshot,
        FailureContext(
            intent=Intent.CALL_SERVICE,
            requested=service,
            domain=domain if domain in snapshot.services else "",
            service=service,
            service_data=service_data or {},
        ),
    ).to_payload()
    if (missing_target := _first_missing_target_literal(snapshot, target)) is not None:
        guidance = dict(guidance)
        reason = str(guidance.get("reason", ""))
        guidance["reason"] = f"{reason} Target `{missing_target}` is not visible in the current snapshot.".strip()
    return guidance


def _first_missing_target_literal(snapshot: HomeSnapshot, target: Mapping[str, object] | None) -> str | None:
    """Return the first target selector literal absent from the frozen visible snapshot."""
    if not isinstance(target, Mapping):
        return None
    indexes: dict[str, Mapping[str, object]] = {
        "entity_id": snapshot.states,
        "area_id": snapshot.areas,
        "device_id": snapshot.devices,
        "floor_id": snapshot.floors,
        "label_id": snapshot.labels,
        "label": snapshot.labels,
    }
    for selector, visible in indexes.items():
        if selector not in target:
            continue
        # Visibility fact only annotates guidance; policy blocking remains unchanged.
        for requested in _target_values(target[selector]):
            if requested not in visible:
                return requested
    return None


def _guidance_candidate_ids(guidance: Mapping[str, object] | None) -> list[str]:
    """Return candidate ids from a serialized guidance payload for legacy message prose."""
    if not guidance:
        return []
    candidates = guidance.get("candidates")
    if not isinstance(candidates, list):
        return []
    ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("id")
        if isinstance(candidate_id, str) and candidate_id:
            ids.append(candidate_id)
    return ids


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


def _target_not_found_message(
    requested: str,
    selector: str,
    scope_domain: str,
    guidance: Mapping[str, object] | None,
) -> str:
    """Return the compact target-layer repair message."""
    candidate_ids = _guidance_candidate_ids(guidance)
    subject = _selector_subject(selector)
    if candidate_ids:
        if selector == "entity_id":
            return f"Entity target '{requested}' not found among visible {scope_domain} entities. Did you mean: {', '.join(candidate_ids)}."
        return f"{subject} target '{requested}' did not resolve to visible {scope_domain} entities. Did you mean: {', '.join(candidate_ids)}."
    if selector == "entity_id":
        return f"Entity target '{requested}' not found among visible {scope_domain} entities."
    return f"{subject} target '{requested}' did not resolve to visible {scope_domain} entities."


def _selector_subject(selector: str) -> str:
    """Return the target-selector noun used in compact repair messages."""
    return {
        "area_id": "Area",
        "device_id": "Device",
        "floor_id": "Floor",
        "label_id": "Label",
        "label": "Label",
        "entity_id": "Entity",
    }.get(selector, "Selector")


def _service_target_not_supported_message(
    domain: str,
    service: str,
    excluded_domains: list[str],
    guidance: Mapping[str, object] | None,
) -> str:
    """Return the compact domain-mismatch repair message."""
    candidate_ids = _guidance_candidate_ids(guidance)
    accepted = f" Try: {', '.join(candidate_ids)}." if candidate_ids else ""
    return f"Service '{domain}.{service}' does not target {', '.join(excluded_domains)} entities.{accepted}"


def _service_field_names(brief: ServiceSchemaBrief | None) -> list[str] | None:
    """Return bounded field names from a service schema brief."""
    if brief is None:
        return None
    names = sorted(str(field["name"]) for field in raw_service_field_names(brief))
    return bounded_strings(names) if names else None


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
    guidance: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    """A visibility-resolved service target (entity_id list or empty)."""

    target: dict[str, object] | None
    adjustments: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _UnresolvedTarget:
    """A target that could not resolve to visible entities; carries structured guidance."""

    requested: str
    selector: str
    scope_domain: str
    guidance: Mapping[str, object] | None = None


def _action_error(
    key: str,
    message: str,
    *,
    guidance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the JSON-safe action error shape."""
    clean = " ".join(message.split())
    error: dict[str, object] = {
        "key": key,
        "message": clean if clean and clean != key else f"Resolve '{key}' before retrying.",
    }
    if guidance:
        error["guidance"] = dict(guidance)
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
            "before execution; report the applied entity id to the user; no retry needed."
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
