"""Shared eval-only seams that do not emulate production tools."""

from collections.abc import Mapping
from dataclasses import replace

from custom_components.llm_sandbox.snapshot import finalize_snapshot
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot, SnapshotScope

EVAL_SCOPE: SnapshotScope = SnapshotScope(
    assistant="conversation",
    restrict_to_assist_exposed=False,
    exclude_hidden=True,
    excluded_entity_categories=frozenset({"config"}),
    include_all_diagnostics=False,
)

USEFUL_DIAGNOSTIC_DEVICE_CLASSES = frozenset(
    {"battery", "battery_charging", "signal_strength", "connectivity", "problem", "power"}
)


def apply_scope(
    snapshot: HomeSnapshot,
    scope: SnapshotScope,
    *,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Return a new snapshot with entities failing the offline scope checks removed.

    Mirrors production ``_passes_visibility`` for the offline-applicable fields only
    (``exclude_hidden`` + ``excluded_entity_categories``). Assist-exposure filtering
    needs live HA and stays a ``build_snapshot`` concern; the eval scope disables it.
    Collection pruning, state enrichment, and index rebuilding are delegated to the
    production snapshot finalizer.
    """
    visible: set[str] = set()
    for entity_id in snapshot.states:
        entry = snapshot.entities.get(entity_id)
        # Branch boundary: state-only entities skip registry-characteristic visibility checks.
        if entry is None:
            visible.add(entity_id)
            continue
        # Branch boundary: hidden registry entities are excluded when the eval scope asks for it.
        if scope.exclude_hidden and entry.hidden_by is not None:
            continue
        # Branch boundary: configured registry categories are excluded by the eval scope.
        if entry.entity_category in scope.excluded_entity_categories:
            continue
        # Branch boundary: selective diagnostics mirror product snapshot filtering for offline evals.
        if (
            entry.entity_category == "diagnostic"
            and not scope.include_all_diagnostics
            and (entry.device_class or entry.original_device_class) not in USEFUL_DIAGNOSTIC_DEVICE_CLASSES
        ):
            continue
        visible.add(entity_id)

    finalized = finalize_snapshot(snapshot, visible=visible, anchor_device_id=anchor_device_id)
    # Preserve fixture insertion order; the finalizer rebuilds these collections from a set.
    ordered_states = {
        entity_id: finalized.states[entity_id] for entity_id in snapshot.states if entity_id in finalized.states
    }
    ordered_entities = {
        entity_id: finalized.entities[entity_id] for entity_id in snapshot.entities if entity_id in finalized.entities
    }
    return replace(finalized, states=ordered_states, entities=ordered_entities)


class RecordingInvoker:
    """Non-live service invoker: records validated ProposedAction dicts, returns None.

    This is the ONLY live seam in the executor path. It never touches Home Assistant.
    """

    def __init__(self) -> None:
        """Initialize the in-memory action recording list."""
        self.calls: list[dict[str, object]] = []

    async def __call__(self, action: dict[str, object]) -> object:
        """Record one already-validated action without dispatching to live Home Assistant."""
        # Safety constraint: copy the proposed action and never call hass.services or any live callback.
        self.calls.append(dict(action))
        return None


def _for_scoring(action: Mapping[str, object]) -> dict[str, object]:
    """Normalize a recorded action so the frozen scorer can read domain/service separately.

    Production compact records fold domain into service (``domain.service``); the
    invoker's ProposedAction already carries separate domain/service. This split
    is eval-only and never mutates the model-facing result.
    """
    normalized = dict(action)
    if "domain" not in normalized:
        service = normalized.get("service")
        if isinstance(service, str) and "." in service:
            domain, _, svc = service.partition(".")
            normalized["domain"] = domain
            normalized["service"] = svc
    return normalized
