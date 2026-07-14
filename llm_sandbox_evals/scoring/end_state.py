"""Pure narrow post-run overlay reducer for end-state predicate scoring.

Applies only direct ``light``/``switch`` ``turn_on``/``turn_off``/``toggle``
transitions from ordered ``RecordingInvoker`` calls to a copied seed state
map.  Unsupported services, indirect selectors, attribute effects, and
service data leave the overlay unchanged.  This is an eval-only primitive;
it never touches live Home Assistant objects.
"""

from collections.abc import Mapping, Sequence

from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import (
    DesiredState,
    EndStateComparison,
    EndStateResult,
    OverlayStateSeed,
)

_SUPPORTED_TRANSITIONS: dict[tuple[str, str], str] = {
    ("light", "turn_on"): "on",
    ("light", "turn_off"): "off",
    ("switch", "turn_on"): "on",
    ("switch", "turn_off"): "off",
}


def extract_overlay_seeds(
    snapshot: HomeSnapshot,
    desired_states: Sequence[DesiredState],
) -> tuple[OverlayStateSeed, ...]:
    """Select only predicate-relevant frozen states from the scoped snapshot.

    Absent entities produce no seed; the assessor treats that as unevaluable.
    Only ``entity_id``, ``domain``, and ``state`` are copied — never
    attributes — so the persisted trace stays minimal and JSON-safe.
    """
    seeds: list[OverlayStateSeed] = []
    for predicate in desired_states:
        state = snapshot.states.get(predicate.entity_id)
        if state is None:
            continue
        seeds.append(OverlayStateSeed(state.entity_id, state.domain, state.state))
    return tuple(seeds)


def assess_end_state(
    desired_states: Sequence[DesiredState],
    seeds: Sequence[OverlayStateSeed],
    calls: Sequence[Mapping[str, object]],
) -> EndStateResult:
    """Evaluate desired-state predicates against a post-run overlay.

    Returns ``not_authored`` when no predicates exist, ``unevaluable`` when
    any predicate lacks a valid binary light/switch seed, and
    ``satisfied``/``unsatisfied`` when all predicates are evaluable.
    """
    if not desired_states:
        return EndStateResult("not_authored", False, False)

    seed_map = {seed.entity_id: seed for seed in seeds}
    for predicate in desired_states:
        seed = seed_map.get(predicate.entity_id)
        # Branch boundary: every predicate must have a valid binary seed or the entire set is unevaluable.
        if seed is None or seed.domain not in ("light", "switch") or seed.state not in ("on", "off"):
            return EndStateResult("unevaluable", False, False)

    # All predicates are evaluable — reduce calls in order.
    overlay = dict(seed_map)
    for call in calls:
        _apply_call(overlay, call)

    comparisons = tuple(
        EndStateComparison(
            predicate,
            (actual := overlay[predicate.entity_id].state),
            actual == predicate.state,
        )
        for predicate in desired_states
    )
    all_satisfied = all(comparison.matched for comparison in comparisons)
    return EndStateResult(
        "satisfied" if all_satisfied else "unsatisfied",
        True,
        all_satisfied,
        comparisons,
    )


def _apply_call(overlay: dict[str, OverlayStateSeed], call: Mapping[str, object]) -> None:
    """Apply one ordered call to the overlay if it is a supported direct transition.

    Unsupported services, indirect selectors, empty/duplicate target collections,
    and service data have no overlay effect.
    """
    domain = call.get("domain")
    service = call.get("service")
    if not isinstance(domain, str) or not isinstance(service, str):
        return

    transition = _SUPPORTED_TRANSITIONS.get((domain, service))
    is_toggle = service == "toggle"
    if transition is None and not is_toggle:
        return

    targets = _direct_targets(call)
    if not targets:
        return

    for entity_id in targets:
        seed = overlay.get(entity_id)
        if seed is None or seed.domain != domain:
            continue
        if seed.state not in ("on", "off"):
            continue
        if is_toggle:
            new_state = "off" if seed.state == "on" else "on"
        elif transition is not None:
            new_state = transition
        else:
            continue
        # State mutation point: overlay is an eval-only copy, never the frozen snapshot.
        overlay[entity_id] = OverlayStateSeed(seed.entity_id, seed.domain, new_state)


def _direct_targets(call: Mapping[str, object]) -> tuple[str, ...]:
    """Extract a duplicate-free, nonempty direct entity-id target list from a call.

    Only ``target.entity_id`` (str or list) and ``target.entity_ids`` (list)
    are recognized; area/device/label selectors are never expanded.
    """
    target = call.get("target")
    if not isinstance(target, Mapping):
        return ()

    values: list[str] = []
    for key in ("entity_id", "entity_ids"):
        value = target.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))

    if not values:
        return ()

    seen: set[str] = set()
    unique: list[str] = []
    for entity_id in values:
        if entity_id not in seen:
            seen.add(entity_id)
            unique.append(entity_id)
    return tuple(unique)
