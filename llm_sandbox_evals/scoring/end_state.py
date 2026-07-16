"""Pure narrow post-run overlay reducer for end-state predicate scoring.

Applies explicit ``light``/``switch``/``cover`` state transitions plus narrow
light, climate, and cover attribute effects from ordered ``RecordingInvoker``
calls to a copied seed map. Unsupported services, selectors, and attributes
leave the overlay unchanged. This eval-only primitive never touches live Home
Assistant objects.
"""

from collections.abc import Mapping, Sequence

from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import (
    DesiredEntity,
    EndStateComparison,
    EndStateResult,
    OverlayStateSeed,
)

_SUPPORTED_STATES_BY_DOMAIN: dict[str, frozenset[str]] = {
    "light": frozenset({"on", "off"}),
    "switch": frozenset({"on", "off"}),
    "cover": frozenset({"open", "closed"}),
}
_SUPPORTED_ATTRIBUTES_BY_DOMAIN: dict[str, frozenset[str]] = {
    "light": frozenset({"brightness", "color_temp_kelvin"}),
    "climate": frozenset({"temperature"}),
    "cover": frozenset({"current_position"}),
}
_SUPPORTED_TRANSITIONS: dict[tuple[str, str], str] = {
    ("light", "turn_on"): "on",
    ("light", "turn_off"): "off",
    ("switch", "turn_on"): "on",
    ("switch", "turn_off"): "off",
    ("cover", "open_cover"): "open",
    ("cover", "close_cover"): "closed",
}


def extract_overlay_seeds(
    snapshot: HomeSnapshot,
    desired_entities: Sequence[DesiredEntity],
) -> tuple[OverlayStateSeed, ...]:
    """Select only predicate-relevant frozen values from the scoped snapshot.

    Absent entities produce no seed; the assessor treats that as unevaluable.
    Only authored attribute keys are copied so persisted traces stay sparse.
    """
    seeds: list[OverlayStateSeed] = []
    for predicate in desired_entities:
        state = snapshot.states.get(predicate.entity_id)
        if state is None:
            continue
        attributes = {key: state.attributes.get(key) for key in predicate.attributes}
        seeds.append(OverlayStateSeed(state.entity_id, state.domain, state.state, attributes))
    return tuple(seeds)


def assess_end_state(
    desired_entities: Sequence[DesiredEntity],
    seeds: Sequence[OverlayStateSeed],
    calls: Sequence[Mapping[str, object]],
) -> EndStateResult:
    """Evaluate sparse desired final-value predicates against a post-run overlay.

    Returns ``not_authored`` when no predicates exist, ``unevaluable`` when
    any authored field lacks reducer support or an entity seed, and
    ``satisfied``/``unsatisfied`` when all predicates are evaluable.
    """
    if not desired_entities:
        return EndStateResult("not_authored", False, False)

    seed_map = {seed.entity_id: seed for seed in seeds}
    for predicate in desired_entities:
        seed = seed_map.get(predicate.entity_id)
        # Branch boundary: every predicate must have a seed and reducer support for all authored fields.
        if seed is None:
            return EndStateResult("unevaluable", False, False)
        if predicate.state is not None and seed.state not in _SUPPORTED_STATES_BY_DOMAIN.get(
            seed.domain, frozenset()
        ):
            return EndStateResult("unevaluable", False, False)
        if predicate.attributes and not predicate.attributes.keys() <= _SUPPORTED_ATTRIBUTES_BY_DOMAIN.get(
            seed.domain, frozenset()
        ):
            return EndStateResult("unevaluable", False, False)

    # All predicates are evaluable — reduce calls in order.
    overlay = dict(seed_map)
    for call in calls:
        _apply_call(overlay, call)

    comparisons: list[EndStateComparison] = []
    for predicate in desired_entities:
        actual = overlay[predicate.entity_id]
        state_matches = predicate.state is None or actual.state == predicate.state
        attributes_match = all(actual.attributes.get(key) == value for key, value in predicate.attributes.items())
        comparisons.append(
            EndStateComparison(
                predicate,
                actual.state,
                dict(actual.attributes),
                state_matches and attributes_match,
            )
        )
    all_satisfied = all(comparison.matched for comparison in comparisons)
    return EndStateResult(
        "satisfied" if all_satisfied else "unsatisfied",
        True,
        all_satisfied,
        tuple(comparisons),
    )


def _apply_call(overlay: dict[str, OverlayStateSeed], call: Mapping[str, object]) -> None:
    """Apply one ordered call to the overlay if it has a supported direct effect.

    Unsupported services, indirect selectors, errored calls, and unsupported
    service data have no overlay effect.
    """
    if call.get("status") == "error":
        return

    domain = call.get("domain")
    service = call.get("service")
    if not isinstance(domain, str) or not isinstance(service, str):
        return

    transition = _SUPPORTED_TRANSITIONS.get((domain, service))
    is_toggle = service == "toggle"
    effects = _attribute_effects(domain, service, call.get("service_data"))
    if transition is None and not is_toggle and effects is None:
        return

    targets = _direct_targets(call)
    if not targets:
        return

    for entity_id in targets:
        seed = overlay.get(entity_id)
        if seed is None or seed.domain != domain:
            continue
        if is_toggle:
            if seed.state not in ("on", "off"):
                continue
            new_state = "off" if seed.state == "on" else "on"
        elif transition is not None:
            new_state = transition
        elif domain == "cover" and service == "set_cover_position" and effects is not None:
            new_state = "closed" if effects["current_position"] == 0 else "open"
        else:
            new_state = seed.state
        attributes = dict(seed.attributes)
        if effects is not None:
            attributes.update((key, value) for key, value in effects.items() if key in attributes)
        # Mutation point: replace the eval-only seed; never mutate the frozen snapshot.
        overlay[entity_id] = OverlayStateSeed(seed.entity_id, seed.domain, new_state, attributes)


def _attribute_effects(domain: str, service: str, service_data: object) -> dict[str, object] | None:
    """Return supported sparse attribute effects for one explicit service."""
    if domain == "cover" and service == "open_cover":
        return {"current_position": 100}
    if domain == "cover" and service == "close_cover":
        return {"current_position": 0}
    if not isinstance(service_data, Mapping):
        return {} if domain == "light" and service == "turn_on" else None

    if domain == "light" and service == "turn_on":
        effects: dict[str, object] = {}
        brightness_pct = service_data.get("brightness_pct")
        if (
            isinstance(brightness_pct, int | float)
            and not isinstance(brightness_pct, bool)
            and 0 <= brightness_pct <= 100
        ):
            effects["brightness"] = round(255 * brightness_pct / 100)
        brightness = service_data.get("brightness")
        if isinstance(brightness, int) and not isinstance(brightness, bool) and 0 <= brightness <= 255:
            effects["brightness"] = brightness
        color_temp_kelvin = service_data.get("color_temp_kelvin")
        if isinstance(color_temp_kelvin, int) and not isinstance(color_temp_kelvin, bool):
            effects["color_temp_kelvin"] = color_temp_kelvin
        return effects

    if domain == "climate" and service == "set_temperature":
        temperature = service_data.get("temperature")
        if isinstance(temperature, int | float) and not isinstance(temperature, bool):
            return {"temperature": temperature}
        return None

    if domain == "cover" and service == "set_cover_position":
        position = service_data.get("position")
        if isinstance(position, int) and not isinstance(position, bool) and 0 <= position <= 100:
            return {"current_position": position}
    return None


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
