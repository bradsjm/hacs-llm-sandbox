"""Unit tests for snapshot-backed service target resolution."""

from collections.abc import Sequence
from dataclasses import replace

import pytest
from custom_components.llm_sandbox.llm_api.resolution import (
    available_hint,
    rank_candidates_for_service,
    resolve_target_entity,
)
from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeConfig,
    SafeContext,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)


@pytest.mark.parametrize(
    ("state_specs", "domain", "query", "expected_resolved", "expected_candidates"),
    [
        pytest.param(
            (("climate.upstairs_thermostat", None),),
            "climate",
            "thermostat",
            "climate.upstairs_thermostat",
            (),
            id="query-subset-of-object-id",
        ),
        pytest.param(
            (("climate.upstairs_thermostat", None),),
            "climate",
            "climate.upstairs_thermostat_extra",
            "climate.upstairs_thermostat",
            (),
            id="candidate-subset-of-query",
        ),
        pytest.param(
            (("light.kitchen_ceiling", "Kitchen Ceiling"),),
            "light",
            "kitchen ceiling",
            "light.kitchen_ceiling",
            (),
            id="name-based-equal-token-set",
        ),
        pytest.param(
            (("light.kitchen_sink", "Kitchen Sink"),),
            "light",
            "light.kitchen_ceiling",
            None,
            (),
            id="review-block-kitchen-ceiling-vs-sink",
        ),
        pytest.param(
            (("lock.front_door", "Front Door"),),
            "lock",
            "lock.back_door",
            None,
            (),
            id="review-block-back-door-vs-front-door",
        ),
        pytest.param(
            (("light.kitchen_ceiling", "Kitchen Ceiling"),),
            "light",
            "light.bedroom_ceiling",
            None,
            (),
            id="single-shared-token-non-containment",
        ),
        pytest.param(
            (("light.kitchen", "Kitchen"),),
            "light",
            "!!!",
            None,
            (),
            id="empty-query-token-guard",
        ),
    ],
)
def test_resolve_target_entity_uses_bidirectional_token_containment(
    state_specs: Sequence[tuple[str, str | None]],
    domain: str,
    query: str,
    expected_resolved: str | None,
    expected_candidates: tuple[str, ...],
) -> None:
    """Resolve only exact ids or unique same-domain bidirectional containment matches."""
    result = resolve_target_entity(_snapshot(state_specs), query, domain)

    assert result.resolved == expected_resolved
    assert result.is_resolved is (expected_resolved is not None)
    assert {candidate.entity_id for candidate in result.candidates} == set(expected_candidates)


def test_resolve_target_entity_returns_ambiguous_candidates_for_multiple_containment_matches() -> None:
    """Multiple visible candidates containing the query are surfaced, not resolved."""
    result = resolve_target_entity(
        _snapshot(
            (
                ("fan.ceiling", None),
                ("fan.living_ceiling", "Living Ceiling"),
            )
        ),
        "ceiling",
        "fan",
    )

    assert result.resolved is None
    assert not result.is_resolved
    assert [candidate.entity_id for candidate in result.candidates] == ["fan.ceiling", "fan.living_ceiling"]


@pytest.mark.parametrize(
    ("state_count", "expect_overflow"),
    [
        pytest.param(8, False, id="at-discovery-limit"),
        pytest.param(9, True, id="above-discovery-limit"),
    ],
)
def test_available_hint_marks_overflow_only_above_discovery_limit(
    state_count: int,
    expect_overflow: bool,
) -> None:
    """Visible-domain hints signal overflow only when more entities exist than displayed."""
    state_specs = tuple((f"light.fixture_{index}", f"Fixture {index}") for index in range(state_count))

    hint = available_hint(_snapshot(state_specs), "light")

    assert hint.endswith(" ...") is expect_overflow


def test_rank_candidates_for_service_surfaces_targeted_entities_first() -> None:
    """Fix-list candidates the service targets sort ahead of unrelated same-domain entities."""
    from custom_components.llm_sandbox.llm_api.resolution import CandidateTarget

    snapshot = replace(
        _snapshot(
            (
                ("cover.blind_supported", None),
                ("cover.blind_plain", None),
            )
        ),
        services_target={
            "cover": {
                "stop_cover": {
                    "entity": [{"domain": ["cover"], "supported_features": [4]}],
                }
            }
        },
        states={
            "cover.blind_supported": _state_with_features("cover.blind_supported", 4),
            "cover.blind_plain": _state_with_features("cover.blind_plain", 0),
        },
    )
    candidates = (
        CandidateTarget(entity_id="cover.blind_plain", name=None, object_id="blind_plain"),
        CandidateTarget(entity_id="cover.blind_supported", name=None, object_id="blind_supported"),
    )

    ranked = rank_candidates_for_service(snapshot, candidates, "cover", "stop_cover")

    assert [candidate.entity_id for candidate in ranked] == [
        "cover.blind_supported",
        "cover.blind_plain",
    ]


def test_rank_candidates_for_service_preserves_order_without_target_metadata() -> None:
    """When the service has no target metadata, ranking is a no-op (HA decides)."""
    from custom_components.llm_sandbox.llm_api.resolution import CandidateTarget

    snapshot = _snapshot((("light.alpha", None), ("light.beta", None)))
    candidates = (
        CandidateTarget(entity_id="light.alpha", name=None, object_id="alpha"),
        CandidateTarget(entity_id="light.beta", name=None, object_id="beta"),
    )

    ranked = rank_candidates_for_service(snapshot, candidates, "light", "turn_on")

    assert [candidate.entity_id for candidate in ranked] == ["light.alpha", "light.beta"]


def _state_with_features(entity_id: str, supported_features: int) -> SafeState:
    """Build a state record carrying a supported_features attribute."""
    base = _state(entity_id, None)
    return replace(base, attributes={"supported_features": supported_features})


def _snapshot(state_specs: Sequence[tuple[str, str | None]]) -> HomeSnapshot:
    """Build a minimal snapshot for pure resolution tests."""
    return HomeSnapshot(
        created_at="2026-06-29T00:00:00+00:00",
        states={entity_id: _state(entity_id, name) for entity_id, name in state_specs},
        entities={},
        devices={},
        areas={},
        floors={},
        config=_config(),
        services={},
        services_supports_response={},
        indexes=SnapshotIndexes(
            entity_ids_by_device_id={},
            entity_ids_by_area_id={},
            device_ids_by_area_id={},
            entity_ids_by_config_entry_id={},
            entity_ids_by_label={},
            device_ids_by_label={},
            area_ids_by_floor_id={},
        ),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema={},
    )


def _state(entity_id: str, name: str | None) -> SafeState:
    """Build a visible state record with only resolution-relevant fields populated."""
    domain, object_id = entity_id.split(".", 1)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state="off",
        attributes={},
        last_changed="2026-06-29T00:00:00+00:00",
        last_reported="2026-06-29T00:00:00+00:00",
        last_updated="2026-06-29T00:00:00+00:00",
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
        area_id=None,
        device_id=None,
        platform=domain,
        unique_id=entity_id,
    )


def _config() -> SafeConfig:
    """Build a minimal config record for snapshot construction."""
    return SafeConfig(
        location_name="Home",
        latitude=0.0,
        longitude=0.0,
        elevation=0,
        time_zone="UTC",
        language="en",
        country=None,
        currency="USD",
        internal_url=None,
        external_url=None,
        units=SafeUnitSystem(
            temperature_unit="°C",
            length_unit="m",
            mass_unit="kg",
            pressure_unit="Pa",
            volume_unit="L",
            area_unit="m²",
            wind_speed_unit="m/s",
            accumulated_precipitation_unit="mm",
        ),
    )
