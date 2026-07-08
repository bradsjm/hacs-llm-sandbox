"""Unit tests for snapshot-backed service target resolution."""

from collections.abc import Sequence

import pytest
from custom_components.llm_sandbox.llm_api.resolution import resolve_target_entity
from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeConfig,
    SafeContext,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)

_TIMESTAMP = 1782691200.0


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
        last_changed_timestamp=_TIMESTAMP,
        last_reported="2026-06-29T00:00:00+00:00",
        last_reported_timestamp=_TIMESTAMP,
        last_updated="2026-06-29T00:00:00+00:00",
        last_updated_timestamp=_TIMESTAMP,
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
