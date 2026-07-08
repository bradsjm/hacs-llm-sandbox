"""Behavior tests for the pure guidance recovery engine."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime

import pytest
from custom_components.llm_sandbox.llm_api.guidance import advise, score
from custom_components.llm_sandbox.llm_api.guidance.context import FailureContext, Intent
from custom_components.llm_sandbox.llm_api.guidance.payload import Candidate
from custom_components.llm_sandbox.llm_api.guidance.policy import (
    MEMORY_WRITE_ALLOWED,
    Confidence,
    decide,
)
from custom_components.llm_sandbox.llm_api.guidance.scoring import Match
from custom_components.llm_sandbox.llm_api.guidance.sources import (
    area_candidates,
    code_attribute_candidates,
    code_global_candidates,
    device_candidates,
    entity_candidates,
    floor_candidates,
    label_candidates,
    service_candidates,
    sql_column_candidates,
    sql_table_candidates,
)
from custom_components.llm_sandbox.snapshot.builder import enrich_states
from custom_components.llm_sandbox.snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeConfig,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeLabelEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
    SnapshotIndexes,
)
from homeassistant.core import SupportsResponse

_CREATED_AT = "2026-06-29T12:00:00+00:00"
type CandidateMapping = dict[str, object]
type RankedCase = list[tuple[CandidateMapping, object]]


@dataclass(frozen=True, slots=True)
class ScoreSignalCase:
    """Inputs and oracle for a single dominant score signal."""

    requested: str
    candidate: CandidateMapping
    ctx: FailureContext
    snapshot: HomeSnapshot | None
    assertion: Callable[[Match], bool]


@pytest.mark.parametrize(
    "case_name",
    [
        pytest.param("exact-id", id="exact-id"),
        pytest.param("name-token", id="name-token"),
        pytest.param("abbreviation-temp", id="abbreviation-temp"),
        pytest.param("abbreviation-hum", id="abbreviation-hum"),
        pytest.param("device-class-vocabulary", id="device-class-vocabulary"),
        pytest.param("unit-vocabulary", id="unit-vocabulary"),
        pytest.param("area-token", id="area-token"),
        pytest.param("floor-token", id="floor-token"),
        pytest.param("alias-token", id="alias-token"),
        pytest.param("service-field-overlap", id="service-field-overlap"),
        pytest.param("service-support-target", id="service-support-target"),
        pytest.param("service-support-domain-fallback", id="service-support-domain-fallback"),
    ],
)
def test_score_signals_rank_by_domain_relevant_evidence(case_name: str) -> None:
    """A. Score exposes the expected public signal for each recovery cue."""
    case = _score_signal_cases()[case_name]

    match = score(case.requested, case.candidate, case.ctx, snapshot=case.snapshot)

    assert case.assertion(match)


def test_score_tiebreak_sorts_identical_signal_candidates_by_id() -> None:
    """A. Identical signal strength sorts deterministically by ascending candidate id."""
    ctx = FailureContext(intent=Intent.READ_STATE, requested="missing", domain="sensor")
    candidates: tuple[CandidateMapping, ...] = (
        {"kind": "entity", "id": "sensor.beta", "name": "Beta", "domain": "sensor"},
        {"kind": "entity", "id": "sensor.alpha", "name": "Alpha", "domain": "sensor"},
    )

    ranked = [(candidate, score(ctx.requested, candidate, ctx)) for candidate in candidates]
    ranked.sort(key=lambda item: item[1].key(), reverse=True)

    assert [str(candidate["id"]) for candidate, _match in ranked] == ["sensor.alpha", "sensor.beta"]


def test_sources_expose_visible_structural_candidate_surfaces() -> None:
    """B. Candidate sources expose scoped, structured, visible inventory facts."""
    snapshot = _home_snapshot()

    sensors = {str(candidate["id"]): candidate for candidate in entity_candidates(snapshot, "sensor")}
    all_entities = {str(candidate["id"]): candidate for candidate in entity_candidates(snapshot)}
    fan_services = {str(candidate["id"]): candidate for candidate in service_candidates(snapshot, "fan")}
    table_names = {str(candidate["id"]) for candidate in sql_table_candidates()}
    state_columns = {str(candidate["id"]) for candidate in sql_column_candidates("states")}

    assert {"name", "area_name", "floor_name", "device_class", "unit", "domain"} <= set(sensors["sensor.living_temp"])
    assert "switch.garage_opener" not in all_entities
    assert set(sensors) == {"sensor.bedroom_humidity", "sensor.living_temp", "sensor.office_temp"}
    assert {str(candidate["kind"]) for candidate in area_candidates(snapshot)} == {"area"}
    assert {str(candidate["kind"]) for candidate in floor_candidates(snapshot)} == {"floor"}
    assert {str(candidate["kind"]) for candidate in label_candidates(snapshot)} == {"label"}
    assert {str(candidate["kind"]) for candidate in device_candidates(snapshot)} == {"device"}
    assert fan_services["fan.set_percentage"]["fields"] == frozenset({"percentage"})
    assert "states" in table_names
    assert "entity_id" in state_columns
    assert code_global_candidates()
    assert code_attribute_candidates(("async_all", "get")) == (
        {"kind": "code_attribute", "id": "async_all", "name": "async_all", "aliases": ()},
        {"kind": "code_attribute", "id": "get", "name": "get", "aliases": ()},
    )


@pytest.mark.parametrize(
    ("confidence", "allowed"),
    [
        pytest.param(Confidence.EXACT, True, id="exact"),
        pytest.param(Confidence.HIGH, True, id="high"),
        pytest.param(Confidence.AMBIGUOUS, False, id="ambiguous"),
        pytest.param(Confidence.LISTING, False, id="listing"),
        pytest.param(Confidence.NONE, False, id="none"),
    ],
)
def test_policy_memory_write_gates_match_confidence(confidence: Confidence, allowed: bool) -> None:
    """C. Confidence policy gates memory writes."""
    assert MEMORY_WRITE_ALLOWED[confidence] is allowed


def test_policy_bounds_candidate_lists_with_overflow_context() -> None:
    """C. Guidance bounds noisy listings to the public discovery limit."""
    guidance = advise(
        _many_sensor_snapshot(12),
        FailureContext(intent=Intent.READ_STATE, requested="sensor.missing", domain="sensor"),
    )

    assert len(guidance.candidates) <= 8
    assert guidance.confidence in {Confidence.AMBIGUOUS, Confidence.LISTING}
    assert guidance.reason


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        pytest.param("exact", Confidence.EXACT, id="exact"),
        pytest.param("ambiguous", Confidence.AMBIGUOUS, id="ambiguous"),
        pytest.param("listing", Confidence.LISTING, id="listing"),
        pytest.param("none", Confidence.NONE, id="none"),
    ],
)
def test_policy_decides_confidence_from_ranked_signal(case_name: str, expected: Confidence) -> None:
    """C. Confidence decisions follow exact, ambiguous, listing, and empty candidate signals."""
    ranked = _decision_cases()[case_name]

    assert decide(ranked, FailureContext(intent=Intent.READ_STATE, requested="missing")) is expected


@pytest.mark.parametrize(
    ("ctx", "uses_imperative"),
    [
        pytest.param(
            FailureContext(intent=Intent.READ_STATE, requested="sensor.not_here", domain="sensor"),
            False,
            id="read-state",
        ),
        pytest.param(
            FailureContext(
                intent=Intent.CALL_SERVICE,
                requested="turn_on",
                domain="fan",
                service="turn_on",
                service_data={"percentage": 50},
            ),
            True,
            id="call-service",
        ),
        pytest.param(
            FailureContext(intent=Intent.RESOLVE_SELECTOR, requested="area_upstairs", domain="light"),
            False,
            id="resolve-selector",
        ),
        pytest.param(
            FailureContext(intent=Intent.QUERY_HISTORY, requested="sensor.not_here", domain="sensor"),
            False,
            id="query-history",
        ),
        pytest.param(
            FailureContext(intent=Intent.SQL_TABLE, requested="missing_table"),
            False,
            id="sql-table",
        ),
        pytest.param(
            FailureContext(intent=Intent.SQL_COLUMN, requested="missing_column", table_name="states"),
            False,
            id="sql-column",
        ),
        pytest.param(
            FailureContext(intent=Intent.CAPTURE_IMAGE, requested="camera.back", domain="camera"),
            False,
            id="capture-image",
        ),
        pytest.param(
            FailureContext(intent=Intent.CODE_NAME, requested="statez"),
            True,
            id="code-name",
        ),
        pytest.param(
            FailureContext(
                intent=Intent.CODE_ATTRIBUTE,
                requested="missing_attr",
                available_attributes=("async_all", "get"),
            ),
            False,
            id="code-attribute",
        ),
    ],
)
def test_advise_next_step_contract_for_non_exact_intents(ctx: FailureContext, uses_imperative: bool) -> None:
    """D. Non-exact guidance always gives a next step with confidence-gated imperative wording."""
    guidance = advise(_home_snapshot(), ctx)

    assert guidance.confidence is not Confidence.EXACT
    assert guidance.next_step
    assert guidance.next_step.startswith("Use `") is uses_imperative


def test_advise_exact_result_has_minimal_noise() -> None:
    """D. Exact guidance explains the resolution but does not add a retry instruction."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested="sensor.living_temp", domain="sensor"),
    )

    assert guidance.confidence is Confidence.EXACT
    assert guidance.candidates[0].id == "sensor.living_temp"
    assert guidance.reason
    assert guidance.next_step == ""


def test_to_payload_uses_documented_json_shape_and_round_trips() -> None:
    """D. Guidance serialization keeps the downstream public contract stable."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested="sensor.living_room_temperature", domain="sensor"),
    )

    payload = guidance.to_payload()
    candidates = payload["candidates"]

    assert set(payload) == {"confidence", "candidates", "reason", "next_step", "cross_kind"}
    assert payload["confidence"] == guidance.confidence.value
    assert isinstance(guidance.candidates[0], Candidate)
    assert isinstance(candidates, list)
    assert isinstance(candidates[0], dict)
    assert set(candidates[0]) == {"id", "name", "match", "detail"}
    assert payload["cross_kind"] == ""
    assert json.loads(json.dumps(payload)) == payload


def test_candidate_detail_omits_absent_values() -> None:
    """Candidate detail strings include useful values but never stringify None."""
    snapshot = replace(
        _home_snapshot(),
        states={
            "sensor.none_detail": _state(
                "sensor.none_detail",
                "1",
                "None Detail",
                attributes={"device_class": None, "unit_of_measurement": "W"},
            )
        },
        entities={},
        areas={},
        floors={},
    )

    guidance = advise(
        snapshot, FailureContext(intent=Intent.READ_STATE, requested="sensor.none_detail", domain="sensor")
    )

    assert "W" in guidance.candidates[0].detail
    assert "None" not in guidance.candidates[0].detail


def test_report_context_location_scope_override() -> None:
    """E. Living-room temperature intent ranks the living temperature sensor first."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested="sensor.living_room_temperature", domain="sensor"),
    )

    assert guidance.confidence is Confidence.HIGH
    assert guidance.candidates[0].id == "sensor.living_temp"
    assert guidance.candidates[0].match in {"name", "device_class: temperature"}
    assert "temperature" in guidance.candidates[0].detail
    assert guidance.candidates[0].id not in {"sensor.bedroom_humidity", "sensor.office_temp"}


def test_report_action_floor_target() -> None:
    """E. An area with no light match can hint its floor and only list floor-scoped lights."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(
            intent=Intent.RESOLVE_SELECTOR,
            requested="area_upstairs",
            domain="light",
            service="turn_on",
        ),
    )

    assert guidance.cross_kind == "floor_upstairs"
    assert {candidate.id for candidate in guidance.candidates} == {"light.bedroom", "light.office_desk"}
    assert "light.living" not in {candidate.id for candidate in guidance.candidates}


def test_report_real_office_blinds_close_is_domain_filtered() -> None:
    """E. Cover selector recovery stays scoped to cover entities."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(
            intent=Intent.RESOLVE_SELECTOR,
            requested="office",
            domain="cover",
            service="set_position",
        ),
    )

    assert guidance.candidates[0].id == "cover.office_blinds"
    assert {candidate.id for candidate in guidance.candidates} == {"cover.office_blinds"}
    assert "light.office_desk" not in {candidate.id for candidate in guidance.candidates}


@pytest.mark.parametrize(
    "requested",
    [pytest.param("switch.garage_opener", id="hidden-garage"), pytest.param("switch.nope", id="nonexistent")],
)
def test_report_blocked_hidden_garage_opener_is_not_imperative(requested: str) -> None:
    """E. Hidden or absent switch guidance must not say Use X."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested=requested, domain="switch"),
    )

    assert guidance.confidence in {Confidence.NONE, Confidence.LISTING}
    assert not MEMORY_WRITE_ALLOWED[guidance.confidence]
    assert not guidance.next_step.startswith("Use `")
    assert "switch.garage_opener" not in {candidate.id for candidate in guidance.candidates}


def test_report_action_domain_not_allowed_gets_non_imperative_guidance() -> None:
    """E. A service request outside available domains gets bounded absence/listing guidance."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.CALL_SERVICE, requested="unlock", domain="lock", service="unlock"),
    )

    assert guidance.confidence in {Confidence.NONE, Confidence.LISTING}
    assert guidance.reason
    assert not guidance.next_step.startswith("Use `")


def test_report_complex_hot_living_turn_on_fan_ranks_field_aware_service_first() -> None:
    """E. Service-data field overlap ranks the nearest available fan service first."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(
            intent=Intent.CALL_SERVICE,
            requested="turn_on",
            domain="fan",
            service="turn_on",
            service_data={"percentage": 50},
        ),
    )

    assert guidance.confidence is Confidence.HIGH
    assert guidance.candidates[0].id == "fan.set_percentage"
    assert guidance.candidates[0].match == "service fields"


def test_report_action_multi_sequence_ranks_nearest_fan_service_by_name() -> None:
    """E. A legacy fan speed service name resolves to the closest available fan service."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.CALL_SERVICE, requested="set_speed", domain="fan", service="set_speed"),
    )

    assert guidance.candidates[0].id == "fan.set_percentage"
    assert guidance.candidates[0].match == "name"


def test_report_action_living_fan_percentage_auto_resolves_named_fan_exactly() -> None:
    """E. Living fan percentage target auto-resolves through visible name-token containment."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(
            intent=Intent.RESOLVE_SELECTOR,
            requested="fan.living_room",
            domain="fan",
            service="set_percentage",
        ),
    )

    assert guidance.confidence is Confidence.EXACT
    assert guidance.candidates[0].id == "fan.living_fan"
    assert "fan.living_fan" in guidance.reason
    assert guidance.next_step == ""


def test_report_depth_active_repairs_does_not_fabricate_candidates() -> None:
    """E. A nonexistent repairs entity reports honest absence with no fabricated suggestions."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested="repair.active_repairs", domain="repair"),
    )

    assert guidance.confidence is Confidence.NONE
    assert guidance.candidates == []
    assert guidance.next_step


@pytest.mark.parametrize("requested", [pytest.param("", id="empty"), pytest.param("   ", id="whitespace")])
def test_guard_empty_requested_service_surface_has_no_candidates(requested: str) -> None:
    """F. Empty requests on an empty service surface do not crash or fabricate candidates."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.CALL_SERVICE, requested=requested, domain="missing"),
    )

    assert guidance.confidence is Confidence.NONE
    assert guidance.candidates == []


def test_guard_bare_requested_uses_context_domain() -> None:
    """F. Bare object ids still use the failure domain to find the visible entity."""
    guidance = advise(
        _home_snapshot(),
        FailureContext(intent=Intent.READ_STATE, requested="living_temp", domain="sensor"),
    )

    assert guidance.candidates[0].id == "sensor.living_temp"


@pytest.mark.parametrize(
    "ctx",
    [
        pytest.param(
            FailureContext(intent=Intent.READ_STATE, requested="switch.nope", domain="switch"),
            id="read-state",
        ),
        pytest.param(
            FailureContext(intent=Intent.CALL_SERVICE, requested="missing", domain="missing"),
            id="call-service",
        ),
        pytest.param(FailureContext(intent=Intent.SQL_TABLE, requested="missing_table"), id="sql-table"),
    ],
)
def test_guard_low_confidence_outcomes_are_non_imperative(ctx: FailureContext) -> None:
    """F. Listing and absence guidance never carry imperative wording."""
    guidance = advise(_home_snapshot(), ctx)

    assert not MEMORY_WRITE_ALLOWED[guidance.confidence]


def test_guard_advise_is_deterministic() -> None:
    """F. Repeated pure guidance calls produce the same serialized payload."""
    snapshot = _home_snapshot()
    ctx = FailureContext(intent=Intent.READ_STATE, requested="sensor.living_room_temperature", domain="sensor")

    first = advise(snapshot, ctx).to_payload()
    second = advise(snapshot, ctx).to_payload()

    assert second == first


def _score_signal_cases() -> dict[str, ScoreSignalCase]:
    """Return score signal cases built after the local snapshot helper is available."""
    snapshot = _home_snapshot()
    fallback_snapshot = replace(snapshot, services_target={})
    fan_selector = FailureContext(
        intent=Intent.RESOLVE_SELECTOR,
        requested="fan target",
        domain="fan",
        service="set_percentage",
    )

    return {
        "exact-id": ScoreSignalCase(
            "sensor.living_temp",
            {"kind": "entity", "id": "sensor.living_temp", "name": "Living Temperature", "domain": "sensor"},
            FailureContext(intent=Intent.READ_STATE, requested="sensor.living_temp", domain="sensor"),
            None,
            lambda match: match.exact == 1,
        ),
        "name-token": ScoreSignalCase(
            "bedroom",
            {"kind": "entity", "id": "light.bedroom", "name": "Bedroom Light", "domain": "light"},
            FailureContext(intent=Intent.READ_STATE, requested="bedroom", domain="light"),
            None,
            lambda match: match.token_overlap > 0.0,
        ),
        "abbreviation-temp": ScoreSignalCase(
            "temperature",
            {"kind": "entity", "id": "sensor.living_temp", "object_id": "living_temp", "name": "Living Temp"},
            FailureContext(intent=Intent.READ_STATE, requested="temperature", domain="sensor"),
            None,
            lambda match: match.token_overlap > 0.0,
        ),
        "abbreviation-hum": ScoreSignalCase(
            "humidity",
            {"kind": "entity", "id": "sensor.bedroom_hum", "object_id": "bedroom_hum", "name": "Bedroom Hum"},
            FailureContext(intent=Intent.READ_STATE, requested="humidity", domain="sensor"),
            None,
            lambda match: match.token_overlap > 0.0,
        ),
        "device-class-vocabulary": ScoreSignalCase(
            "temperature",
            {"kind": "entity", "id": "sensor.outdoor", "name": "Outdoor Sensor", "device_class": "temperature"},
            FailureContext(intent=Intent.READ_STATE, requested="temperature", domain="sensor"),
            None,
            lambda match: match.capability == 2,
        ),
        "unit-vocabulary": ScoreSignalCase(
            "c",
            {"kind": "entity", "id": "sensor.outdoor", "name": "Outdoor Sensor", "unit": "°C"},
            FailureContext(intent=Intent.READ_STATE, requested="c", domain="sensor"),
            None,
            lambda match: match.capability >= 1,
        ),
        "area-token": ScoreSignalCase(
            "living",
            {"kind": "entity", "id": "light.living", "name": "Lamp", "area_name": "Living Room"},
            FailureContext(intent=Intent.RESOLVE_SELECTOR, requested="living", domain="light"),
            None,
            lambda match: match.area_floor > 0,
        ),
        "floor-token": ScoreSignalCase(
            "upstairs",
            {"kind": "entity", "id": "light.bedroom", "name": "Lamp", "floor_name": "Upstairs"},
            FailureContext(intent=Intent.RESOLVE_SELECTOR, requested="upstairs", domain="light"),
            None,
            lambda match: match.area_floor > 0,
        ),
        "alias-token": ScoreSignalCase(
            "desk",
            {
                "kind": "entity",
                "id": "light.office_desk",
                "name": "Office Light",
                "domain": "light",
                "aliases": ("desk lamp",),
            },
            FailureContext(intent=Intent.READ_STATE, requested="desk", domain="light"),
            None,
            lambda match: match.token_overlap > 0.0,
        ),
        "service-field-overlap": ScoreSignalCase(
            "set",
            {
                "kind": "service",
                "id": "climate.set_temperature",
                "name": "set temperature",
                "fields": frozenset({"temperature", "hvac_mode"}),
            },
            FailureContext(
                intent=Intent.CALL_SERVICE,
                requested="set",
                domain="climate",
                service_data={"temperature": 21},
            ),
            None,
            lambda match: match.field_overlap > 0,
        ),
        "service-support-target": ScoreSignalCase(
            "fan target",
            {"kind": "entity", "id": "fan.living_fan", "name": "Living Room Fan", "domain": "fan"},
            fan_selector,
            snapshot,
            lambda match: match.service_support > 0,
        ),
        "service-support-domain-fallback": ScoreSignalCase(
            "fan target",
            {"kind": "entity", "id": "fan.living_fan", "name": "Living Room Fan", "domain": "fan"},
            fan_selector,
            fallback_snapshot,
            lambda match: match.service_support == 0,
        ),
    }


def _decision_cases() -> dict[str, RankedCase]:
    """Return ranked inputs for policy decisions."""
    return {
        "exact": [({"id": "sensor.one"}, _match(exact=1, label="id", tiebreak="sensor.one"))],
        "ambiguous": [
            ({"id": "sensor.alpha"}, _match(token_overlap=0.6, label="name", tiebreak="sensor.alpha")),
            ({"id": "sensor.beta"}, _match(token_overlap=0.6, label="name", tiebreak="sensor.beta")),
        ],
        "listing": [({"id": "sensor.one"}, _match(label="id", tiebreak="sensor.one"))],
        "none": [],
    }


def _match(
    *,
    exact: int = 0,
    token_overlap: float = 0.0,
    capability: int = 0,
    area_floor: int = 0,
    service_support: int = 0,
    field_overlap: int = 0,
    tiebreak: str = "",
    label: str = "id",
) -> Match:
    """Build a policy-level match without coupling tests to unrelated score signals."""
    return Match(
        exact,
        token_overlap,
        capability,
        area_floor,
        service_support,
        field_overlap,
        tiebreak,
        label,
    )


def _home_snapshot() -> HomeSnapshot:
    """Build a local home_default-like snapshot without importing eval fixtures."""
    states = {
        entity_id: _state(entity_id, state, name, attrs)
        for entity_id, state, name, attrs in (
            ("light.living", "on", "Living Room Light", {"brightness": 210}),
            ("light.bedroom", "off", "Bedroom Light", {}),
            ("light.office_desk", "on", "Office Desk Light", {"brightness": 120}),
            (
                "sensor.living_temp",
                "25.2",
                "Living Temperature",
                {"device_class": "temperature", "unit_of_measurement": "°C"},
            ),
            (
                "sensor.bedroom_humidity",
                "64",
                "Bedroom Humidity",
                {"device_class": "humidity", "unit_of_measurement": "%"},
            ),
            (
                "sensor.office_temp",
                "22.1",
                "Office Temperature",
                {"device_class": "temperature", "unit_of_measurement": "°C"},
            ),
            ("switch.dehumidifier", "off", "Bedroom Dehumidifier", {}),
            ("fan.living_fan", "off", "Living Room Fan", {"percentage": 0}),
            ("cover.office_blinds", "open", "Office Blinds", {"current_position": 80}),
            ("camera.front_door", "idle", "Front Door Camera", {}),
            ("image.doorbell", "2026-06-29T12:00:00+00:00", "Doorbell Image", {}),
        )
    }
    entities = {
        entity_id: _entity(
            entity_id,
            device_id,
            aliases=("desk lamp",) if entity_id == "light.office_desk" else (),
            hidden_by="integration" if entity_id == "switch.garage_opener" else None,
        )
        for entity_id, device_id in _ENTITY_DEVICES.items()
    }
    devices = {
        "device_living_light": _device("device_living_light", "Living Light Controller", "area_living"),
        "device_living_climate": _device("device_living_climate", "Living Climate Sensor", "area_living"),
        "device_bedroom_lamp": _device("device_bedroom_lamp", "Bedroom Lamp", "area_bedroom"),
        "device_bedroom_climate": _device("device_bedroom_climate", "Bedroom Thermostat", "area_bedroom"),
        "device_dehumidifier": _device("device_dehumidifier", "Bedroom Dehumidifier", "area_bedroom"),
        "device_office_desk": _device("device_office_desk", "Office Desk", "area_office"),
        "device_office_climate": _device("device_office_climate", "Office Climate Sensor", "area_office"),
        "device_living_fan": _device("device_living_fan", "Living Fan", "area_living"),
        "device_office_cover": _device("device_office_cover", "Office Cover", "area_office"),
        "device_front_camera": _device("device_front_camera", "Front Door Camera", "area_living"),
        "device_doorbell_image": _device("device_doorbell_image", "Doorbell Image", "area_office"),
        "device_garage_opener": _device("device_garage_opener", "Garage Opener", "area_living"),
    }
    areas = {
        "area_living": _area("area_living", "Living Room", "floor_main"),
        "area_bedroom": _area("area_bedroom", "Bedroom", "floor_upstairs"),
        "area_office": _area("area_office", "Office", "floor_upstairs"),
        "area_upstairs": _area("area_upstairs", "Upstairs", "floor_upstairs"),
    }
    states = enrich_states(states, entities, devices, areas)
    floors = {
        "floor_main": _floor("floor_main", "Main Floor", 1),
        "floor_upstairs": _floor("floor_upstairs", "Upstairs", 2),
    }
    services_schema = {
        "fan": {"set_percentage": {"fields": [{"name": "percentage"}], "dynamic": False}},
        "light": {
            "turn_on": {"fields": [{"name": "brightness"}], "dynamic": False},
            "turn_off": {"fields": [], "dynamic": False},
        },
        "switch": {"toggle": {"fields": [], "dynamic": False}},
        "cover": {"set_position": {"fields": [{"name": "position"}], "dynamic": False}},
        "climate": {
            "set_temperature": {
                "fields": [{"name": "temperature"}, {"name": "hvac_mode"}],
                "dynamic": False,
            }
        },
    }
    return HomeSnapshot(
        created_at=_CREATED_AT,
        states=states,
        entities=entities,
        devices=devices,
        areas=areas,
        floors=floors,
        config=_config(),
        services={
            "fan": ("set_percentage",),
            "light": ("turn_on", "turn_off"),
            "switch": ("toggle",),
            "cover": ("set_position",),
            "climate": ("set_temperature",),
        },
        services_supports_response={
            "fan": {"set_percentage": SupportsResponse.NONE.value},
            "light": {"turn_on": SupportsResponse.NONE.value, "turn_off": SupportsResponse.NONE.value},
            "switch": {"toggle": SupportsResponse.NONE.value},
            "cover": {"set_position": SupportsResponse.NONE.value},
            "climate": {"set_temperature": SupportsResponse.NONE.value},
        },
        indexes=_indexes(entities, devices, areas, floors),
        labels={"label_security": _label("label_security", "Security")},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema=services_schema,
        services_target={
            "fan": {"set_percentage": {"entity": [{"domain": ["fan"]}]}},
            "light": {"turn_on": {"entity": [{"domain": ["light"]}]}},
            "cover": {"set_position": {"entity": [{"domain": ["cover"]}]}},
            "climate": {"set_temperature": {"entity": [{"domain": ["climate"]}]}},
        },
    )


_ENTITY_DEVICES: Mapping[str, str] = {
    "light.living": "device_living_light",
    "light.bedroom": "device_bedroom_lamp",
    "light.office_desk": "device_office_desk",
    "sensor.living_temp": "device_living_climate",
    "sensor.bedroom_humidity": "device_bedroom_climate",
    "sensor.office_temp": "device_office_climate",
    "switch.dehumidifier": "device_dehumidifier",
    "switch.garage_opener": "device_garage_opener",
    "fan.living_fan": "device_living_fan",
    "cover.office_blinds": "device_office_cover",
    "camera.front_door": "device_front_camera",
    "image.doorbell": "device_doorbell_image",
}


def _many_sensor_snapshot(count: int) -> HomeSnapshot:
    """Build a snapshot with enough same-domain candidates to exercise bounding."""
    snapshot = _home_snapshot()
    states = dict(snapshot.states)
    entities = dict(snapshot.entities)
    devices = dict(snapshot.devices)
    for index in range(count):
        entity_id = f"sensor.extra_{index:02d}"
        device_id = f"device_extra_sensor_{index:02d}"
        states[entity_id] = _state(entity_id, str(index), f"Extra Sensor {index:02d}", {})
        entities[entity_id] = _entity(entity_id, device_id)
        devices[device_id] = _device(device_id, f"Extra Sensor {index:02d}", "area_living")
    states = enrich_states(states, entities, devices, snapshot.areas)
    return replace(
        snapshot,
        states=states,
        entities=entities,
        devices=devices,
        indexes=_indexes(entities, devices, snapshot.areas, snapshot.floors),
    )


def _state(entity_id: str, state: str, name: str, attributes: dict[str, object]) -> SafeState:
    """Build a visible state record."""
    domain, object_id = entity_id.split(".", 1)
    return SafeState(
        entity_id=entity_id,
        domain=domain,
        object_id=object_id,
        name=name,
        state=state,
        attributes={"friendly_name": name, **attributes},
        last_changed=_CREATED_AT,
        last_changed_timestamp=datetime.fromisoformat(_CREATED_AT).timestamp(),
        last_reported=_CREATED_AT,
        last_reported_timestamp=datetime.fromisoformat(_CREATED_AT).timestamp(),
        last_updated=_CREATED_AT,
        last_updated_timestamp=datetime.fromisoformat(_CREATED_AT).timestamp(),
        context=SafeContext(id="ctx", parent_id=None, user_id=None),
    )


def _entity(
    entity_id: str,
    device_id: str,
    *,
    aliases: tuple[str, ...] = (),
    hidden_by: str | None = None,
) -> SafeRegistryEntry:
    """Build a registry entry for a state."""
    domain, _object_id = entity_id.split(".", 1)
    return SafeRegistryEntry(
        entity_id=entity_id,
        domain=domain,
        unique_id=entity_id,
        platform=domain,
        config_entry_id="entry_default",
        device_id=device_id,
        area_id=None,
        name=None,
        original_name=None,
        aliases=aliases,
        labels=(),
        disabled_by=None,
        hidden_by=hidden_by,
        entity_category=None,
        device_class=None,
        original_device_class=None,
        capabilities=None,
        supported_features=0,
        translation_key=None,
        has_entity_name=True,
    )


def _device(device_id: str, name: str, area_id: str) -> SafeDeviceEntry:
    """Build a device registry entry."""
    return SafeDeviceEntry(
        id=device_id,
        name=name,
        name_by_user=None,
        manufacturer="Test",
        model="Fixture",
        model_id=None,
        sw_version=None,
        hw_version=None,
        serial_number=None,
        area_id=area_id,
        labels=(),
        identifiers=(("test", device_id),),
        connections=(),
        configuration_url=None,
        entry_type=None,
        config_entries=("entry_default",),
        via_device_id=None,
        disabled_by=None,
    )


def _area(area_id: str, name: str, floor_id: str) -> SafeAreaEntry:
    """Build an area registry entry."""
    return SafeAreaEntry(
        id=area_id,
        area_id=area_id,
        name=name,
        aliases=(),
        floor_id=floor_id,
        labels=(),
        icon=None,
        picture=None,
        humidity_entity_id=None,
        temperature_entity_id=None,
        created_at=_CREATED_AT,
        modified_at=_CREATED_AT,
    )


def _floor(floor_id: str, name: str, level: int) -> SafeFloorEntry:
    """Build a floor registry entry."""
    return SafeFloorEntry(
        floor_id=floor_id,
        id=floor_id,
        name=name,
        aliases=(),
        level=level,
        icon=None,
        created_at=_CREATED_AT,
        modified_at=_CREATED_AT,
    )


def _label(label_id: str, name: str) -> SafeLabelEntry:
    """Build a label registry entry."""
    return SafeLabelEntry(
        label_id=label_id,
        name=name,
        normalized_name=name.casefold(),
        description=None,
        color=None,
        icon=None,
        created_at=_CREATED_AT,
        modified_at=_CREATED_AT,
    )


def _indexes(
    entities: Mapping[str, SafeRegistryEntry],
    devices: Mapping[str, SafeDeviceEntry],
    areas: Mapping[str, SafeAreaEntry],
    floors: Mapping[str, SafeFloorEntry],
) -> SnapshotIndexes:
    """Build sorted indexes using the effective-area rule."""
    by_device: dict[str, list[str]] = {}
    by_area: dict[str, list[str]] = {}
    for entity in entities.values():
        if entity.device_id is not None:
            by_device.setdefault(entity.device_id, []).append(entity.entity_id)
            area_id = devices[entity.device_id].area_id
            if area_id is not None:
                by_area.setdefault(area_id, []).append(entity.entity_id)
    return SnapshotIndexes(
        entity_ids_by_device_id={key: tuple(sorted(value)) for key, value in by_device.items()},
        entity_ids_by_area_id={key: tuple(sorted(value)) for key, value in by_area.items()},
        device_ids_by_area_id={area.area_id: () for area in areas.values()},
        entity_ids_by_config_entry_id={},
        entity_ids_by_label={},
        device_ids_by_label={},
        area_ids_by_floor_id={
            floor.floor_id: tuple(sorted(area.area_id for area in areas.values() if area.floor_id == floor.floor_id))
            for floor in floors.values()
        },
    )


def _config() -> SafeConfig:
    """Build a minimal config record."""
    return SafeConfig(
        location_name="Test Home",
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
