from typing import cast

from homeassistant.core import SupportsResponse
from llm_sandbox_evals.homes import get_home
import pytest


def test_minimal_home_has_only_the_action_case_surface() -> None:
    fixture = get_home("home_minimal")
    snapshot = fixture.snapshot()

    assert set(snapshot.states) == {"light.bedroom", "light.living"}
    assert set(snapshot.entities) == {"light.bedroom", "light.living"}
    assert set(snapshot.devices) == {"device_bedroom", "device_living"}
    assert set(snapshot.areas) == {"area_bedroom", "area_living"}
    assert snapshot.services == {"light": ("turn_off", "turn_on")}
    assert snapshot.indexes.entity_ids_by_device_id == {
        "device_bedroom": ("light.bedroom",),
        "device_living": ("light.living",),
    }
    assert snapshot.indexes.entity_ids_by_area_id == {
        "area_bedroom": ("light.bedroom",),
        "area_living": ("light.living",),
    }
    assert snapshot.indexes.entity_ids_by_config_entry_id == {"entry_minimal": ("light.bedroom", "light.living")}
    assert snapshot.config.location_name == "Home"
    assert fixture.recorder() == {"history": {}, "statistics": {}, "logbook": {}}


def test_cover_home_exposes_exact_frozen_capability_surface() -> None:
    fixture = get_home("home_cover")
    snapshot = fixture.snapshot()

    assert snapshot.created_at == "2026-06-29T12:00:00+00:00"
    assert {entity_id: state.state for entity_id, state in snapshot.states.items()} == {
        "cover.office_blinds": "closed",
        "cover.bedroom_shade": "open",
    }
    assert {
        entity_id: (
            state.name,
            state.attributes["current_position"],
            state.attributes["device_class"],
            state.attributes["supported_features"],
            state.area_id,
            state.device_id,
        )
        for entity_id, state in snapshot.states.items()
    } == {
        "cover.office_blinds": ("Office Blinds", 0, "blind", 7, "area_office", "device_office_blinds"),
        "cover.bedroom_shade": ("Bedroom Shade", 100, "shade", 7, "area_bedroom", "device_bedroom_shade"),
    }
    assert {
        entity_id: (entity.device_id, entity.area_id, entity.config_entry_id, entity.supported_features)
        for entity_id, entity in snapshot.entities.items()
    } == {
        "cover.office_blinds": ("device_office_blinds", None, "entry_cover", 7),
        "cover.bedroom_shade": ("device_bedroom_shade", None, "entry_cover", 7),
    }
    assert {device_id: (device.name, device.area_id) for device_id, device in snapshot.devices.items()} == {
        "device_office_blinds": ("Office Blinds", "area_office"),
        "device_bedroom_shade": ("Bedroom Shade", "area_bedroom"),
    }
    assert {area_id: area.name for area_id, area in snapshot.areas.items()} == {
        "area_office": "Office",
        "area_bedroom": "Bedroom",
    }
    assert snapshot.indexes.entity_ids_by_device_id == {
        "device_office_blinds": ("cover.office_blinds",),
        "device_bedroom_shade": ("cover.bedroom_shade",),
    }
    assert snapshot.indexes.entity_ids_by_area_id == {
        "area_office": ("cover.office_blinds",),
        "area_bedroom": ("cover.bedroom_shade",),
    }
    assert snapshot.indexes.device_ids_by_area_id == {
        "area_office": ("device_office_blinds",),
        "area_bedroom": ("device_bedroom_shade",),
    }
    assert snapshot.indexes.entity_ids_by_config_entry_id == {
        "entry_cover": ("cover.bedroom_shade", "cover.office_blinds")
    }
    assert snapshot.services == {"cover": ("close_cover", "open_cover", "set_cover_position")}
    assert snapshot.services_supports_response == {
        "cover": {
            "close_cover": SupportsResponse.NONE.value,
            "open_cover": SupportsResponse.NONE.value,
            "set_cover_position": SupportsResponse.NONE.value,
        }
    }
    assert snapshot.services_schema == {}
    assert fixture.recorder() == {"history": {}, "statistics": {}, "logbook": {}}


def test_full_home_preserves_312_entity_inventory() -> None:
    snapshot = get_home("home_full").snapshot()

    assert len(snapshot.states) == 312
    assert len(snapshot.entities) == 312


def test_full_home_climate_inputs_select_conditional_branches() -> None:
    snapshot = get_home("home_full").snapshot()

    assert {
        entity_id: (
            snapshot.states[entity_id].state,
            snapshot.states[entity_id].attributes["current_temperature"],
            snapshot.states[entity_id].attributes["temperature"],
        )
        for entity_id in ("climate.workshop", "climate.storage_room")
    } == {
        "climate.workshop": ("heat", 21.0, 20.0),
        "climate.storage_room": ("heat", 21.0, 20.0),
    }


def test_full_home_exposes_balcony_statistics() -> None:
    fixture = get_home("home_full")
    snapshot = fixture.snapshot()
    statistics = cast(dict[str, list[dict[str, object]]], fixture.recorder()["statistics"])
    rows = statistics["sensor.balcony_power"]

    assert snapshot.states["sensor.balcony_power"].attributes["state_class"] == "measurement"
    assert [row["mean"] for row in rows] == [38.0, 42.0]


def test_full_home_selector_indexes_match_eval_cases() -> None:
    snapshot = get_home("home_full").snapshot()
    indexes = snapshot.indexes
    living_room_evening_entities = set(indexes.entity_ids_by_area_id["living_room"]) & set(
        indexes.entity_ids_by_label["label_evening"]
    )
    second_floor_climate_entities = {
        f"climate.{area_id}" for area_id in indexes.area_ids_by_floor_id["floor_second"]
    }

    assert living_room_evening_entities == {
        "light.living_room_ceiling",
        "light.living_room_accent",
    }
    assert len(second_floor_climate_entities) == 9
    assert second_floor_climate_entities <= set(indexes.entity_ids_by_label["label_climate"])

def test_full_home_basement_ceiling_lights_start_off() -> None:
    snapshot = get_home("home_full").snapshot()
    basement_ceiling_ids = {
        "light.utility_room_ceiling",
        "light.storage_room_ceiling",
        "light.workshop_ceiling",
        "light.wine_cellar_ceiling",
        "light.home_gym_ceiling",
        "light.media_room_ceiling",
        "light.laundry_room_ceiling",
        "light.basement_bathroom_ceiling",
        "light.playroom_ceiling",
        "light.wine_tasting_room_ceiling",
        "light.basement_hallway_ceiling",
        "light.server_room_ceiling",
    }

    assert {snapshot.states[entity_id].state for entity_id in basement_ceiling_ids} == {"off"}
    assert snapshot.states["light.utility_room_accent"].state == "on"
    assert snapshot.states["switch.utility_room_outlet"].state == "off"


@pytest.mark.parametrize(
    ("entity_id", "expected_last_changed"),
    [
        ("light.living_room_ceiling", "2026-06-29T11:45:00+00:00"),
        ("light.living_room_accent", "2026-06-29T11:50:00+00:00"),
    ],
)
def test_full_home_lights_match_terminal_history(entity_id: str, expected_last_changed: str) -> None:
    fixture = get_home("home_full")
    snapshot = fixture.snapshot()
    history = cast(dict[str, object], fixture.recorder()["history"])
    rows = cast(list[dict[str, object]], history[entity_id])
    terminal_row = rows[-1]
    state = snapshot.states[entity_id]

    assert state.state == "on"
    assert state.last_changed == expected_last_changed
    assert state.last_changed == cast(str, terminal_row["last_changed"])
    assert state.state == cast(str, terminal_row["state"])


@pytest.mark.parametrize(
    ("entity_id", "expected_last_changed"),
    [("switch.hallway_outlet", "2026-06-29T09:00:00+00:00")],
)
def test_full_home_switch_matches_terminal_logbook(entity_id: str, expected_last_changed: str) -> None:
    fixture = get_home("home_full")
    snapshot = fixture.snapshot()
    logbook = cast(dict[str, object], fixture.recorder()["logbook"])
    events = cast(list[dict[str, object]], logbook[entity_id])
    terminal_event = events[-1]
    state = snapshot.states[entity_id]

    assert state.state == "off"
    assert state.last_changed == expected_last_changed
    assert state.last_changed == cast(str, terminal_event["when"])


@pytest.mark.parametrize("name", ["home_default", "home_large", "home_real"])
def test_removed_home_names_are_rejected(name: str) -> None:
    with pytest.raises(KeyError):
        get_home(name)
