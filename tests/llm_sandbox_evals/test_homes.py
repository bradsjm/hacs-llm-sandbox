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


def test_full_home_preserves_288_entity_inventory() -> None:
    snapshot = get_home("home_full").snapshot()

    assert len(snapshot.states) == 288
    assert len(snapshot.entities) == 288


@pytest.mark.parametrize("name", ["home_default", "home_large", "home_real"])
def test_removed_home_names_are_rejected(name: str) -> None:
    with pytest.raises(KeyError):
        get_home(name)
