from collections import Counter
import json
from math import isclose
from pathlib import Path
from typing import Any, cast

from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from llm_sandbox_evals.homes import get_home
import pytest
import yaml

_DATA_DIR = Path(__file__).parents[2] / "llm_sandbox_evals" / "data"
_CATEGORY_COUNTS = {
    "action": 26,
    "state": 18,
    "history": 12,
    "registry": 8,
    "statistics": 5,
    "logbook": 4,
    "automation": 3,
    "safety": 2,
    "system": 2,
}
_LEGACY_KEYS = {"provenance_values", "tool_result_checks", "tool_call_par", "max_attempts"}


def _dataset() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load((_DATA_DIR / "cases.yaml").read_text(encoding="utf-8")))


def _cases() -> list[dict[str, Any]]:
    return [case["inputs"] for case in _dataset()["cases"]]


def _walk(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        keys.extend(str(key) for key in value)
        for child in value.values():
            keys.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk(child))
    return keys


def test_dataset_preserves_exact_case_taxonomy_and_counts() -> None:
    cases = _cases()

    assert len(cases) == 80
    assert Counter(case["category"] for case in cases) == _CATEGORY_COUNTS
    assert {case["category"] for case in cases} == set(_CATEGORY_COUNTS)


def test_dataset_is_v2_only_and_has_no_legacy_contract_fields() -> None:
    dataset = _dataset()
    cases = _cases()

    assert all(case["oracle_version"] == 2 for case in cases)
    assert not (_LEGACY_KEYS & set(_walk(dataset)))
    assert all(set(case["expected"]) == {"conclusions", "actions", "blocked_outcome"} for case in cases)


def test_every_read_case_has_typed_conclusions_and_action_cases_have_one_ledger_form() -> None:
    for case in _cases():
        expected = case["expected"]
        has_block = expected["blocked_outcome"] is not None
        has_actions = bool(expected["actions"])
        if case["category"] != "action":
            assert expected["conclusions"]
        assert has_block + has_actions + bool(expected["conclusions"]) >= 1
        assert all(set(conclusion) == {"claim", "assertion", "tolerance"} for conclusion in expected["conclusions"])


def test_no_data_cases_use_resolved_entity_scopes() -> None:
    cases = _cases()

    no_data = {
        case["id"]: case["expected"]["conclusions"][0]["claim"]
        for case in cases
        if case["expected"]["conclusions"] and case["expected"]["conclusions"][0]["claim"]["kind"] == "no_data"
    }
    assert {
        "recovery_statistic_no_data": no_data["recovery_statistic_no_data"],
        "honesty_empty_logbook": no_data["honesty_empty_logbook"],
    } == {
        "recovery_statistic_no_data": {
            "kind": "no_data",
            "source": "statistics",
            "scope_entity_ids": ["sensor.office_power"],
        },
        "honesty_empty_logbook": {
            "kind": "no_data",
            "source": "logbook",
            "scope_entity_ids": ["sensor.bedroom_humidity"],
        },
    }
    for case in cases:
        for conclusion in case["expected"]["conclusions"]:
            if conclusion["claim"]["kind"] != "no_data":
                continue
            claim = conclusion["claim"]
            scope = claim["scope_entity_ids"]
            assert conclusion["assertion"] == "empty"
            assert conclusion["tolerance"] is None
            assert scope
            assert scope == sorted(set(scope))


def test_default_fixture_has_distinct_multi_sensor_populations() -> None:
    snapshot = get_home("home_default").snapshot()
    temperature_ids = {
        entity.entity_id for entity in snapshot.entities.values() if entity.device_class == "temperature"
    }
    humidity_ids = {entity.entity_id for entity in snapshot.entities.values() if entity.device_class == "humidity"}

    assert temperature_ids >= {"sensor.living_temp", "sensor.office_temp"}
    assert humidity_ids >= {"sensor.bedroom_humidity", "sensor.living_humidity", "sensor.office_humidity"}
    assert snapshot.entities["sensor.living_temp"].area_id != snapshot.entities["sensor.office_temp"].area_id
    assert snapshot.entities["sensor.bedroom_humidity"].area_id != snapshot.entities["sensor.living_humidity"].area_id
    assert snapshot.entities["sensor.office_humidity"].area_id == "area_office"


def test_reviewed_case_oracles_reference_reachable_fixture_facts() -> None:
    cases = {case["id"]: case for case in _cases()}
    snapshot = get_home("home_default").snapshot()

    blocked = cases["blocked_hidden_garage_opener"]["expected"]["blocked_outcome"]
    assert blocked["error_keys"] == ["service_not_found"]
    assert blocked["actions"] == [
        {"domain": "switch", "service": "turn_on", "target_entity_ids": ["switch.garage_opener"]}
    ]
    assert "turn_on" not in snapshot.services["switch"]
    assert snapshot.entities["switch.garage_opener"].hidden_by == "integration"

    issue = next(issue for issue in snapshot.issues if issue.issue_id == "living_temp_device_automation_invalid")
    repair_claim = cases["depth_active_repairs"]["expected"]["conclusions"][0]["claim"]
    assert (repair_claim["field"], repair_claim["value"]) == ("status", issue.active)

    state_claims = [
        conclusion["claim"] for conclusion in cases["multi_state_last_changed_living_light"]["expected"]["conclusions"]
    ]
    assert {claim["field"] for claim in state_claims} == {"state", "attribute"}
    assert {claim.get("attribute_name") for claim in state_claims} == {None, "brightness"}
    assert snapshot.states["light.living"].attributes["brightness"] == 210


def test_collection_oracles_match_fixture_entity_populations() -> None:
    for case in _cases():
        for conclusion in case["expected"]["conclusions"]:
            claim = conclusion["claim"]
            if claim["kind"] != "collection" or claim["collection"] != "entity_ids":
                continue
            snapshot = get_home(case["home"]).snapshot()
            actual = {
                entity_id
                for entity_id, entity in snapshot.entities.items()
                if _collection_matches(snapshot, entity_id, claim["filter_kind"], claim["filter_value"])
            }
            expected = set(claim["items"])
            assert expected <= actual, case["id"]
            if conclusion["assertion"] == "exact_items":
                assert expected == actual, case["id"]


def test_aggregate_oracles_match_fixture_recorder_and_state_facts() -> None:
    supported = {"count", "mean", "minimum", "maximum", "sum"}
    for case in _cases():
        recorder = get_home(case["home"]).recorder()
        snapshot = get_home(case["home"]).snapshot()
        for conclusion in case["expected"]["conclusions"]:
            claim = conclusion["claim"]
            if claim["kind"] != "aggregate":
                continue
            assert claim["operator"] in supported, case["id"]
            values = _aggregate_inputs(snapshot, recorder, claim)
            actual = _aggregate_value(values, claim["operator"])
            expected = claim["value"]
            assert actual is not None, case["id"]
            if conclusion["assertion"] == "approximate":
                assert isclose(float(actual), float(expected), abs_tol=conclusion["tolerance"]), case["id"]
            else:
                assert actual == expected, case["id"]


def _collection_matches(snapshot: HomeSnapshot, entity_id: str, filter_kind: str, filter_value: str | None) -> bool:
    entity = snapshot.entities[entity_id]
    if filter_kind == "all":
        return True
    if filter_kind == "domain":
        return entity.domain == filter_value
    if filter_kind == "state":
        return snapshot.states[entity_id].state == filter_value
    if filter_kind == "label":
        return filter_value in entity.labels
    if filter_kind == "area":
        effective_area = entity.area_id
        if effective_area is None and entity.device_id is not None:
            effective_area = snapshot.devices[entity.device_id].area_id
        return effective_area == filter_value
    return False


def _aggregate_inputs(snapshot: HomeSnapshot, recorder: dict[str, object], claim: dict[str, object]) -> list[object]:
    source = claim["source"]
    subject_ids = cast(list[str], claim["subject_ids"])
    if source == "states":
        return [snapshot.states[entity_id].state for entity_id in subject_ids]
    if source == "history":
        history = cast(dict[str, list[dict[str, object]]], recorder["history"])
        return [row["state"] for entity_id in subject_ids for row in history[entity_id]]
    if source == "statistics":
        statistics = cast(dict[str, list[dict[str, object]]], recorder["statistics"])
        field = cast(str, claim["input_field"])
        return [row[field] for entity_id in subject_ids for row in statistics[entity_id]]
    raise AssertionError(f"unsupported fixture aggregate source: {source}")


def _aggregate_value(values: list[object], operator: str) -> int | float | None:
    if operator == "count":
        return len(values)
    numeric = [float(value) for value in values if isinstance(value, (int, float, str)) and _is_number(value)]
    if not numeric:
        return None
    if operator == "mean":
        return sum(numeric) / len(numeric)
    if operator == "minimum":
        return min(numeric)
    if operator == "maximum":
        return max(numeric)
    if operator == "sum":
        return sum(numeric)
    raise AssertionError(f"unsupported aggregate operator: {operator}")


def _is_number(value: int | float | str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    ("case_id", "expected_actions"),
    [
        pytest.param("complex_hot_living_turn_on_fan", True, id="temperature-true"),
        pytest.param("real_conditional_close_blinds", False, id="temperature-false"),
        pytest.param("action_high_consequence_clear_intent", True, id="humidity-true"),
        pytest.param("complex_humidity_dehumidifier", False, id="humidity-false"),
        pytest.param("multi_history_then_living_fan", True, id="history-true"),
        pytest.param("multi_history_then_living_light_off", False, id="history-false"),
        pytest.param("multi_logbook_then_living_light_off", True, id="logbook-true"),
        pytest.param("action_floor_target", False, id="logbook-false"),
    ],
)
def test_conditional_action_population_is_explicit(case_id: str, expected_actions: bool) -> None:
    case = next(case for case in _cases() if case["id"] == case_id)

    assert bool(case["expected"]["actions"]) is expected_actions
    assert case["expected"]["conclusions"]


def test_schema_sidecar_is_closed_and_keeps_category_contract() -> None:
    schema = json.loads((_DATA_DIR / "cases_schema.json").read_text(encoding="utf-8"))
    category = schema["$defs"]["case"]["properties"]["category"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["expected"]["additionalProperties"] is False
    assert set(category["enum"]) == set(_CATEGORY_COUNTS)
    assert category["maxLength"] == 13
    assert "oracle_version" in schema["$defs"]["case"]["required"]
    assert not _LEGACY_KEYS & set(_walk(schema))
