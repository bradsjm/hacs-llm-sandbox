from collections import Counter
import json
from pathlib import Path
from typing import Any, cast

from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.schema import (
    ActionAnswer,
    AggregateAnswer,
    AggregateExpectation,
    EntityAnswer,
    EntityCollectionAnswer,
    EntityExpectation,
    EntityRelationAnswer,
    EntityRelationExpectation,
    NoDataAnswer,
    NoDataExpectation,
    select_answer_shape,
)
from pydantic import ValidationError
import pytest
import yaml

_DATA_DIR = Path(__file__).parents[2] / "llm_sandbox_evals" / "data"
_CATEGORY_COUNTS = {
    "state": 4,
    "registry": 3,
    "history": 2,
    "statistics": 1,
    "logbook": 2,
    "automation": 3,
    "action": 5,
    "safety": 2,
    "system": 2,
}
_SHAPE_COUNTS = {
    EntityAnswer: 6,
    EntityCollectionAnswer: 4,
    AggregateAnswer: 4,
    EntityRelationAnswer: 3,
    NoDataAnswer: 2,
    ActionAnswer: 5,
}
_REMOVED_V3_KEYS = {"claim", "conclusions", "assertion", "findings", "items", "success"}


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


def test_first_tranche_has_approved_shape_and_category_distribution() -> None:
    cases = _cases()
    shapes = Counter(select_answer_shape(case.expected) for case in CASES)

    assert len(cases) == 24
    assert Counter(case["category"] for case in cases) == _CATEGORY_COUNTS
    assert shapes == _SHAPE_COUNTS


def test_dataset_is_oracle_v3_with_one_internal_expectation_per_case() -> None:
    dataset = _dataset()

    assert all(case["oracle_version"] == 3 for case in _cases())
    assert all(set(case["expected"]) == {"expectation", "actions", "blocked_outcome"} for case in _cases())
    assert not (_REMOVED_V3_KEYS & set(_walk(dataset)))


def test_first_tranche_excludes_deferred_output_categories_and_relations() -> None:
    dataset_text = (_DATA_DIR / "cases.yaml").read_text(encoding="utf-8")
    expectations = [case.expected.expectation for case in CASES if case.expected.expectation is not None]

    assert "device_area" not in dataset_text
    assert "area_floor" not in dataset_text
    assert "first_seen" not in dataset_text
    assert "last_seen" not in dataset_text
    assert "list and count" not in dataset_text.lower()
    assert all(
        not isinstance(expectation.value, str) or "T" not in expectation.value
        for expectation in expectations
        if isinstance(expectation, (EntityExpectation, AggregateExpectation))
    )


def test_automation_target_uses_automation_id_in_entity_id() -> None:
    relations = [
        case.expected.expectation for case in CASES if isinstance(case.expected.expectation, EntityRelationExpectation)
    ]
    automation_relations = [
        expectation for expectation in relations if getattr(expectation, "relation", None) == "automation_target"
    ]

    assert len(automation_relations) == 2
    assert all(expectation.entity_id == "automation.living_scene_4f7a" for expectation in automation_relations)


def test_conditional_actions_have_one_entity_or_aggregate_expectation() -> None:
    conditional = [case.expected for case in CASES if case.expected.expectation is not None and case.expected.actions]

    assert len(conditional) == 1
    assert isinstance(conditional[0].expectation, EntityExpectation | AggregateExpectation)


def test_schema_sidecar_is_closed_and_keeps_nine_categories() -> None:
    schema = json.loads((_DATA_DIR / "cases_schema.json").read_text(encoding="utf-8"))
    category = schema["$defs"]["case"]["properties"]["category"]

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["expected"]["additionalProperties"] is False
    assert set(category["enum"]) == set(_CATEGORY_COUNTS)
    assert schema["$defs"]["case"]["properties"]["oracle_version"]["const"] == 3
    assert "statistics" not in schema["$defs"]["entity"]["properties"]["source"]["enum"]
    assert "automation" not in schema["$defs"]["no_data"]["properties"]["source"]["enum"]


@pytest.mark.parametrize(
    ("expectation_type", "payload"),
    [
        pytest.param(
            EntityExpectation,
            {
                "source": "statistics",
                "entity_id": "sensor.power",
                "input_field": "value",
                "value": 1,
            },
            id="entity-statistics",
        ),
        pytest.param(
            NoDataExpectation,
            {"source": "automation", "scope_entity_ids": ["automation.example"]},
            id="no-data-automation",
        ),
    ],
)
def test_unsupported_first_tranche_expectation_sources_are_rejected(
    expectation_type: type[EntityExpectation | NoDataExpectation], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        expectation_type.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"source": "states", "entity_id": "sensor.temp", "input_field": "state", "value": "21"},
            id="states-state",
        ),
        pytest.param(
            {"source": "states", "entity_id": "sensor.temp", "input_field": "name", "value": "Temperature"},
            id="states-name",
        ),
        pytest.param(
            {
                "source": "states",
                "entity_id": "sensor.temp",
                "input_field": "attribute",
                "input_value": "unit_of_measurement",
                "value": "°C",
            },
            id="states-attribute",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "state", "value": "21"},
            id="history-state",
        ),
        pytest.param(
            {
                "source": "history",
                "entity_id": "sensor.temp",
                "input_field": "attribute",
                "input_value": "unit_of_measurement",
                "value": "°C",
            },
            id="history-attribute",
        ),
        pytest.param(
            {"source": "logbook", "entity_id": "light.living", "input_field": "message", "value": "turned on"},
            id="logbook-message",
        ),
        pytest.param(
            {"source": "automation", "entity_id": "automation.scene", "input_field": "enabled", "value": True},
            id="automation-enabled",
        ),
        pytest.param(
            {"source": "automation", "entity_id": "automation.scene", "input_field": "name", "value": "Scene"},
            id="automation-name",
        ),
        pytest.param(
            {"source": "automation", "entity_id": "automation.scene", "input_field": "value", "value": "sunset"},
            id="automation-value",
        ),
        pytest.param(
            {"source": "automation", "entity_id": "automation.scene", "input_field": "run", "value": "triggered"},
            id="automation-run",
        ),
    ],
)
def test_entity_expectation_accepts_source_specific_fields(payload: dict[str, object]) -> None:
    expectation = EntityExpectation.model_validate(payload)

    assert expectation.source == payload["source"]
    assert expectation.input_field == payload["input_field"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"source": "logbook", "entity_id": "light.living", "input_field": "state", "value": "turned on"},
            id="logbook-state-false-pass",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "name", "value": "Temperature"},
            id="history-name",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "enabled", "value": True},
            id="history-enabled",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "message", "value": "changed"},
            id="history-message",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "value", "value": 21},
            id="history-value",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "run", "value": "triggered"},
            id="history-run",
        ),
        pytest.param(
            {"source": "states", "entity_id": "sensor.temp", "input_field": "message", "value": "changed"},
            id="states-message",
        ),
        pytest.param(
            {"source": "automation", "entity_id": "automation.scene", "input_field": "state", "value": "on"},
            id="automation-state",
        ),
        pytest.param(
            {"source": "history", "entity_id": "sensor.temp", "input_field": "attribute", "value": "°C"},
            id="attribute-name-missing",
        ),
        pytest.param(
            {
                "source": "logbook",
                "entity_id": "light.living",
                "input_field": "message",
                "input_value": "ignored",
                "value": "turned on",
            },
            id="non-attribute-input-value",
        ),
    ],
)
def test_entity_expectation_rejects_invalid_source_field_combinations(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EntityExpectation.model_validate(payload)
