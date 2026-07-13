import json
from pathlib import Path


def test_authoring_schema_contains_only_the_minimal_case_contract() -> None:
    schema_path = Path("llm_sandbox_evals/data/cases_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    case = schema["$defs"]["case"]

    assert set(case["properties"]) == {"id", "home", "user_request", "required_actions"}
    assert case["required"] == ["id", "home", "user_request", "required_actions"]
    assert set(schema["$defs"]["action"]["properties"]) == {
        "domain",
        "service",
        "target_entity_ids",
        "service_data",
    }
