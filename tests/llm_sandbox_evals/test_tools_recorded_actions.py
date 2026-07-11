from llm_sandbox_evals.tools import _for_scoring
import pytest


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        pytest.param(
            {"service": "light.turn_on", "target": {"entity_id": ["light.kitchen"]}, "status": "ok"},
            {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": ["light.kitchen"]},
                "status": "ok",
            },
            id="compact-production-action",
        ),
        pytest.param(
            {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": ["light.kitchen"]},
                "status": "ok",
            },
            {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": ["light.kitchen"]},
                "status": "ok",
            },
            id="already-split-invoker-action",
        ),
    ],
)
def test_for_scoring_normalizes_recorded_actions(
    action: dict[str, object],
    expected: dict[str, object],
) -> None:
    normalized = _for_scoring(action)

    assert normalized == expected
    assert action["target"] == expected["target"]
    assert action["status"] == expected["status"]
