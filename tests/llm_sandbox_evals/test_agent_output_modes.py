from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.agent_runner import _latest_automation_run_value, build_agent
from llm_sandbox_evals.config import EvalOutputMode
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import EvalRuntime, build_eval_runtime
from llm_sandbox_evals.schema import (
    ActionAnswer,
    AggregateAnswer,
    AggregateExpectation,
    AnswerShape,
    CaseContext,
    EntityAnswer,
    EntityCollectionAnswer,
    EntityCollectionExpectation,
    EntityExpectation,
    EntityRelationAnswer,
    EntityRelationExpectation,
    EvalCase,
    Expected,
    ExpectedAction,
    NoDataAnswer,
    NoDataExpectation,
    PromptCandidate,
)
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
from pydantic_ai.messages import ModelRequest, ToolReturnPart
import pytest


def _runtime(user_request: str, expected: Expected, home: str = "home_default") -> EvalRuntime:
    case = EvalCase(
        id="output-mode",
        category="state",
        home=home,
        user_request=user_request,
        actions_enabled=True,
        llm_context=CaseContext(),
        expected=expected,
    )
    candidate = PromptCandidate("test", "test prompt", "execute", "history", "statistics", "logbook", "automation")
    fixture = get_home(case.home)
    return build_eval_runtime(
        case,
        candidate,
        resolve_profile("balanced"),
        apply_scope(fixture.snapshot(), EVAL_SCOPE),
        fixture,
    )


@pytest.mark.parametrize("output_mode", [pytest.param("tool", id="tool"), pytest.param("json-schema", id="native")])
@pytest.mark.parametrize(
    ("user_request", "expected", "answer_type", "home"),
    [
        pytest.param(
            "current living room temperature",
            Expected(
                expectation=EntityExpectation(
                    source="states", entity_id="sensor.living_temp", input_field="state", value="25.2"
                )
            ),
            EntityAnswer,
            "home_default",
            id="entity",
        ),
        pytest.param(
            "List the light entities I can control.",
            Expected(
                expectation=EntityCollectionExpectation(
                    entity_ids=["light.bedroom", "light.living", "light.office_desk"],
                    filter_kind="domain",
                    filter_value="light",
                )
            ),
            EntityCollectionAnswer,
            "home_default",
            id="collection",
        ),
        pytest.param(
            "What was the highest recorded outside temperature value over the last day?",
            Expected(
                expectation=AggregateExpectation(
                    source="history",
                    operator="maximum",
                    subject_ids=["sensor.tempest_temperature"],
                    input_field="state",
                    input_value="state",
                    value=78.8,
                    unit="°F",
                )
            ),
            AggregateAnswer,
            "home_real",
            id="aggregate",
        ),
        pytest.param(
            "Which of my evening automations controls the living room light?",
            Expected(
                expectation=EntityRelationExpectation(
                    relation="automation_target", entity_id="automation.living_scene_4f7a", related_id="light.living"
                )
            ),
            EntityRelationAnswer,
            "home_default",
            id="relation",
        ),
        pytest.param(
            "office power statistics over the last day",
            Expected(expectation=NoDataExpectation(source="statistics", scope_entity_ids=["sensor.office_power"])),
            NoDataAnswer,
            "home_default",
            id="no-data",
        ),
        pytest.param(
            "Turn off the living room light.",
            Expected(actions=(ExpectedAction("light", "turn_off", ("light.living",)),)),
            ActionAnswer,
            "home_default",
            id="action",
        ),
    ],
)
async def test_real_stub_parses_each_concrete_shape_in_each_output_mode(
    user_request: str,
    expected: Expected,
    answer_type: type[AnswerShape],
    home: str,
    output_mode: EvalOutputMode,
) -> None:
    runtime = _runtime(user_request, expected, home)

    result = await build_agent(runtime, "stub", output_mode).run(runtime.case.user_request, deps=runtime)

    assert type(result.output) is answer_type


@pytest.mark.parametrize("output_mode", [pytest.param("tool", id="tool"), pytest.param("json-schema", id="native")])
async def test_automation_run_stub_routes_and_selects_latest_outcome(output_mode: EvalOutputMode) -> None:
    expected = Expected(
        expectation=EntityExpectation(
            source="automation",
            entity_id="automation.living_scene_4f7a",
            input_field="run",
            value="triggered",
        )
    )
    runtime = _runtime(
        "What was the outcome of the most recent Evening Living Room Lights automation run?",
        expected,
    )

    result = await build_agent(runtime, "stub", output_mode).run(runtime.case.user_request, deps=runtime)

    assert isinstance(result.output, EntityAnswer)
    assert result.output.entity_id == "automation.living_scene_4f7a"
    assert result.output.value == "triggered"


def test_automation_run_stub_orders_outcomes_by_timestamp() -> None:
    entity_id = "automation.living_scene_4f7a"
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "get_automation",
                    {
                        "automations": [
                            {
                                "entity_id": entity_id,
                                "runs": [
                                    {"when": "2026-06-29T12:00:00+00:00", "message": "latest"},
                                    {"when": "2026-06-29T08:00:00+00:00", "message": "earlier"},
                                ],
                            }
                        ]
                    },
                )
            ]
        )
    ]

    assert _latest_automation_run_value(messages, entity_id) == "latest"
