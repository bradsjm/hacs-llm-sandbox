from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.agent_runner import build_agent
from llm_sandbox_evals.config import EvalOutputMode
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import EvalRuntime, build_eval_runtime
from llm_sandbox_evals.schema import (
    ActionAnswer,
    AnswerShape,
    BlockedOutcome,
    CaseContext,
    CollectionClaim,
    EvalCase,
    Expected,
    ExpectedConclusion,
    ListAnswer,
    PromptCandidate,
    ReadAnswer,
    ValueClaim,
)
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
import pytest


def _runtime(user_request: str, expected: Expected) -> EvalRuntime:
    """Build the smallest fixture-backed runtime needed to run the real stub model."""
    case = EvalCase(
        id="output-mode",
        category="state",
        home="home_default",
        user_request=user_request,
        actions_enabled=False,
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


@pytest.mark.parametrize(
    "output_mode",
    [
        pytest.param("tool", id="tool"),
        pytest.param("json-schema", id="native"),
    ],
)
@pytest.mark.parametrize(
    ("user_request", "expected", "answer_type"),
    [
        pytest.param(
            "Return a result.",
            Expected(blocked_outcome=BlockedOutcome()),
            ActionAnswer,
            id="action",
        ),
        pytest.param(
            "current living room temperature",
            Expected(
                conclusions=(
                    ExpectedConclusion(
                        claim=ValueClaim(
                            subject_kind="entity",
                            subject_id="sensor.living_temp",
                            field="state",
                            value="21.5",
                        ),
                        assertion="equals",
                    ),
                )
            ),
            ReadAnswer,
            id="read",
        ),
        pytest.param(
            "List the light entities I can control.",
            Expected(
                conclusions=(
                    ExpectedConclusion(
                        claim=CollectionClaim(
                            collection="entity_ids",
                            filter_kind="domain",
                            filter_value="light",
                            items=["light.bedroom"],
                        ),
                        assertion="contains_items",
                    ),
                )
            ),
            ListAnswer,
            id="list",
        ),
    ],
)
async def test_real_stub_parses_case_selected_shape_in_each_output_mode(
    user_request: str,
    expected: Expected,
    answer_type: type[AnswerShape],
    output_mode: EvalOutputMode,
) -> None:
    runtime = _runtime(user_request, expected)

    result = await build_agent(runtime, "stub", output_mode).run(runtime.case.user_request, deps=runtime)

    assert type(result.output) is answer_type
