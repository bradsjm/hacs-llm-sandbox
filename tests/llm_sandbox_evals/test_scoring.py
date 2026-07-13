from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import run_case
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.schema import (
    ActionComparison,
    ActionResult,
    EvalCase,
    ObservedAction,
    RequiredAction,
)
from llm_sandbox_evals.scoring import evaluate_case
import pytest


def _action(
    domain: str,
    service: str,
    entity_id: str,
    service_data: dict[str, object] | None = None,
    *,
    status: str | None = None,
) -> dict[str, object]:
    action: dict[str, object] = {
        "domain": domain,
        "service": service,
        "target": {"entity_id": entity_id},
        "service_data": service_data or {},
    }
    if status is not None:
        action["status"] = status
    return action


def _observed(
    domain: str,
    service: str,
    entity_id: str,
    service_data: dict[str, object] | None = None,
) -> ObservedAction:
    return ObservedAction(domain, service, (entity_id,), service_data or {})


@pytest.mark.parametrize(
    "case",
    # The offline stub routes only direct_*/brightness_*/color_* requests. Every
    # other family (no_action_*, ambiguous_*, discover_*, condition_*) is
    # intentionally outside this stub smoke and is exercised by real-model runs;
    # do not broaden this allow-list — non-routed non-empty cases would no-op
    # and score incorrect under the stub.
    [case for case in CASES if case.id.startswith(("direct_", "brightness_", "color_"))],
    ids=lambda case: case.id,
)
async def test_each_authored_direct_action_passes(case: EvalCase, tmp_path: Path) -> None:
    trace = await run_case(
        baseline_candidate(),
        "stub",
        case,
        EvalConfig(
            models=["stub"],
            candidates=["baseline"],
            prompt_profile=DEFAULT_PROMPT_PROFILE,
            cases=None,
            homes=None,
            runs_dir=tmp_path,
        ),
        profile=resolve_profile(DEFAULT_PROMPT_PROFILE),
    )

    assert trace.outcome.state == "correct"
    assert trace.outcome.reason == trace.action_result.reason == "ok"
    assert trace.answer == "Done."
    assert trace.action_result.passed is True


_BEDROOM_ON = RequiredAction("light", "turn_on", ("light.bedroom",))
_LIVING_OFF = RequiredAction("light", "turn_off", ("light.living",))
_BRIGHT_BEDROOM = RequiredAction("light", "turn_on", ("light.bedroom",), {"brightness": 100})


def _case(*actions: RequiredAction) -> EvalCase:
    return EvalCase("action-case", "home_minimal", "Perform actions", actions)


@pytest.mark.parametrize(
    ("case", "recorded", "expected_result"),
    [
        pytest.param(
            _case(_BEDROOM_ON),
            [],
            ActionResult(
                False,
                "no_action",
                (ActionComparison(_BEDROOM_ON, None, False, False, False, False),),
            ),
            id="no-action",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [_action("light", "turn_on", "light.bedroom", status="error")],
            ActionResult(
                False,
                "action_rejected",
                (ActionComparison(_BEDROOM_ON, None, False, False, False, False),),
            ),
            id="action-rejected",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [_action("light", "turn_off", "light.bedroom")],
            ActionResult(
                False,
                "wrong_service",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_off", "light.bedroom"),
                        False,
                        True,
                        True,
                        False,
                    ),
                ),
            ),
            id="wrong-service-right-target",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [_action("light", "turn_on", "light.living")],
            ActionResult(
                False,
                "wrong_target",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_on", "light.living"),
                        True,
                        False,
                        True,
                        False,
                    ),
                ),
            ),
            id="right-service-wrong-target",
        ),
        pytest.param(
            _case(_BRIGHT_BEDROOM),
            [_action("light", "turn_on", "light.bedroom", {"brightness": 90})],
            ActionResult(
                False,
                "wrong_service_data",
                (
                    ActionComparison(
                        _BRIGHT_BEDROOM,
                        _observed("light", "turn_on", "light.bedroom", {"brightness": 90}),
                        True,
                        True,
                        False,
                        False,
                    ),
                ),
            ),
            id="right-service-target-wrong-data",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [_action("switch", "turn_off", "switch.living")],
            ActionResult(
                False,
                "wrong_service_and_target",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("switch", "turn_off", "switch.living"),
                        False,
                        False,
                        True,
                        False,
                    ),
                ),
            ),
            id="wrong-service-and-target",
        ),
        pytest.param(
            _case(_BRIGHT_BEDROOM),
            [_action("switch", "turn_on", "light.bedroom", {"brightness": 90})],
            ActionResult(
                False,
                "wrong_service_and_data",
                (
                    ActionComparison(
                        _BRIGHT_BEDROOM,
                        _observed("switch", "turn_on", "light.bedroom", {"brightness": 90}),
                        False,
                        True,
                        False,
                        False,
                    ),
                ),
            ),
            id="wrong-service-and-data",
        ),
        pytest.param(
            _case(_BRIGHT_BEDROOM),
            [_action("light", "turn_on", "light.living", {"brightness": 90})],
            ActionResult(
                False,
                "wrong_target_and_data",
                (
                    ActionComparison(
                        _BRIGHT_BEDROOM,
                        _observed("light", "turn_on", "light.living", {"brightness": 90}),
                        True,
                        False,
                        False,
                        False,
                    ),
                ),
            ),
            id="wrong-target-and-data",
        ),
        pytest.param(
            _case(_BRIGHT_BEDROOM),
            [_action("switch", "turn_off", "switch.living", {"brightness": 90})],
            ActionResult(
                False,
                "wrong_service_target_and_data",
                (
                    ActionComparison(
                        _BRIGHT_BEDROOM,
                        _observed("switch", "turn_off", "switch.living", {"brightness": 90}),
                        False,
                        False,
                        False,
                        False,
                    ),
                ),
            ),
            id="wrong-service-target-and-data",
        ),
        pytest.param(
            _case(_BEDROOM_ON, _LIVING_OFF),
            [_action("light", "turn_on", "light.bedroom")],
            ActionResult(
                False,
                "missing_action",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_on", "light.bedroom"),
                        True,
                        True,
                        True,
                        True,
                    ),
                    ActionComparison(_LIVING_OFF, None, False, False, False, False),
                ),
            ),
            id="missing-action-after-match",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [
                _action("light", "turn_on", "light.bedroom"),
                _action("switch", "turn_on", "switch.garage"),
            ],
            ActionResult(
                False,
                "unexpected_action",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_on", "light.bedroom"),
                        True,
                        True,
                        True,
                        True,
                    ),
                ),
                (_observed("switch", "turn_on", "switch.garage"),),
            ),
            id="unrelated-extra",
        ),
        pytest.param(
            _case(_BEDROOM_ON),
            [
                _action("light", "turn_on", "light.bedroom"),
                _action("light", "turn_on", "light.bedroom"),
            ],
            ActionResult(
                False,
                "duplicate_action",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_on", "light.bedroom"),
                        True,
                        True,
                        True,
                        True,
                    ),
                ),
                (_observed("light", "turn_on", "light.bedroom"),),
            ),
            id="duplicate-exact",
        ),
        pytest.param(
            _case(_BEDROOM_ON, _LIVING_OFF),
            [
                _action("light", "turn_off", "light.bedroom"),
                _action("light", "turn_on", "light.garage"),
            ],
            ActionResult(
                False,
                "multiple_action_mismatches",
                (
                    ActionComparison(
                        _BEDROOM_ON,
                        _observed("light", "turn_off", "light.bedroom"),
                        False,
                        True,
                        True,
                        False,
                    ),
                    ActionComparison(
                        _LIVING_OFF,
                        _observed("light", "turn_on", "light.garage"),
                        False,
                        False,
                        True,
                        False,
                    ),
                ),
            ),
            id="multiple-mismatches",
        ),
        pytest.param(
            _case(_BRIGHT_BEDROOM),
            [_action("light", "turn_on", "light.bedroom", {"brightness": 100})],
            ActionResult(
                True,
                "ok",
                (
                    ActionComparison(
                        _BRIGHT_BEDROOM,
                        _observed("light", "turn_on", "light.bedroom", {"brightness": 100}),
                        True,
                        True,
                        True,
                        True,
                    ),
                ),
            ),
            id="passing-comparison",
        ),
    ],
)
def test_action_assessment_failure_taxonomy(
    case: EvalCase,
    recorded: list[dict[str, object]],
    expected_result: ActionResult,
) -> None:
    outcome, result, _ledger = evaluate_case(case, recorded)

    assert result == expected_result
    assert outcome.reason == result.reason
    assert outcome.state == ("correct" if result.passed else "incorrect")


def test_unset_service_data_is_not_an_implicit_oracle_field() -> None:
    outcome, result, _ledger = evaluate_case(
        _case(_BEDROOM_ON),
        [_action("light", "turn_on", "light.bedroom", {"transition": 1})],
    )

    assert outcome.state == "correct"
    assert result.comparisons[0].service_data_matches is True
    assert result.comparisons[0].actual == _observed("light", "turn_on", "light.bedroom", {"transition": 1})


def test_successful_match_remains_correct_when_an_attempt_was_rejected() -> None:
    successful = _action("light", "turn_on", "light.bedroom")
    rejected = _action("light", "turn_off", "light.living", status="error")

    outcome, result, ledger = evaluate_case(_case(_BEDROOM_ON), [rejected, successful])

    assert outcome.state == "correct"
    assert result.reason == "ok"
    assert ledger.rejected == (rejected,)
