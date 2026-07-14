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
    DesiredState,
    EvalCase,
    ObservedAction,
    OverlayStateSeed,
    RequiredAction,
)
from llm_sandbox_evals.scoring import evaluate_case
import pytest


def _action(
    domain: str,
    service: str,
    entity_id: str | list[str],
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
    ("case", "expected_mode", "expected_reason"),
    # The offline stub routes only direct_*/brightness_*/color_* requests. Every
    # other family (no_action_*, ambiguous_*, discover_*, condition_*) is
    # intentionally outside this stub smoke and is exercised by real-model runs;
    # do not broaden this allow-list — non-routed non-empty cases would no-op
    # and score incorrect under the stub.
    [
        pytest.param(
            case,
            "end_state" if case.id.startswith("direct_") else "actions",
            "end_state_satisfied" if case.id.startswith("direct_") else "ok",
            id=case.id,
        )
        for case in CASES
        if case.id.startswith(("direct_", "brightness_", "color_"))
    ],
)
async def test_each_authored_direct_action_passes(
    case: EvalCase, expected_mode: str, expected_reason: str, tmp_path: Path
) -> None:
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
    assert trace.outcome.scoring_mode == expected_mode
    assert trace.outcome.score_reason == expected_reason
    assert trace.action_result.reason == "ok"
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
    outcome, result, _ledger, _end_state = evaluate_case(case, recorded, overlay_seeds=(), invoker_calls=())

    assert result == expected_result
    assert outcome.score_reason == result.reason
    assert outcome.state == ("correct" if result.passed else "incorrect")


def test_unset_service_data_is_not_an_implicit_oracle_field() -> None:
    outcome, result, _ledger, _end_state = evaluate_case(
        _case(_BEDROOM_ON),
        [_action("light", "turn_on", "light.bedroom", {"transition": 1})],
        overlay_seeds=(),
        invoker_calls=(),
    )

    assert outcome.state == "correct"
    assert result.comparisons[0].service_data_matches is True
    assert result.comparisons[0].actual == _observed("light", "turn_on", "light.bedroom", {"transition": 1})


def test_successful_match_remains_correct_when_an_attempt_was_rejected() -> None:
    successful = _action("light", "turn_on", "light.bedroom")
    rejected = _action("light", "turn_off", "light.living", status="error")

    outcome, result, ledger, _end_state = evaluate_case(
        _case(_BEDROOM_ON), [rejected, successful], overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "correct"
    assert result.reason == "ok"
    assert ledger.rejected == (rejected,)


_PARTITION_TARGETS = ("light.gamma", "light.alpha", "light.delta", "light.beta")
_PARTITION_ON = RequiredAction("light", "turn_on", _PARTITION_TARGETS)


def test_exact_multi_target_action_remains_an_exact_match() -> None:
    recorded = [
        _action(
            "light",
            "turn_on",
            ["light.delta", "light.beta", "light.alpha", "light.gamma"],
        )
    ]

    outcome, result, _ledger, _end_state = evaluate_case(
        _case(_PARTITION_ON), recorded, overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "correct"
    assert result.passed is True
    assert result.reason == "ok"
    assert result.comparisons[0].actual == ObservedAction(
        "light",
        "turn_on",
        tuple(sorted(_PARTITION_TARGETS)),
        {},
    )


@pytest.mark.parametrize(
    "recorded",
    [
        pytest.param(
            [
                _action("light", "turn_on", "light.beta"),
                _action("light", "turn_on", "light.gamma"),
                _action("light", "turn_on", "light.delta"),
                _action("light", "turn_on", "light.alpha"),
            ],
            id="all-single-calls-in-varied-order",
        ),
        pytest.param(
            [
                _action("light", "turn_on", ["light.beta", "light.delta"]),
                _action("light", "turn_on", "light.gamma"),
                _action("light", "turn_on", "light.alpha"),
            ],
            id="mixed-size-calls-in-varied-order",
        ),
    ],
)
def test_disjoint_target_partitions_are_equivalent(
    recorded: list[dict[str, object]],
) -> None:
    outcome, result, ledger, _end_state = evaluate_case(
        _case(_PARTITION_ON), recorded, overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "correct"
    assert result.passed is True
    assert result.reason == "equivalent_target_partition"
    assert result.comparisons == (
        ActionComparison(
            _PARTITION_ON,
            ObservedAction(
                "light",
                "turn_on",
                tuple(sorted(_PARTITION_TARGETS)),
                {},
            ),
            True,
            True,
            True,
            True,
        ),
    )
    assert ledger.successful == tuple(recorded)


@pytest.mark.parametrize(
    ("required", "recorded"),
    [
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta")),
            [
                _action(
                    "light",
                    "turn_on",
                    "light.beta",
                    {"transition": 1.0, "options": {"levels": [1.0, 2]}},
                ),
                _action(
                    "light",
                    "turn_on",
                    "light.alpha",
                    {"options": {"levels": [1, 2.0]}, "transition": 1},
                ),
            ],
            id="unspecified-authored-data-still-requires-identical-canonical-call-data",
        ),
        pytest.param(
            RequiredAction(
                "light",
                "turn_on",
                ("light.alpha", "light.beta"),
                {"transition": 1, "options": {"levels": [1, 2]}},
            ),
            [
                _action(
                    "light",
                    "turn_on",
                    "light.beta",
                    {"options": {"levels": [1.0, 2]}, "transition": 1.0},
                ),
                _action(
                    "light",
                    "turn_on",
                    "light.alpha",
                    {"transition": 1, "options": {"levels": [1, 2.0]}},
                ),
            ],
            id="authored-data-matches-canonical-call-data",
        ),
    ],
)
def test_partition_service_data_uses_canonical_equivalence(
    required: RequiredAction,
    recorded: list[dict[str, object]],
) -> None:
    outcome, result, _ledger, _end_state = evaluate_case(
        _case(required), recorded, overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "correct"
    assert result.passed is True
    assert result.reason == "equivalent_target_partition"
    assert result.comparisons[0].service_data_matches is True
    assert result.comparisons[0].actual == ObservedAction(
        "light",
        "turn_on",
        ("light.alpha", "light.beta"),
        {"options": {"levels": [1, 2]}, "transition": 1},
    )


@pytest.mark.parametrize(
    ("required", "recorded"),
    [
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", ["light.alpha", "light.alpha"]),
                _action("light", "turn_on", ["light.beta", "light.gamma"]),
            ],
            id="duplicate-target-within-one-call",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", ["light.alpha", "light.beta"]),
                _action("light", "turn_on", ["light.beta", "light.gamma"]),
            ],
            id="overlapping-targets-across-calls",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", "light.alpha"),
                _action("light", "turn_on", "light.beta"),
            ],
            id="missing-target",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", ["light.alpha", "light.beta"]),
                _action("light", "turn_on", ["light.gamma", "light.delta"]),
            ],
            id="extra-target",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", "light.alpha"),
                _action("switch", "turn_on", "light.beta"),
                _action("light", "turn_on", "light.gamma"),
            ],
            id="wrong-domain-member",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", "light.alpha"),
                _action("light", "turn_off", "light.beta"),
                _action("light", "turn_on", "light.gamma"),
            ],
            id="wrong-service-member",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", "light.alpha", {"transition": 1}),
                _action(
                    "light",
                    "turn_on",
                    ["light.beta", "light.gamma"],
                    {"transition": 2},
                ),
            ],
            id="differing-call-data-with-unspecified-authored-data",
        ),
        pytest.param(
            RequiredAction(
                "light",
                "turn_on",
                ("light.alpha", "light.beta", "light.gamma"),
                {"transition": 1},
            ),
            [
                _action("light", "turn_on", "light.alpha", {"transition": 2}),
                _action(
                    "light",
                    "turn_on",
                    ["light.beta", "light.gamma"],
                    {"transition": 2.0},
                ),
            ],
            id="authored-data-mismatch",
        ),
        pytest.param(
            RequiredAction("light", "turn_on", ("light.alpha", "light.beta", "light.gamma")),
            [
                _action("light", "turn_on", "light.alpha"),
                _action("light", "turn_on", ["light.beta", "light.gamma"]),
                _action("switch", "turn_on", "switch.unrelated"),
            ],
            id="unrelated-extra-call",
        ),
    ],
)
def test_invalid_target_partitions_fail(
    required: RequiredAction,
    recorded: list[dict[str, object]],
) -> None:
    outcome, result, _ledger, _end_state = evaluate_case(
        _case(required), recorded, overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "incorrect"
    assert result.passed is False
    assert result.reason == "multiple_action_mismatches"


def test_exact_match_is_consumed_before_singular_partition_fallback() -> None:
    aggregate = RequiredAction(
        "light",
        "turn_on",
        ("light.alpha", "light.beta", "light.gamma"),
    )
    exact = RequiredAction("switch", "turn_on", ("switch.garage",))
    recorded = [
        _action("light", "turn_on", "light.gamma"),
        _action("switch", "turn_on", "switch.garage"),
        _action("light", "turn_on", ["light.beta", "light.alpha"]),
    ]

    outcome, result, _ledger, _end_state = evaluate_case(
        _case(aggregate, exact), recorded, overlay_seeds=(), invoker_calls=()
    )

    assert outcome.state == "correct"
    assert result.passed is True
    assert result.reason == "equivalent_target_partition"
    assert tuple(comparison.actual for comparison in result.comparisons) == (
        ObservedAction(
            "light",
            "turn_on",
            ("light.alpha", "light.beta", "light.gamma"),
            {},
        ),
        _observed("switch", "turn_on", "switch.garage"),
    )


# ---------------------------------------------------------------------------
# End-state primary scoring: state satisfaction overrides action diagnostics
# ---------------------------------------------------------------------------


def _state_case(*desired: DesiredState, actions: tuple[RequiredAction, ...] = ()) -> EvalCase:
    return EvalCase("state-case", "home_minimal", "Perform actions", actions, desired)


def _seed(entity_id: str, state: str) -> OverlayStateSeed:
    return OverlayStateSeed(entity_id, entity_id.split(".", 1)[0], state)


def test_initially_satisfied_with_no_calls_is_end_state_correct() -> None:
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    outcome, _result, _ledger, end_state = evaluate_case(
        _state_case(*desired), [], overlay_seeds=seeds, invoker_calls=()
    )
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    assert end_state.status == "satisfied"


def test_satisfied_state_overrides_wrong_action_diagnostics() -> None:
    # The light is already on; a wrong-target call does not change the overlay.
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    required = RequiredAction("light", "turn_on", ("light.bedroom",))
    wrong_call = _action("light", "turn_on", "light.living")
    outcome, result, _ledger, end_state = evaluate_case(
        _state_case(*desired, actions=(required,)),
        [wrong_call],
        overlay_seeds=seeds,
        invoker_calls=[wrong_call],
    )
    # State is primary: satisfied despite the action ledger showing a wrong-target mismatch.
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    assert end_state.status == "satisfied"
    # Action diagnostics still report the mismatch.
    assert result.passed is False


def test_satisfied_state_overrides_extra_action() -> None:
    desired = (DesiredState("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    extra_call = _action("light", "turn_on", "light.living")
    outcome, result, _ledger, _end_state = evaluate_case(
        _state_case(*desired), [extra_call], overlay_seeds=seeds, invoker_calls=[extra_call]
    )
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    # Action diagnostics show an unexpected action.
    assert result.passed is False


def test_evaluable_unsatisfied_state_is_incorrect_even_if_actions_pass() -> None:
    # The desired state is off but the call turns the light on.
    desired = (DesiredState("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    call = _action("light", "turn_on", "light.bedroom")
    required = RequiredAction("light", "turn_on", ("light.bedroom",))
    outcome, result, _ledger, end_state = evaluate_case(
        _state_case(*desired, actions=(required,)),
        [call],
        overlay_seeds=seeds,
        invoker_calls=[call],
    )
    # State is primary: unsatisfied even though the action ledger matches.
    assert outcome.state == "incorrect"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_unsatisfied"
    assert end_state.status == "unsatisfied"
    assert result.passed is True


def test_no_desired_states_uses_action_fallback() -> None:
    outcome, result, _ledger, end_state = evaluate_case(
        _case(_BEDROOM_ON),
        [_action("light", "turn_on", "light.bedroom")],
        overlay_seeds=(),
        invoker_calls=(),
    )
    assert outcome.scoring_mode == "actions"
    assert outcome.score_reason == result.reason == "ok"
    assert end_state.status == "not_authored"


def test_unevaluable_predicate_uses_action_fallback() -> None:
    desired = (DesiredState("light.missing", "on"),)
    outcome, result, _ledger, end_state = evaluate_case(
        _state_case(*desired, actions=(_BEDROOM_ON,)),
        [_action("light", "turn_on", "light.bedroom")],
        overlay_seeds=(),
        invoker_calls=(),
    )
    assert outcome.scoring_mode == "actions"
    assert outcome.score_reason == result.reason == "ok"
    assert end_state.status == "unevaluable"


def test_ordered_toggle_changes_final_state_verdict() -> None:
    desired = (DesiredState("switch.outlet", "on"),)
    seeds = (_seed("switch.outlet", "off"),)
    single_toggle = _action("switch", "toggle", "switch.outlet")
    double_toggle = [single_toggle, _action("switch", "toggle", "switch.outlet")]

    outcome1, _, _, end1 = evaluate_case(
        _state_case(*desired), [single_toggle], overlay_seeds=seeds, invoker_calls=[single_toggle]
    )
    outcome2, _, _, end2 = evaluate_case(
        _state_case(*desired), double_toggle, overlay_seeds=seeds, invoker_calls=double_toggle
    )
    assert outcome1.state == "correct"
    assert end1.status == "satisfied"
    # Two toggles return to off — unsatisfied.
    assert outcome2.state == "incorrect"
    assert end2.status == "unsatisfied"


@pytest.mark.parametrize(
    "case_id",
    ["no_action_light_already_on"],
)
async def test_no_action_already_on_passes_state_primary(case_id: str, tmp_path: Path) -> None:
    """The already-on no-route case passes via end-state with zero tool calls."""
    case = next(c for c in CASES if c.id == case_id)
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
    assert trace.outcome.scoring_mode == "end_state"
    assert trace.outcome.score_reason == "end_state_satisfied"
    assert trace.end_state_result.status == "satisfied"
    assert trace.recorded_invocations == ()
