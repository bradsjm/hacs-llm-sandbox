from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import run_case
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.schema import (
    ActionComparison,
    ActionResult,
    DesiredEntity,
    EvalCase,
    ObservedAction,
    OverlayStateSeed,
    RequestVariant,
    RequiredAction,
)
from llm_sandbox_evals.scoring import evaluate_case, extract_overlay_seeds
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
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


def _home_full_case(case_id: str) -> EvalCase:
    """Return one authored home_full case by its stable corpus identifier."""
    return next(case for case in CASES if case.id == case_id)


def _home_full_seeds(case: EvalCase) -> tuple[OverlayStateSeed, ...]:
    """Extract predicate seeds from the same scoped fixture snapshot used by the harness."""
    fixture = get_home(case.home)
    snapshot = apply_scope(fixture.snapshot(), EVAL_SCOPE)
    return extract_overlay_seeds(snapshot, case.desired_entities)


_BASEMENT_CEILING_TARGETS = (
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
)


@pytest.mark.parametrize(
    "case",
    # The offline stub routes only direct_*/brightness_*/color_* requests. Every
    # other family (no_action_*, ambiguous_*, discover_*, condition_*) is
    # intentionally outside this stub smoke and is exercised by real-model runs;
    # do not broaden this allow-list — non-routed non-empty cases would no-op
    # and score incorrect under the stub.
    [pytest.param(case, id=case.id) for case in CASES if case.id.startswith(("direct_", "brightness_", "color_"))],
)
async def test_each_authored_direct_action_passes(case: EvalCase, tmp_path: Path) -> None:
    trace = await run_case(
        baseline_candidate(),
        "stub",
        case,
        case.requests[0],
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
    assert trace.action_result.reason == "ok"
    assert trace.answer == "Done."
    assert trace.action_result.passed is True


@pytest.mark.parametrize(
    ("case_id", "expected_service_data"),
    [
        pytest.param("brightness_utility_room_ceiling", {"brightness_pct": 50}, id="brightness"),
        pytest.param("color_utility_room_accent", {"color_temp_kelvin": 2700}, id="color-temperature"),
    ],
)
async def test_stub_attribute_actions_record_canonical_service_data(
    case_id: str, expected_service_data: dict[str, object], tmp_path: Path
) -> None:
    case = _home_full_case(case_id)
    trace = await run_case(
        baseline_candidate(),
        "stub",
        case,
        case.requests[0],
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
    assert trace.end_state_result.status == "satisfied"
    assert trace.action_result.passed is True
    assert trace.recorded_invocations[0]["service_data"] == expected_service_data
    assert trace.end_state_result.comparisons[0].actual_attributes == case.desired_entities[0].attributes


@pytest.mark.parametrize(
    ("recorded", "expected_satisfied"),
    [
        pytest.param([], False, id="no-action"),
        pytest.param(
            [_action("light", "turn_on", "light.storage_room_ceiling")],
            False,
            id="wrong-target",
        ),
        pytest.param(
            [_action("light", "turn_on", "light.utility_room_ceiling")],
            True,
            id="correct-target",
        ),
    ],
)
def test_direct_turn_on_utility_room_ceiling_requires_a_transition(
    recorded: list[dict[str, object]], expected_satisfied: bool
) -> None:
    case = _home_full_case("direct_turn_on_utility_room_ceiling")
    evaluation = evaluate_case(
        case,
        recorded,
        overlay_seeds=_home_full_seeds(case),
        invoker_calls=recorded,
    )

    assert evaluation.outcome.scoring_mode == "end_state"
    assert evaluation.outcome.state == ("correct" if expected_satisfied else "incorrect")
    assert evaluation.end_state_result.status == ("satisfied" if expected_satisfied else "unsatisfied")


@pytest.mark.parametrize(
    ("recorded", "expected_passed"),
    [
        pytest.param([], False, id="no-action"),
        pytest.param(
            [_action("light", "turn_on", "light.utility_room_ceiling")],
            False,
            id="ceiling-only-partial",
        ),
        pytest.param(
            [_action("light", "turn_on", "light.storage_room_ceiling")],
            False,
            id="wrong-target",
        ),
        pytest.param(
            [_action("light", "turn_on", ["light.utility_room_accent", "light.utility_room_ceiling"])],
            True,
            id="complete-two-target-action",
        ),
    ],
)
def test_utility_room_discovery_uses_exact_action_fallback(
    recorded: list[dict[str, object]], expected_passed: bool
) -> None:
    case = _home_full_case("discover_utility_room_lights")
    evaluation = evaluate_case(
        case,
        recorded,
        overlay_seeds=_home_full_seeds(case),
        invoker_calls=recorded,
    )

    assert evaluation.outcome.scoring_mode == "actions"
    assert evaluation.outcome.state == ("correct" if expected_passed else "incorrect")
    assert evaluation.action_result.passed is expected_passed
    assert evaluation.end_state_result.status == "not_authored"


@pytest.mark.parametrize(
    ("recorded", "expected_satisfied"),
    [
        pytest.param([], False, id="no-action"),
        pytest.param(
            [_action("light", "turn_on", list(_BASEMENT_CEILING_TARGETS[:1]))],
            False,
            id="partial-target-set",
        ),
        pytest.param(
            [_action("light", "turn_on", "light.utility_room_accent")],
            False,
            id="wrong-target",
        ),
        pytest.param(
            [_action("light", "turn_on", list(_BASEMENT_CEILING_TARGETS))],
            True,
            id="complete-twelve-target-set",
        ),
    ],
)
def test_basement_ceiling_discovery_requires_all_twelve_state_transitions(
    recorded: list[dict[str, object]], expected_satisfied: bool
) -> None:
    case = _home_full_case("discover_basement_ceiling_lights")
    evaluation = evaluate_case(
        case,
        recorded,
        overlay_seeds=_home_full_seeds(case),
        invoker_calls=recorded,
    )

    assert evaluation.outcome.scoring_mode == "end_state"
    assert evaluation.outcome.state == ("correct" if expected_satisfied else "incorrect")
    assert evaluation.end_state_result.status == ("satisfied" if expected_satisfied else "unsatisfied")


@pytest.mark.parametrize(
    ("case_id", "service_data", "expected_passed", "expected_reason"),
    [
        pytest.param(
            "brightness_utility_room_ceiling",
            {"brightness_pct": 50},
            True,
            "ok",
            id="brightness-canonical-data",
        ),
        pytest.param(
            "brightness_utility_room_ceiling",
            None,
            False,
            "wrong_service_data",
            id="brightness-missing-data",
        ),
        pytest.param(
            "color_utility_room_accent",
            {"color_temp_kelvin": 2700},
            True,
            "ok",
            id="color-canonical-data",
        ),
        pytest.param(
            "color_utility_room_accent",
            {"color_temp_kelvin": 3000},
            False,
            "wrong_service_data",
            id="color-wrong-data",
        ),
    ],
)
def test_attribute_final_value_is_primary_with_action_diagnostics(
    case_id: str,
    service_data: dict[str, object] | None,
    expected_passed: bool,
    expected_reason: str,
) -> None:
    case = _home_full_case(case_id)
    required = case.required_actions[0]
    recorded = [_action(required.domain, required.service, list(required.target_entity_ids), service_data)]
    evaluation = evaluate_case(
        case,
        recorded,
        overlay_seeds=_home_full_seeds(case),
        invoker_calls=recorded,
    )

    assert evaluation.outcome.scoring_mode == "end_state"
    assert evaluation.outcome.state == ("correct" if expected_passed else "incorrect")
    assert evaluation.outcome.score_reason == ("end_state_satisfied" if expected_passed else "end_state_unsatisfied")
    assert evaluation.action_result.reason == expected_reason
    assert evaluation.action_result.passed is expected_passed
    assert evaluation.end_state_result.status == ("satisfied" if expected_passed else "unsatisfied")


_BEDROOM_ON = RequiredAction("light", "turn_on", ("light.bedroom",))
_LIVING_OFF = RequiredAction("light", "turn_off", ("light.living",))
_BRIGHT_BEDROOM = RequiredAction("light", "turn_on", ("light.bedroom",), {"brightness": 100})


def _case(*actions: RequiredAction) -> EvalCase:
    return EvalCase(
        "action-case",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Perform actions"),),
        actions,
    )


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
    evaluation = evaluate_case(case, recorded, overlay_seeds=(), invoker_calls=())
    outcome, result = evaluation.outcome, evaluation.action_result

    assert result == expected_result
    assert outcome.score_reason == result.reason
    assert outcome.state == ("correct" if result.passed else "incorrect")


def test_unset_service_data_is_not_an_implicit_oracle_field() -> None:
    evaluation = evaluate_case(
        _case(_BEDROOM_ON),
        [_action("light", "turn_on", "light.bedroom", {"transition": 1})],
        overlay_seeds=(),
        invoker_calls=(),
    )
    outcome, result = evaluation.outcome, evaluation.action_result

    assert outcome.state == "correct"
    assert result.comparisons[0].service_data_matches is True
    assert result.comparisons[0].actual == _observed("light", "turn_on", "light.bedroom", {"transition": 1})


def test_successful_match_remains_correct_when_an_attempt_was_rejected() -> None:
    successful = _action("light", "turn_on", "light.bedroom")
    rejected = _action("light", "turn_off", "light.living", status="error")

    evaluation = evaluate_case(_case(_BEDROOM_ON), [rejected, successful], overlay_seeds=(), invoker_calls=())
    outcome, result, ledger = evaluation.outcome, evaluation.action_result, evaluation.action_ledger

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

    evaluation = evaluate_case(_case(_PARTITION_ON), recorded, overlay_seeds=(), invoker_calls=())
    outcome, result = evaluation.outcome, evaluation.action_result

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
    evaluation = evaluate_case(_case(_PARTITION_ON), recorded, overlay_seeds=(), invoker_calls=())
    outcome, result, ledger = evaluation.outcome, evaluation.action_result, evaluation.action_ledger

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
    evaluation = evaluate_case(_case(required), recorded, overlay_seeds=(), invoker_calls=())
    outcome, result = evaluation.outcome, evaluation.action_result

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
    evaluation = evaluate_case(_case(required), recorded, overlay_seeds=(), invoker_calls=())
    outcome, result = evaluation.outcome, evaluation.action_result

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

    evaluation = evaluate_case(_case(aggregate, exact), recorded, overlay_seeds=(), invoker_calls=())
    outcome, result = evaluation.outcome, evaluation.action_result

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


def _state_case(*desired: DesiredEntity, actions: tuple[RequiredAction, ...] = ()) -> EvalCase:
    return EvalCase(
        "state-case",
        "home_minimal",
        "test",
        (RequestVariant("canonical", "Perform actions"),),
        actions,
        desired,
    )


def _seed(entity_id: str, state: str) -> OverlayStateSeed:
    return OverlayStateSeed(entity_id, entity_id.split(".", 1)[0], state)


def test_initially_satisfied_with_no_calls_is_end_state_correct() -> None:
    desired = (DesiredEntity("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    evaluation = evaluate_case(_state_case(*desired), [], overlay_seeds=seeds, invoker_calls=())
    outcome, end_state = evaluation.outcome, evaluation.end_state_result
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    assert end_state.status == "satisfied"


def test_satisfied_state_overrides_wrong_action_diagnostics() -> None:
    # The light is already on; a wrong-target call does not change the overlay.
    desired = (DesiredEntity("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    required = RequiredAction("light", "turn_on", ("light.bedroom",))
    wrong_call = _action("light", "turn_on", "light.living")
    evaluation = evaluate_case(
        _state_case(*desired, actions=(required,)),
        [wrong_call],
        overlay_seeds=seeds,
        invoker_calls=[wrong_call],
    )
    outcome, result, end_state = (
        evaluation.outcome,
        evaluation.action_result,
        evaluation.end_state_result,
    )
    # State is primary: satisfied despite the action ledger showing a wrong-target mismatch.
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    assert end_state.status == "satisfied"
    # Action diagnostics still report the mismatch.
    assert result.passed is False


def test_satisfied_state_overrides_extra_action() -> None:
    desired = (DesiredEntity("light.bedroom", "on"),)
    seeds = (_seed("light.bedroom", "on"),)
    extra_call = _action("light", "turn_on", "light.living")
    evaluation = evaluate_case(_state_case(*desired), [extra_call], overlay_seeds=seeds, invoker_calls=[extra_call])
    outcome, result = evaluation.outcome, evaluation.action_result
    assert outcome.state == "correct"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_satisfied"
    # Action diagnostics show an unexpected action.
    assert result.passed is False


def test_evaluable_unsatisfied_state_is_incorrect_even_if_actions_pass() -> None:
    # The desired state is off but the call turns the light on.
    desired = (DesiredEntity("light.bedroom", "off"),)
    seeds = (_seed("light.bedroom", "off"),)
    call = _action("light", "turn_on", "light.bedroom")
    required = RequiredAction("light", "turn_on", ("light.bedroom",))
    evaluation = evaluate_case(
        _state_case(*desired, actions=(required,)),
        [call],
        overlay_seeds=seeds,
        invoker_calls=[call],
    )
    outcome, result, end_state = (
        evaluation.outcome,
        evaluation.action_result,
        evaluation.end_state_result,
    )
    # State is primary: unsatisfied even though the action ledger matches.
    assert outcome.state == "incorrect"
    assert outcome.scoring_mode == "end_state"
    assert outcome.score_reason == "end_state_unsatisfied"
    assert end_state.status == "unsatisfied"
    assert result.passed is True


def test_no_desired_entities_uses_action_fallback() -> None:
    evaluation = evaluate_case(
        _case(_BEDROOM_ON),
        [_action("light", "turn_on", "light.bedroom")],
        overlay_seeds=(),
        invoker_calls=(),
    )
    outcome, result, end_state = (
        evaluation.outcome,
        evaluation.action_result,
        evaluation.end_state_result,
    )
    assert outcome.scoring_mode == "actions"
    assert outcome.score_reason == result.reason == "ok"
    assert end_state.status == "not_authored"


def test_unevaluable_predicate_uses_action_fallback() -> None:
    desired = (DesiredEntity("light.missing", "on"),)
    evaluation = evaluate_case(
        _state_case(*desired, actions=(_BEDROOM_ON,)),
        [_action("light", "turn_on", "light.bedroom")],
        overlay_seeds=(),
        invoker_calls=(),
    )
    outcome, result, end_state = (
        evaluation.outcome,
        evaluation.action_result,
        evaluation.end_state_result,
    )
    assert outcome.scoring_mode == "actions"
    assert outcome.score_reason == result.reason == "ok"
    assert end_state.status == "unevaluable"


def test_unsupported_attribute_predicate_uses_action_fallback() -> None:
    desired = (DesiredEntity("light.bedroom", attributes={"hue": 120}),)
    call = _action("light", "turn_on", "light.bedroom")
    evaluation = evaluate_case(
        _state_case(*desired, actions=(_BEDROOM_ON,)),
        [call],
        overlay_seeds=(OverlayStateSeed("light.bedroom", "light", "off", {"hue": 0}),),
        invoker_calls=[call],
    )
    outcome, result, end_state = (
        evaluation.outcome,
        evaluation.action_result,
        evaluation.end_state_result,
    )

    assert outcome.scoring_mode == "actions"
    assert outcome.score_reason == result.reason == "ok"
    assert end_state.status == "unevaluable"


def test_ordered_toggle_changes_final_state_verdict() -> None:
    desired = (DesiredEntity("switch.outlet", "on"),)
    seeds = (_seed("switch.outlet", "off"),)
    single_toggle = _action("switch", "toggle", "switch.outlet")
    double_toggle = [single_toggle, _action("switch", "toggle", "switch.outlet")]

    evaluation1 = evaluate_case(
        _state_case(*desired), [single_toggle], overlay_seeds=seeds, invoker_calls=[single_toggle]
    )
    evaluation2 = evaluate_case(_state_case(*desired), double_toggle, overlay_seeds=seeds, invoker_calls=double_toggle)
    assert evaluation1.outcome.state == "correct"
    assert evaluation1.end_state_result.status == "satisfied"
    # Two toggles return to off — unsatisfied.
    assert evaluation2.outcome.state == "incorrect"
    assert evaluation2.end_state_result.status == "unsatisfied"


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
        case.requests[0],
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
