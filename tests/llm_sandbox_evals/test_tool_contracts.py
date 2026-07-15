from typing import cast

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts.profiles import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, RecorderSource
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from homeassistant.helpers import llm
from llm_sandbox_evals.agent_runner import _validate_recorder_tool
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.runtime import build_eval_runtime, build_fixture_recorder_source
from llm_sandbox_evals.schema import EvalCase, RequestVariant, RequiredAction
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope


async def test_recorder_selector_no_match_returns_error_with_guidance() -> None:
    snapshot = _scoped_snapshot()
    tool = GetHistoryTool("eval")
    data = cast(dict[str, object], tool.parameters({"area_id": "area_missing", "hours": 24}))

    result = await tool.run_query(snapshot, data, _recorder_source(snapshot))

    assert result["status"] == "error"
    error = cast(dict[str, object], result["error"])
    assert error["key"] == "selector_no_match"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]
    assert "guidance" in error


def test_eval_recorder_validation_returns_invalid_tool_input_for_bad_iso() -> None:
    validation = _validate_recorder_tool(
        GetHistoryTool("eval"),
        {"entity_ids": ["light.bedroom"], "start": "not-a-date"},
    )

    assert validation.data == {}
    assert validation.error is not None
    error = cast(dict[str, object], validation.error["error"])
    assert validation.error["status"] == "error"
    assert error["key"] == "invalid_tool_input"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]


async def test_recorder_window_too_large_returns_stable_error_key() -> None:
    snapshot = _scoped_snapshot()
    tool = GetHistoryTool("eval")
    data = cast(dict[str, object], tool.parameters({"entity_ids": ["light.bedroom"], "hours": 10_000}))

    result = await tool.run_query(snapshot, data, _recorder_source(snapshot))

    assert result["status"] == "error"
    error = cast(dict[str, object], result["error"])
    assert error["key"] == "time_window_too_large"
    assert isinstance(error["message"], str)
    assert error["message"] != error["key"]


async def test_eval_sql_query_can_filter_visible_entities_by_domain() -> None:
    result, _invoker_calls = await _run_execute(
        _case(),
        "result = await hass.query(\"select entity_id from states where domain = 'light' order by entity_id\")",
    )

    assert result["execution"] == {"status": "ok"}
    entity_ids = [row["entity_id"] for row in cast(list[dict[str, object]], result["output"])]
    assert entity_ids == ["light.bedroom", "light.living"]


async def _run_execute(case: EvalCase, code: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    fixture = get_home(case.home)
    snapshot = _scoped_snapshot()
    runtime = build_eval_runtime(
        case,
        baseline_candidate(),
        resolve_profile(DEFAULT_PROMPT_PROFILE),
        snapshot,
        fixture,
    )
    data = cast(dict[str, object], runtime.code_tool.parameters({"code": code}))

    result = cast(
        dict[str, object],
        await runtime.code_tool.run_execute(
            snapshot,
            data,
            llm.LLMContext("test", None, "en", None, None),
            runtime.runtime_context_factory(),
        ),
    )
    return result, runtime.invoker.calls


def _case() -> EvalCase:
    return EvalCase(
        id="tool-contract-unit",
        home="home_minimal",
        category="test",
        requests=(RequestVariant("canonical", "exercise production tool contract"),),
        required_actions=(RequiredAction("light", "turn_on", ("light.living",)),),
    )


def _scoped_snapshot(*, anchor_device_id: str | None = None) -> HomeSnapshot:
    snapshot = cast(HomeSnapshot, get_home("home_minimal").snapshot())
    return apply_scope(snapshot, EVAL_SCOPE, anchor_device_id=anchor_device_id)


def _recorder_source(snapshot: HomeSnapshot) -> RecorderSource:
    return build_fixture_recorder_source(snapshot, get_home("home_minimal"))
