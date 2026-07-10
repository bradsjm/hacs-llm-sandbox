"""Tests for pure Monty executor support helpers."""

from typing import cast

import pytest
from custom_components.llm_sandbox.llm_api.errors import HelperExecutionError
from custom_components.llm_sandbox.llm_api.executor_support import (
    ExecutionState,
    error_key,
    error_placeholders,
    helper_error_payload_for_state,
    helper_response,
    json_safe,
    underlying_exception,
    validation_error,
)


class JsonHookObject:
    def __llm_sandbox_json__(self) -> dict[str, object]:
        return {"value": float("nan")}


class WrappedError(Exception):
    def __init__(self, inner: Exception | object) -> None:
        super().__init__("wrapped")
        self._inner = inner

    def exception(self) -> Exception | object:
        return self._inner


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(float("nan"), "nan", id="nan"),
        pytest.param(float("inf"), "inf", id="inf"),
        pytest.param(float("-inf"), "-inf", id="negative-inf"),
        pytest.param({1: "one", "two": 2}, {"1": "one", "two": 2}, id="stringified-dict-keys"),
        pytest.param(("a", 1), ["a", 1], id="tuple-to-list"),
        pytest.param(JsonHookObject(), {"value": "nan"}, id="sandbox-json-hook"),
    ],
)
def test_json_safe_known_shapes(value: object, expected: object) -> None:
    assert json_safe(value) == expected


def test_json_safe_set_contents_and_unknown_object_stringification() -> None:
    assert set(cast(list[object], json_safe({"a", "b"}))) == {"a", "b"}
    assert json_safe(object()).startswith("<object object at ")


async def test_helper_response_returns_json_safe_sync_value() -> None:
    state = ExecutionState(service_call_limit=2)

    value = await helper_response(state, "helper.sync", lambda: {1: {"a", "b"}})

    assert value["1"] in (["a", "b"], ["b", "a"])
    assert state.dispatched_service_calls == 0
    assert state.last_helper_error is None


async def test_helper_response_maps_service_validation_and_clears_before_later_success() -> None:
    state = ExecutionState(service_call_limit=3)
    service_error = validation_error("invalid_tool_input", {"field": "code"})

    with pytest.raises(HelperExecutionError) as err:
        await helper_response(state, "helper.invalid", lambda: (_ for _ in ()).throw(service_error))

    assert err.value.key == "invalid_tool_input"
    assert err.value.placeholders == {"field": "code"}
    assert state.last_helper_error is err.value

    assert await helper_response(state, "helper.ok", lambda: "ok") == "ok"
    assert state.last_helper_error is None


def test_underlying_exception_unwraps_monty_style_wrapper() -> None:
    inner = ValueError("bad")

    assert underlying_exception(WrappedError(inner)) is inner
    assert underlying_exception(WrappedError("not an exception")).__class__ is WrappedError


def test_error_key_and_placeholders_extract_stable_fields() -> None:
    err = validation_error("invalid_tool_input", {"field": "code", "limit": "10"})

    assert error_key(err) == "invalid_tool_input"
    assert error_placeholders(err) == {"field": "code", "limit": "10"}


@pytest.mark.parametrize(
    ("err", "tokens"),
    [
        pytest.param(
            HelperExecutionError("query", "sql_too_long", {"max_length": "4000"}),
            ("SQL", "4000"),
            id="sql-too-long",
        ),
        pytest.param(HelperExecutionError("query", "sql_timeout", {}), ("SQL", "timed out"), id="sql-timeout"),
        pytest.param(
            HelperExecutionError("query", "sql_unknown", {"reason": "bad column"}),
            ("bad column", "query"),
            id="reason-first",
        ),
    ],
)
def test_helper_error_payload_messages_are_specific(
    err: HelperExecutionError,
    tokens: tuple[str, ...],
) -> None:
    """Helper payloads use shared specific messages while preserving reason overrides."""
    payload = helper_error_payload_for_state(err, ExecutionState(service_call_limit=2))

    message = payload["execution"]["message"]
    assert all(token in message for token in tokens)
