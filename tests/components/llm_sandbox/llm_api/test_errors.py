"""Tests for shared LLM-facing recoverable error messages."""

from custom_components.llm_sandbox.llm_api.errors import (
    setup_error_payload,
    tool_error_envelope,
    tool_error_message,
)
from custom_components.llm_sandbox.types import TranslationPlaceholders
import pytest


@pytest.mark.parametrize(
    ("key", "placeholders"),
    [
        pytest.param("monty_code_required", {}, id="monty-code-required"),
        pytest.param("monty_code_too_long", {"max_length": "4000"}, id="monty-code-too-long"),
        pytest.param("unknown_config_entry", {"config_entry_id": "abc"}, id="unknown-entry"),
        pytest.param("config_entry_not_loaded", {"config_entry_id": "abc"}, id="entry-not-loaded"),
        pytest.param("invalid_tool_input", {"error": "bad field"}, id="invalid-input"),
        pytest.param("time_window_too_large", {"max_hours": "168"}, id="window-too-large"),
        pytest.param("recorder_unavailable", {}, id="recorder-unavailable"),
        pytest.param("logbook_unavailable", {}, id="logbook-unavailable"),
        pytest.param("authorization_denied", {}, id="authorization-denied"),
        pytest.param("automation_unavailable", {}, id="automation-unavailable"),
        pytest.param("automation_content_unavailable", {}, id="automation-content-unavailable"),
        pytest.param("query_failed", {"error": "TimeoutError"}, id="query-failed"),
        pytest.param("invalid_cursor", {}, id="invalid-cursor"),
        pytest.param(
            "analytics_unknown_op",
            {"op": "median", "valid": "count, sum"},
            id="bad-op",
        ),
        pytest.param(
            "analytics_unknown_group_key",
            {"group_key": "room", "valid": "domain"},
            id="bad-group",
        ),
        pytest.param(
            "analytics_bad_bucket",
            {"bucket": "7x", "examples": "15m, 1h"},
            id="bad-bucket",
        ),
        pytest.param(
            "capture_failed",
            {"entity_id": "camera.front", "error": "TimeoutError"},
            id="capture-failed",
        ),
        pytest.param(
            "image_too_large",
            {"entity_id": "camera.front", "target_width": "640", "max_bytes": "1024"},
            id="image-too-large",
        ),
        pytest.param("sql_too_long", {"max_length": "4000"}, id="sql-too-long"),
        pytest.param("sql_timeout", {}, id="sql-timeout"),
    ],
)
def test_tool_error_message_resolves_known_keys(
    key: str,
    placeholders: TranslationPlaceholders,
) -> None:
    """Known recoverable keys resolve and interpolate their structured placeholders."""
    message = tool_error_message(key, placeholders)

    assert message is not None
    assert all(value in message for value in placeholders.values())


def test_tool_error_message_unknown_key_returns_none() -> None:
    """Unknown keys still defer to the caller's generic fallback."""
    assert tool_error_message("unknown_key", {}) is None


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("invalid_tool_input", id="invalid-input"),
        pytest.param("image_too_large", id="image-too-large"),
        pytest.param("analytics_unknown_op", id="analytics-op"),
    ],
)
def test_tool_error_message_empty_placeholders_do_not_raise(key: str) -> None:
    """Resolver defaults keep empty placeholder maps safe."""
    assert tool_error_message(key, {})


def test_tool_error_envelope_uses_resolver_before_generic_fallback() -> None:
    """The shared tool envelope exposes placeholder-aware messages by default."""
    payload = tool_error_envelope("time_window_too_large", {"max_hours": "168"})

    error = payload["error"]
    assert error["key"] == "time_window_too_large"
    assert "168" in str(error["message"])


def test_setup_error_payload_uses_resolver_placeholders() -> None:
    """Setup payloads include known setup placeholder values."""
    payload = setup_error_payload("unknown_config_entry", {"config_entry_id": "abc"})

    execution = payload["execution"]
    assert execution["status"] == "setup_error"
    assert "abc" in execution["message"]
