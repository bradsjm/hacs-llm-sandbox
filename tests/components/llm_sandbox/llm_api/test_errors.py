"""Tests for shared LLM-facing recoverable error messages."""

import pytest
from custom_components.llm_sandbox.llm_api.errors import (
    setup_error_payload,
    tool_error_envelope,
    tool_error_message,
)
from custom_components.llm_sandbox.types import TranslationPlaceholders


@pytest.mark.parametrize(
    ("key", "placeholders", "tokens"),
    [
        pytest.param("monty_code_required", {}, ("code", "non-empty"), id="monty-code-required"),
        pytest.param("monty_code_too_long", {"max_length": "4000"}, ("4000", "Shorten"), id="monty-code-too-long"),
        pytest.param("unknown_config_entry", {"config_entry_id": "abc"}, ("abc", "unknown"), id="unknown-entry"),
        pytest.param("config_entry_not_loaded", {"config_entry_id": "abc"}, ("abc", "loaded"), id="entry-not-loaded"),
        pytest.param("invalid_tool_input", {"error": "bad field"}, ("bad field", "schema"), id="invalid-input"),
        pytest.param("time_window_too_large", {"max_hours": "168"}, ("168", "hours"), id="window-too-large"),
        pytest.param("recorder_unavailable", {}, ("recorder", "history"), id="recorder-unavailable"),
        pytest.param("logbook_unavailable", {}, ("get_logbook", "logbook"), id="logbook-unavailable"),
        pytest.param("query_failed", {"error": "TimeoutError"}, ("TimeoutError", "query"), id="query-failed"),
        pytest.param("invalid_cursor", {}, ("cursor", "restart"), id="invalid-cursor"),
        pytest.param(
            "analytics_unknown_op", {"op": "median", "valid": "count, sum"}, ("median", "count, sum"), id="bad-op"
        ),
        pytest.param(
            "analytics_unknown_group_key", {"group_key": "room", "valid": "domain"}, ("room", "domain"), id="bad-group"
        ),
        pytest.param(
            "analytics_bad_bucket", {"bucket": "7x", "examples": "15m, 1h"}, ("7x", "15m, 1h"), id="bad-bucket"
        ),
        pytest.param(
            "capture_failed",
            {"entity_id": "camera.front", "error": "TimeoutError"},
            ("camera.front", "TimeoutError"),
            id="capture-failed",
        ),
        pytest.param(
            "image_too_large",
            {"entity_id": "camera.front", "target_width": "640", "max_bytes": "1024"},
            ("camera.front", "640", "1024"),
            id="image-too-large",
        ),
        pytest.param("sql_too_long", {"max_length": "4000"}, ("4000", "SQL"), id="sql-too-long"),
        pytest.param("sql_timeout", {}, ("SQL", "timed out"), id="sql-timeout"),
    ],
)
def test_tool_error_message_resolves_known_keys(
    key: str,
    placeholders: TranslationPlaceholders,
    tokens: tuple[str, ...],
) -> None:
    """Known recoverable keys produce compact messages naming useful placeholders."""
    message = tool_error_message(key, placeholders)

    assert message is not None
    assert all(token in message for token in tokens)


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
    assert "Resolve the" not in str(error["message"])


def test_setup_error_payload_uses_resolver_placeholders() -> None:
    """Setup payloads include known setup placeholder values."""
    payload = setup_error_payload("unknown_config_entry", {"config_entry_id": "abc"})

    execution = payload["execution"]
    assert execution["status"] == "setup_error"
    assert "abc" in execution["message"]
