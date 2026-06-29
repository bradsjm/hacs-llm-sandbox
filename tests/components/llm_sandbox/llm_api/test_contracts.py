"""Tests for Monty-facing runtime contract generation."""

from custom_components.llm_sandbox.llm_api.contracts import MONTY_TYPE_STUBS


def test_monty_type_stubs_exclude_private_methods() -> None:
    """Private facade helpers are not part of the LLM-facing API surface."""
    assert "_require_state" not in MONTY_TYPE_STUBS
    private_methods = [
        line.strip()
        for line in MONTY_TYPE_STUBS.splitlines()
        if line.strip().startswith("def _") and not line.strip().startswith("def __")
    ]
    assert private_methods == []

    # Public methods and explicitly supported operator dunders remain exposed.
    assert "async def async_call" in MONTY_TYPE_STUBS
    assert "def async_get" in MONTY_TYPE_STUBS
    assert "def __getitem__" in MONTY_TYPE_STUBS
