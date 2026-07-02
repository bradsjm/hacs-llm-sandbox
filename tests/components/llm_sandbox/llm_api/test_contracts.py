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
    assert (
        "async def async_call(self, domain: str, service: str, service_data: Mapping[str, object] | None = None, blocking: bool = None"
        in MONTY_TYPE_STUBS
    )
    assert "def async_services_for_domain" in MONTY_TYPE_STUBS
    assert "def async_services_for_target" in MONTY_TYPE_STUBS
    assert "def supports_response" in MONTY_TYPE_STUBS
    assert "def async_get" in MONTY_TYPE_STUBS
    assert "def async_entries_for_label" in MONTY_TYPE_STUBS
    assert "def __getitem__" in MONTY_TYPE_STUBS


def test_monty_type_stubs_include_safe_config_surface() -> None:
    """Safe configuration objects are part of the LLM-facing API surface."""
    assert "class SafeConfig:" in MONTY_TYPE_STUBS
    assert "class SafeUnitSystem:" in MONTY_TYPE_STUBS


def test_monty_type_stubs_include_alias_fields() -> None:
    """Denormalized alias fields appear next to their canonical keys in stubs."""
    # SafeFloorEntry: floor_id followed by the id alias.
    assert "floor_id: str\n    id: str" in MONTY_TYPE_STUBS
    # SafeAreaEntry: id followed by the area_id alias.
    assert "id: str\n    area_id: str" in MONTY_TYPE_STUBS


def test_monty_type_stubs_include_datetime_facades() -> None:
    """Date/datetime facade and value classes are part of the LLM-facing API surface."""
    assert "class SafeDate:" in MONTY_TYPE_STUBS
    assert "class SafeDateTime:" in MONTY_TYPE_STUBS
    assert "class SafeDateFacade:" in MONTY_TYPE_STUBS
    assert "class SafeDateTimeFacade:" in MONTY_TYPE_STUBS
    assert "def today" in MONTY_TYPE_STUBS
    assert "def now" in MONTY_TYPE_STUBS
    assert "def utcnow" in MONTY_TYPE_STUBS
    assert "def fromisoformat" in MONTY_TYPE_STUBS
    assert "def isoformat" in MONTY_TYPE_STUBS


def test_monty_type_stubs_include_datetime_globals() -> None:
    """date and datetime appear as declared globals in the type stubs."""
    assert "\ndate: Any" in MONTY_TYPE_STUBS
    assert "\ndatetime: Any" in MONTY_TYPE_STUBS


def test_monty_type_stubs_include_entity_entry_domain() -> None:
    """The derived domain field appears on the entity entry record in stubs."""
    assert "class SafeRegistryEntry:\n    entity_id: str\n    domain: str" in MONTY_TYPE_STUBS
