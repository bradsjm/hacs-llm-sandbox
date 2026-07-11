"""Tests for Monty-facing runtime contract generation."""

import re
import subprocess
import sys

from custom_components.llm_sandbox.llm_api.contracts import MONTY_TYPE_STUBS
import pytest


def test_facades_and_executor_import_in_fresh_process() -> None:
    """Direct imports do not depend on package-level tool import order."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import custom_components.llm_sandbox.llm_api.facades; "
            "import custom_components.llm_sandbox.llm_api.executor",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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
    assert "def __getitem__" in MONTY_TYPE_STUBS
    assert "def get" in MONTY_TYPE_STUBS
    assert "def keys" in MONTY_TYPE_STUBS
    assert "def items" in MONTY_TYPE_STUBS
    assert "def values" in MONTY_TYPE_STUBS


@pytest.mark.parametrize(
    "symbol",
    [
        pytest.param("def async_services_for_domain", id="services-for-domain"),
        pytest.param("def async_services_for_target", id="services-for-target"),
        pytest.param("def supports_response", id="supports-response"),
        pytest.param("def async_get", id="async-get"),
        pytest.param("def async_entries_for_label", id="async-entries-for-label"),
        pytest.param("async def history", id="history"),
        pytest.param("async def logbook", id="logbook"),
        pytest.param("async def query", id="query"),
        pytest.param("class SafeConfig:", id="safe-config"),
        pytest.param("class SafeUnitSystem:", id="safe-unit-system"),
        pytest.param("class SafeDate:", id="safe-date"),
        pytest.param("class SafeDateTime:", id="safe-datetime"),
        pytest.param("class SafeDateFacade:", id="safe-date-facade"),
        pytest.param("class SafeDateTimeFacade:", id="safe-datetime-facade"),
        pytest.param("def today", id="today"),
        pytest.param("def now", id="now"),
        pytest.param("def utcnow", id="utcnow"),
        pytest.param("def fromisoformat", id="fromisoformat"),
        pytest.param("def isoformat", id="isoformat"),
        pytest.param("\ndate: Any", id="date-global"),
        pytest.param("\ndatetime: Any", id="datetime-global"),
        pytest.param("class SafeRegistryEntry:", id="registry-entry"),
        pytest.param("    domain: str", id="registry-entry-domain"),
    ],
)
def test_monty_type_stubs_expose_required_surface(symbol: str) -> None:
    """The LLM-facing stub surface keeps required public types and methods."""
    assert symbol in MONTY_TYPE_STUBS


def test_monty_type_stubs_expose_bounded_logbook_signature() -> None:
    """The generated facade contract exposes only the public logbook inputs."""
    assert re.search(
        r"async def logbook\(self, entity_ids: str \| list\[str\] \| None = None, hours: float \| None = None\)",
        MONTY_TYPE_STUBS,
    )


def test_monty_type_stubs_include_alias_fields() -> None:
    """Denormalized alias fields appear next to their canonical keys in stubs."""
    assert re.search(r"class SafeFloorEntry:[\s\S]*\bfloor_id: str\b[\s\S]*\bid: str\b", MONTY_TYPE_STUBS)
    assert re.search(r"class SafeAreaEntry:[\s\S]*\bid: str\b[\s\S]*\barea_id: str\b", MONTY_TYPE_STUBS)


@pytest.mark.parametrize("class_name", ["SafeState", "SafeLLMContext"])
def test_monty_type_stubs_expose_read_only_mapping_surface(class_name: str) -> None:
    """Snapshot records and llm_context expose reads but no mapping mutation methods."""
    class_stub = re.search(rf"class {class_name}:[\s\S]*?(?=\nclass |\Z)", MONTY_TYPE_STUBS)

    assert class_stub is not None
    assert all(f"def {method}" in class_stub.group() for method in ("get", "keys", "items", "values"))
    assert "def __setitem__" not in class_stub.group()
    assert "def update" not in class_stub.group()
