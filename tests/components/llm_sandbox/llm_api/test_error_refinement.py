"""Tests for Monty error-forgiveness refinement."""

from custom_components.llm_sandbox.llm_api.executor_support import refine_code_error
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot, SafeConfig, SafeUnitSystem, SnapshotIndexes
import pytest


@pytest.fixture(name="snapshot")
def snapshot_fixture() -> HomeSnapshot:
    """Build the minimal frozen snapshot required by the guidance engine."""
    return HomeSnapshot(
        created_at="2026-06-29T00:00:00+00:00",
        states={},
        entities={},
        devices={},
        areas={},
        floors={},
        config=SafeConfig(
            location_name="Home",
            latitude=0.0,
            longitude=0.0,
            elevation=0,
            time_zone="UTC",
            language="en",
            country=None,
            currency="USD",
            internal_url=None,
            external_url=None,
            units=SafeUnitSystem(
                temperature_unit="°C",
                length_unit="m",
                mass_unit="kg",
                pressure_unit="Pa",
                volume_unit="L",
                area_unit="m²",
                wind_speed_unit="m/s",
                accumulated_precipitation_unit="mm",
            ),
        ),
        services={},
        services_supports_response={},
        indexes=SnapshotIndexes(
            entity_ids_by_device_id={},
            entity_ids_by_area_id={},
            device_ids_by_area_id={},
            entity_ids_by_config_entry_id={},
            entity_ids_by_label={},
            device_ids_by_label={},
            area_ids_by_floor_id={},
        ),
        labels={},
        categories={},
        issues=[],
        notifications=[],
        config_entries=[],
        services_schema={},
    )


def _candidate_ids(guidance: dict[str, object] | None) -> set[str]:
    """Return guidance candidate ids for concise behavior assertions."""
    assert guidance is not None
    candidates = guidance["candidates"]
    assert isinstance(candidates, list)
    return {str(candidate["id"]) for candidate in candidates if isinstance(candidate, dict)}


def _confidence(guidance: dict[str, object] | None) -> object:
    """Return serialized guidance confidence after proving guidance exists."""
    assert guidance is not None
    return guidance["confidence"]


def test_refine_strips_info_lines(snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "Exception",
        "plain error\ninfo: noisy line\ninfo: more",
        "x",
        snapshot,
    )

    assert refined_kind == "Exception"
    assert guidance is None
    assert "info:" not in refined_message
    assert "plain error" in refined_message


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("dir", id="dir"),
        pytest.param("vars", id="vars"),
    ],
)
def test_refine_reclassifies_disabled_discovery_builtin_with_attributes(name: str, snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyTypingError",
        f"unresolved-reference: name `{name}` is used when not defined",
        f"{name}(floor_registry)",
        snapshot,
    )

    assert refined_kind == "NameError"
    assert _candidate_ids(guidance) == {
        "async_get",
        "async_get_floor",
        "async_get_floor_by_name",
        "async_list_floors",
        "floors",
    }
    assert _confidence(guidance) != "none"
    assert name in refined_message


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("setattr", id="setattr"),
        pytest.param("delattr", id="delattr"),
    ],
)
def test_refine_reclassifies_mutation_builtin_without_attributes(name: str, snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyTypingError",
        f"unresolved-reference: name `{name}` is used when not defined",
        "x=1",
        snapshot,
    )

    assert refined_kind == "NameError"
    assert guidance is None
    assert name in refined_message


def test_refine_arbitrary_unknown_name_omits_misleading_candidates(snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyTypingError",
        "unresolved-reference: name `nope` is used when not defined",
        "nope()",
        snapshot,
    )

    assert refined_kind == "NameError"
    assert guidance is None
    assert "nope" in refined_message
    assert "Available sandbox globals include" in refined_message


def test_refine_plausible_global_typo_keeps_candidate_guidance(snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyTypingError",
        "unresolved-reference: name `statez` is used when not defined",
        "statez.async_all()",
        snapshot,
    )

    assert refined_kind == "NameError"
    assert "states" in _candidate_ids(guidance)
    assert _confidence(guidance) == "high"
    assert "statez" in refined_message


@pytest.mark.parametrize(
    ("class_name", "attr_code", "expected_attributes"),
    [
        pytest.param(
            "SafeFloorRegistry",
            "floor_registry.nope",
            [
                "async_get",
                "async_get_floor",
                "async_get_floor_by_name",
                "async_list_floors",
                "floors",
            ],
            id="facade-class",
        ),
        pytest.param(
            "SafeFloorEntry",
            "f.zzz",
            [
                "aliases",
                "created_at",
                "floor_id",
                "icon",
                "id",
                "level",
                "modified_at",
                "name",
            ],
            id="record-class-includes-alias",
        ),
        pytest.param(
            "SafeServiceRegistry",
            "hass.services.nope",
            [
                "async_call",
                "async_services",
                "async_services_for_domain",
                "async_services_for_target",
                "has_service",
                "services_schema",
                "services_supports_response",
            ],
            id="service-registry-facade",
        ),
        pytest.param(
            "SafeRegistryEntry",
            "entry.zzz",
            [
                "aliases",
                "area_id",
                "capabilities",
                "config_entry_id",
                "device_class",
                "device_id",
                "disabled_by",
            ],
            id="entity-entry-record-includes-domain",
        ),
    ],
)
def test_refine_attribute_error_surfaces_known_class_attributes(
    class_name: str,
    attr_code: str,
    expected_attributes: list[str],
    snapshot: HomeSnapshot,
) -> None:
    refined_kind, _refined_message, guidance = refine_code_error(
        "Exception",
        f"'{class_name}' object has no attribute 'zzz'",
        attr_code,
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert set(expected_attributes) <= _candidate_ids(guidance)
    assert _confidence(guidance) != "none"


def test_refine_attribute_error_unknown_class_has_no_surface(snapshot: HomeSnapshot) -> None:
    refined_kind, _refined_message, guidance = refine_code_error(
        "Exception",
        "'RandomThing' object has no attribute 'x'",
        "r.x",
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert guidance is None


@pytest.mark.parametrize(
    ("message", "module_name"),
    [
        pytest.param(
            "error[unresolved-import]: Cannot resolve imported module `statistics`",
            "statistics",
            id="statistics",
        ),
        pytest.param("ModuleNotFoundError: No module named 'collections'", "collections", id="collections"),
        pytest.param(
            "error[unresolved-import]: Cannot resolve imported module 'itertools'",
            "itertools",
            id="itertools",
        ),
    ],
)
def test_refine_unresolved_import_guides_to_builtins(message: str, module_name: str, snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error("Exception", message, "import x", snapshot)

    assert refined_kind == "ImportError"
    assert guidance is None
    assert refined_message != message
    assert module_name in refined_message


def test_refine_percent_format_guides_to_fstring(snapshot: HomeSnapshot) -> None:
    message = "unsupported operand type(s) for %: 'str' and 'int'"
    refined_kind, refined_message, _guidance = refine_code_error(
        "TypeError",
        message,
        "x = '%d' % 5",
        snapshot,
    )

    assert refined_kind == "TypeError"
    assert refined_message != message


def test_refine_str_format_guides_to_fstring(snapshot: HomeSnapshot) -> None:
    message = "'str' object has no attribute 'format'"
    refined_kind, refined_message, guidance = refine_code_error(
        "AttributeError",
        message,
        "'{}'.format(1)",
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert guidance is None
    assert refined_message != message


@pytest.mark.parametrize(
    ("class_name", "friendly"),
    [
        pytest.param("SafeFloorRegistry", "floor_registry", id="floor-registry"),
        pytest.param("SafeState", "state", id="state-record"),
        pytest.param("SafeServiceRegistry", "hass.services", id="services"),
        pytest.param("SafeEntityRegistry", "entity_registry", id="entity-registry"),
    ],
)
def test_refine_scrubs_internal_facade_class_names(class_name: str, friendly: str, snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, _guidance = refine_code_error(
        "Exception",
        f"'{class_name}' object has no attribute 'nope'",
        "x.nope",
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert class_name not in refined_message
    assert friendly in refined_message


def test_refine_parse_miss_returns_inputs_unchanged(snapshot: HomeSnapshot) -> None:
    refined_kind, refined_message, guidance = refine_code_error(
        "Exception",
        "something completely unrelated",
        "x",
        snapshot,
    )

    assert refined_kind == "Exception"
    assert guidance is None
    assert refined_message == "something completely unrelated"


def test_refine_import_hint_for_dunder_import(snapshot: HomeSnapshot) -> None:
    message = "unresolved-reference: name `__import__` is used when not defined"
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyTypingError",
        message,
        "json = __import__('json')",
        snapshot,
    )

    assert refined_kind == "NameError"
    assert guidance is None
    assert "__import__" in refined_message
    assert refined_message != message


def test_refine_collection_dict_method_hint(snapshot: HomeSnapshot) -> None:
    message = "'list' object has no attribute 'items'"
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyRuntimeError",
        message,
        "for eid, st in states.async_all().items(): ...",
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert guidance is None
    assert "list" in refined_message
    assert refined_message != message


def test_refine_none_deref_hint(snapshot: HomeSnapshot) -> None:
    message = "'NoneType' object has no attribute 'lower'"
    refined_kind, refined_message, guidance = refine_code_error(
        "MontyRuntimeError",
        message,
        "name = hass.states.get('light.x').name.lower()",
        snapshot,
    )

    assert refined_kind == "AttributeError"
    assert guidance is None
    assert "None" in refined_message
    assert refined_message != message
