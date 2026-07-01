"""Tests for Monty error-forgiveness refinement."""

import pytest
from custom_components.llm_sandbox.llm_api.executor_support import refine_code_error


def test_refine_strips_info_lines() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "Exception",
        "plain error\ninfo: noisy line\ninfo: more",
        "x",
    )

    assert refined_kind == "Exception"
    assert available_attributes is None
    assert "info:" not in refined_message
    assert "plain error" in refined_message


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("dir", id="dir"),
        pytest.param("vars", id="vars"),
    ],
)
def test_refine_reclassifies_disabled_discovery_builtin_with_attributes(name: str) -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyTypingError",
        f"unresolved-reference: name `{name}` is used when not defined",
        f"{name}(floor_registry)",
    )

    assert refined_kind == "NameError"
    assert available_attributes == [
        "async_get",
        "async_get_floor",
        "async_get_floor_by_name",
        "async_list_floors",
        "floors",
    ]
    assert name in refined_message


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("setattr", id="setattr"),
        pytest.param("delattr", id="delattr"),
    ],
)
def test_refine_reclassifies_mutation_builtin_without_attributes(name: str) -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyTypingError",
        f"unresolved-reference: name `{name}` is used when not defined",
        "x=1",
    )

    assert refined_kind == "NameError"
    assert available_attributes is None
    assert name in refined_message


def test_refine_unknown_name_keeps_message_no_attributes() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyTypingError",
        "unresolved-reference: name `nope` is used when not defined",
        "nope()",
    )

    assert refined_kind == "NameError"
    assert available_attributes is None
    assert "nope" in refined_message


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
                "has_service",
                "services",
                "services_schema",
                "services_supports_response",
                "supports_response",
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
                "domain",
                "entity_category",
                "entity_id",
                "has_entity_name",
                "hidden_by",
                "labels",
                "name",
                "original_device_class",
                "original_name",
                "platform",
                "supported_features",
                "translation_key",
                "unique_id",
            ],
            id="entity-entry-record-includes-domain",
        ),
    ],
)
def test_refine_attribute_error_surfaces_known_class_attributes(
    class_name: str,
    attr_code: str,
    expected_attributes: list[str],
) -> None:
    refined_kind, _refined_message, available_attributes = refine_code_error(
        "Exception",
        f"'{class_name}' object has no attribute 'zzz'",
        attr_code,
    )

    assert refined_kind == "AttributeError"
    assert available_attributes == expected_attributes


def test_refine_attribute_error_unknown_class_has_no_surface() -> None:
    refined_kind, _refined_message, available_attributes = refine_code_error(
        "Exception",
        "'RandomThing' object has no attribute 'x'",
        "r.x",
    )

    assert refined_kind == "AttributeError"
    assert available_attributes is None


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("error[unresolved-import]: Cannot resolve imported module `statistics`", id="statistics"),
        pytest.param("ModuleNotFoundError: No module named 'collections'", id="collections"),
        pytest.param("error[unresolved-import]: Cannot resolve imported module 'itertools'", id="itertools"),
    ],
)
def test_refine_unresolved_import_guides_to_builtins(message: str) -> None:
    refined_kind, refined_message, available_attributes = refine_code_error("Exception", message, "import x")

    assert refined_kind == "ImportError"
    assert available_attributes is None
    assert "json, math, re" in refined_message


def test_refine_percent_format_guides_to_fstring() -> None:
    refined_kind, refined_message, _available_attributes = refine_code_error(
        "TypeError",
        "unsupported operand type(s) for %: 'str' and 'int'",
        "x = '%d' % 5",
    )

    assert refined_kind == "TypeError"
    assert "f-string" in refined_message


def test_refine_str_format_guides_to_fstring() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "AttributeError",
        "'str' object has no attribute 'format'",
        "'{}'.format(1)",
    )

    assert refined_kind == "AttributeError"
    assert "f-string" in refined_message
    assert available_attributes is None


@pytest.mark.parametrize(
    ("class_name", "friendly"),
    [
        pytest.param("SafeFloorRegistry", "floor_registry", id="floor-registry"),
        pytest.param("SafeState", "state", id="state-record"),
        pytest.param("SafeServiceRegistry", "hass.services", id="services"),
        pytest.param("SafeEntityRegistry", "entity_registry", id="entity-registry"),
    ],
)
def test_refine_scrubs_internal_facade_class_names(class_name: str, friendly: str) -> None:
    refined_kind, refined_message, _available_attributes = refine_code_error(
        "Exception",
        f"'{class_name}' object has no attribute 'nope'",
        "x.nope",
    )

    assert refined_kind == "AttributeError"
    assert class_name not in refined_message
    assert friendly in refined_message


def test_refine_parse_miss_returns_inputs_unchanged() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "Exception",
        "something completely unrelated",
        "x",
    )

    assert refined_kind == "Exception"
    assert available_attributes is None
    assert refined_message == "something completely unrelated"


def test_refine_import_hint_for_dunder_import() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyTypingError",
        "unresolved-reference: name `__import__` is used when not defined",
        "json = __import__('json')",
    )

    assert refined_kind == "NameError"
    assert available_attributes is None
    assert "__import__" in refined_message
    assert "json" in refined_message


def test_refine_collection_dict_method_hint() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyRuntimeError",
        "'list' object has no attribute 'items'",
        "for eid, st in states.async_all().items(): ...",
    )

    assert refined_kind == "AttributeError"
    assert available_attributes is None
    assert "list" in refined_message
    assert "comprehension" in refined_message


def test_refine_none_deref_hint() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "MontyRuntimeError",
        "'NoneType' object has no attribute 'lower'",
        "name = hass.states.get('light.x').name.lower()",
    )

    assert refined_kind == "AttributeError"
    assert available_attributes is None
    assert "None" in refined_message
    assert "is not None" in refined_message
