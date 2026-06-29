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
                "services_supports_response",
                "supports_response",
            ],
            id="service-registry-facade",
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


def test_refine_parse_miss_returns_inputs_unchanged() -> None:
    refined_kind, refined_message, available_attributes = refine_code_error(
        "Exception",
        "something completely unrelated",
        "x",
    )

    assert refined_kind == "Exception"
    assert available_attributes is None
    assert refined_message == "something completely unrelated"
