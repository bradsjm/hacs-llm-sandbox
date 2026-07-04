"""Tests for Monty result binding helpers."""

import pytest
from custom_components.llm_sandbox.llm_api.result_binding import (
    PROMOTED_LAST_EXPRESSION,
    RESULT_NONE_DEFAULT,
    append_result_expression,
    promote_last_expression_to_result,
)


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param("result = 1", "result = 1\nresult", [], id="module-assignment"),
        pytest.param(
            "if ok:\n    result = 1",
            "result = None\nif ok:\n    result = 1\nresult",
            [RESULT_NONE_DEFAULT],
            id="if-assignment-defaults-then-appends",
        ),
        pytest.param(
            "for item in items:\n    result = item",
            "result = None\nfor item in items:\n    result = item\nresult",
            [RESULT_NONE_DEFAULT],
            id="for-assignment-defaults-then-appends",
        ),
        pytest.param(
            "try:\n    result = 1\nexcept Exception:\n    pass",
            "result = None\ntry:\n    result = 1\nexcept Exception:\n    pass\nresult",
            [RESULT_NONE_DEFAULT],
            id="try-assignment-defaults-then-appends",
        ),
        pytest.param("def fn():\n    result = 1", "def fn():\n    result = 1", [], id="function-assignment-ignored"),
        pytest.param("class C:\n    result = 1", "class C:\n    result = 1", [], id="class-assignment-ignored"),
        pytest.param("import math", "import math", [], id="import-only"),
        pytest.param("value = 1", "value = 1", [], id="other-assignment"),
    ],
)
def test_append_result_expression_only_when_module_scope_result_is_assigned(
    code: str, expected_code: str, expected_labels: list[str]
) -> None:
    assert append_result_expression(code) == (expected_code, expected_labels)


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param("1 + 2", "result = 1 + 2", [PROMOTED_LAST_EXPRESSION], id="bare-expression"),
        pytest.param(
            "value = 1\nvalue + 2",
            "value = 1\nresult = value + 2",
            [PROMOTED_LAST_EXPRESSION],
            id="trailing-expression",
        ),
        pytest.param(
            "if c:\n    result = 1\n2",
            "if c:\n    result = 1\nresult = 2",
            [PROMOTED_LAST_EXPRESSION],
            id="conditional-result-promotes-trailing",
        ),
        pytest.param("result = 1\n2", "result = 1\n2", [], id="explicit-result-suppresses"),
        pytest.param("value = 1", "value = 1", [], id="trailing-assignment"),
        pytest.param("if ok:\n    value = 1", "if ok:\n    value = 1", [], id="compound-block-not-promoted"),
        pytest.param("def fn():\n    return 1", "def fn():\n    return 1", [], id="function-def"),
        pytest.param("result = ", "result = ", [], id="syntax-error-fail-open"),
    ],
)
def test_promote_last_expression_to_result(code: str, expected_code: str, expected_labels: list[str]) -> None:
    promoted, labels = promote_last_expression_to_result(code)

    assert promoted == expected_code
    assert labels == expected_labels
