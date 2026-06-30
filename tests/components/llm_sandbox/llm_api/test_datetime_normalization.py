"""Tests for datetime import normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.datetime_normalization import (
    DATETIME_IMPORTS_RESOLVED,
    normalize_datetime_imports,
)


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "from datetime import datetime\nresult = datetime.now()",
            "result = datetime.now()",
            {DATETIME_IMPORTS_RESOLVED},
            id="from-datetime-dropped",
        ),
        pytest.param(
            "from datetime import date\nresult = date.today()",
            "result = date.today()",
            {DATETIME_IMPORTS_RESOLVED},
            id="from-date-dropped",
        ),
        pytest.param(
            "from datetime import datetime as dt\nresult = dt.now()",
            "dt = datetime\nresult = dt.now()",
            {DATETIME_IMPORTS_RESOLVED},
            id="from-datetime-alias-assigned",
        ),
        pytest.param(
            "from datetime import date as d\nresult = d.today()",
            "d = date\nresult = d.today()",
            {DATETIME_IMPORTS_RESOLVED},
            id="from-date-alias-assigned",
        ),
        pytest.param(
            "import datetime\nresult = datetime.datetime.now()",
            "result = datetime.now()",
            {DATETIME_IMPORTS_RESOLVED},
            id="module-datetime-attribute-rewritten",
        ),
        pytest.param(
            "import datetime as dt\nresult = dt.datetime.now()",
            "result = datetime.now()",
            {DATETIME_IMPORTS_RESOLVED},
            id="module-alias-datetime-attribute-rewritten",
        ),
        pytest.param(
            "import datetime as dt\nresult = dt.date.today()",
            "result = date.today()",
            {DATETIME_IMPORTS_RESOLVED},
            id="module-alias-date-attribute-rewritten",
        ),
        pytest.param(
            "from datetime import timedelta\nresult = 1",
            "from datetime import timedelta\nresult = 1",
            set(),
            id="unsupported-from-import-left-alone",
        ),
        pytest.param(
            "from datetime import datetime, timedelta\nresult = 1",
            "from datetime import timedelta\nresult = 1",
            {DATETIME_IMPORTS_RESOLVED},
            id="mixed-from-import-keeps-unsupported-name",
        ),
        pytest.param(
            "from datetime import datetime as dt, timedelta",
            "dt = datetime\nfrom datetime import timedelta",
            {DATETIME_IMPORTS_RESOLVED},
            id="mixed-from-import-alias-before-kept-name",
        ),
        pytest.param(
            "import datetime, os",
            "import os",
            {DATETIME_IMPORTS_RESOLVED},
            id="mixed-import-drops-datetime-module",
        ),
        pytest.param(
            "result = datetime.now()",
            "result = datetime.now()",
            set(),
            id="bare-datetime-global-left-alone",
        ),
        pytest.param(
            "result = date.today()",
            "result = date.today()",
            set(),
            id="bare-date-global-left-alone",
        ),
        pytest.param(
            "from datetime import(",
            "from datetime import(",
            set(),
            id="syntax-error-fail-open",
        ),
        pytest.param(
            "import datetime\nresult = datetime.datetime.fromisoformat('x').year",
            "result = datetime.fromisoformat('x').year",
            {DATETIME_IMPORTS_RESOLVED},
            id="module-datetime-chain-rewritten",
        ),
        pytest.param(
            "import datetime as dt\nresult = dt.date.fromisoformat('x').year",
            "result = date.fromisoformat('x').year",
            {DATETIME_IMPORTS_RESOLVED},
            id="module-alias-date-chain-rewritten",
        ),
    ],
)
def test_normalize_datetime_imports(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = normalize_datetime_imports(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


def test_module_alias_shadowed_by_param_not_rewritten() -> None:
    """A module-import alias shadowed by a local binding is left intact.

    Conservative behavior: rewriting is skipped so user-code semantics are not
    silently changed; the real import is preserved and Monty surfaces a natural
    import error if executed.
    """
    code = "import datetime as dt\ndef f(dt):\n    return dt.datetime.now()\nresult = f(1)"
    normalized, labels = normalize_datetime_imports(code)
    assert labels == []
    assert "import datetime as dt" in normalized
    assert "dt.datetime" in normalized
