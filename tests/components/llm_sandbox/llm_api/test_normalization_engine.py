"""Tests for unified AST normalization."""

import pytest
from custom_components.llm_sandbox.llm_api.normalization import rewrite
from custom_components.llm_sandbox.llm_api.normalization.rules.await_rules import (
    AWAITED_ASYNC_CALLS,
    STRIPPED_AWAIT_FROM_SYNC,
)
from custom_components.llm_sandbox.llm_api.normalization.rules.builtin_rules import (
    TYPE_NAME_RESOLVED,
    WRAPPED_NEXT_ITER,
)
from custom_components.llm_sandbox.llm_api.normalization.rules.datetime_rules import (
    DATETIME_IMPORTS_RESOLVED,
)
from custom_components.llm_sandbox.llm_api.normalization.rules.registry_import_rules import (
    REGISTRY_IMPORTS_RESOLVED,
)
from custom_components.llm_sandbox.llm_api.normalization.rules.service_target_rules import (
    REWROTE_POSITIONAL_SERVICE_TARGET,
)
from custom_components.llm_sandbox.llm_api.normalization.rules.state_sugar_rules import (
    REWROTE_SYNC_SUBSCRIPT,
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
        pytest.param(
            "import datetime as dt\ndef f(dt):\n    return dt.datetime.now()\nresult = f(1)",
            "import datetime as dt\ndef f(dt):\n    return dt.datetime.now()\nresult = f(1)",
            set(),
            id="module-alias-shadowed-by-param-left-alone",
        ),
    ],
)
def test_rewrite_datetime_imports(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "from homeassistant.helpers import entity_registry as er\nresult = er.async_get(hass)",
            "result = er.async_get(hass)",
            {REGISTRY_IMPORTS_RESOLVED},
            id="short-entity-registry-alias",
        ),
        pytest.param(
            "from homeassistant.helpers import floor_registry\nresult = floor_registry.async_list_floors()",
            "result = floor_registry.async_list_floors()",
            {REGISTRY_IMPORTS_RESOLVED},
            id="long-floor-registry-name",
        ),
        pytest.param(
            "from homeassistant.helpers import area_registry as ar, label_registry as lr\nresult = ar.async_list_areas()",
            "result = ar.async_list_areas()",
            {REGISTRY_IMPORTS_RESOLVED},
            id="multiple-supported-registry-imports",
        ),
    ],
)
def test_rewrite_supported_module_registry_imports(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "from homeassistant.helpers import entity_registry as er",
            "",
            {REGISTRY_IMPORTS_RESOLVED},
            id="direct-module-import-rewritten",
        ),
        pytest.param(
            "if enabled:\n    from homeassistant.helpers import entity_registry as er",
            "if enabled:\n    from homeassistant.helpers import entity_registry as er",
            set(),
            id="module-if-import-left-alone",
        ),
        pytest.param(
            "try:\n    from homeassistant.helpers import entity_registry as er\nexcept Exception:\n    pass",
            "try:\n    from homeassistant.helpers import entity_registry as er\nexcept Exception:\n    pass",
            set(),
            id="module-try-import-left-alone",
        ),
    ],
)
def test_rewrite_registry_imports_only_from_direct_module_body(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = type(floor_registry).__name__",
            "result = 'SafeFloorRegistry'",
            {TYPE_NAME_RESOLVED},
            id="type-name-resolved",
        ),
        pytest.param(
            "result = type(date).__name__",
            "result = 'SafeDateFacade'",
            {TYPE_NAME_RESOLVED},
            id="type-name-date",
        ),
        pytest.param(
            "result = type(datetime).__name__",
            "result = 'SafeDateTimeFacade'",
            {TYPE_NAME_RESOLVED},
            id="type-name-datetime",
        ),
        pytest.param(
            "result = type(floor_registry)",
            "result = type(floor_registry)",
            set(),
            id="bare-type-left-alone",
        ),
        pytest.param(
            "x = floor_registry\nresult = hasattr(x, 'async_list_floors')",
            "x = floor_registry\nresult = hasattr(x, 'async_list_floors')",
            set(),
            id="loop-variable-not-rewritten",
        ),
        pytest.param(
            "result = hasattr(floor_registry.async_list_floors(), 'name')",
            "result = hasattr(floor_registry.async_list_floors(), 'name')",
            set(),
            id="chain-root-not-rewritten",
        ),
        pytest.param(
            "result = hasattr(floor_registry, name)",
            "result = hasattr(floor_registry, name)",
            set(),
            id="dynamic-name-not-rewritten",
        ),
        pytest.param(
            "result = getattr(area_registry, 'missing')",
            "result = getattr(area_registry, 'missing')",
            set(),
            id="getattr-miss-no-default-left-alone",
        ),
        pytest.param(
            "result = hasattr(floor_registry,",
            "result = hasattr(floor_registry,",
            set(),
            id="syntax-error-fail-open",
        ),
        pytest.param(
            "result = next(xs)",
            "result = next(iter(xs))",
            {WRAPPED_NEXT_ITER},
            id="next-one-arg",
        ),
        pytest.param(
            "result = next(xs, default)",
            "result = next(iter(xs), default)",
            {WRAPPED_NEXT_ITER},
            id="next-default",
        ),
        pytest.param(
            "result = next(iter(xs))",
            "result = next(iter(xs))",
            set(),
            id="explicit-iter-unchanged",
        ),
    ],
)
def test_rewrite_builtins(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "result = hass.states.get('light.bedroom')",
            "result = hass.states.get('light.bedroom')",
            set(),
            id="sync-state-get-unchanged",
        ),
        pytest.param(
            "result = await hass.states.get('light.bedroom')",
            "result = hass.states.get('light.bedroom')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-get",
        ),
        pytest.param(
            "result = hass.services.async_call('light', 'turn_on')",
            "result = await hass.services.async_call('light', 'turn_on')",
            {AWAITED_ASYNC_CALLS},
            id="wrap-missing-await-on-async-call",
        ),
        pytest.param(
            "result = await hass.services.async_call('light', 'turn_on')",
            "result = await hass.services.async_call('light', 'turn_on')",
            set(),
            id="already-correct-await",
        ),
        pytest.param(
            "result = await hass.services.async_services_for_domain('light')",
            "result = hass.services.async_services_for_domain('light')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-domain-services",
        ),
        pytest.param(
            "result = await hass.services.supports_response('light', 'turn_on')",
            "result = hass.services.supports_response('light', 'turn_on')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-supports-response",
        ),
        pytest.param(
            "result = await er.async_get(hass)",
            "result = er.async_get(hass)",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-module-get",
        ),
        pytest.param(
            "result = await area_registry.async_get_area_by_name('Bedroom')",
            "result = area_registry.async_get_area_by_name('Bedroom')",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-sync-area-lookup",
        ),
        pytest.param(
            "result = await date.today()",
            "result = date.today()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-date-today",
        ),
        pytest.param(
            "result = await datetime.now()",
            "result = datetime.now()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-datetime-now",
        ),
        pytest.param(
            "result = await datetime.utcnow().isoformat()",
            "result = datetime.utcnow().isoformat()",
            {STRIPPED_AWAIT_FROM_SYNC},
            id="strip-await-from-datetime-utcnow-chain",
        ),
        pytest.param(
            "state = await hass.states.get('light.bedroom')\nresult = hass.services.async_call('light', 'turn_on')",
            "state = hass.states.get('light.bedroom')\nresult = await hass.services.async_call('light', 'turn_on')",
            {AWAITED_ASYNC_CALLS, STRIPPED_AWAIT_FROM_SYNC},
            id="wrap-and-strip",
        ),
        pytest.param(
            "result = states['light.bedroom']",
            "result = states.get('light.bedroom')",
            {REWROTE_SYNC_SUBSCRIPT},
            id="rewrite-states-subscript",
        ),
        pytest.param(
            "result = await hass.states['light.bedroom']",
            "result = hass.states.get('light.bedroom')",
            {REWROTE_SYNC_SUBSCRIPT, STRIPPED_AWAIT_FROM_SYNC},
            id="rewrite-and-strip-hass-states-subscript",
        ),
        pytest.param(
            "result = len(states)",
            "result = len(states.async_entity_ids())",
            {REWROTE_SYNC_SUBSCRIPT},
            id="rewrite-state-machine-len",
        ),
        pytest.param(
            "states = hass.states.async_all('lieght')\nresult = len(states)",
            "states = hass.states.async_all('lieght')\nresult = len(states)",
            set(),
            id="local-states-shadow-keeps-len",
        ),
        pytest.param(
            "states = {'light.bedroom': 'on'}\nresult = states['light.bedroom']",
            "states = {'light.bedroom': 'on'}\nresult = states['light.bedroom']",
            set(),
            id="local-states-shadow-keeps-subscript",
        ),
        pytest.param(
            "hass = obj\nresult = hass.states['light.bedroom']",
            "hass = obj\nresult = hass.states['light.bedroom']",
            set(),
            id="local-hass-shadow-keeps-subscript",
        ),
        pytest.param(
            "def f():\n    hass = object()\nresult = hass.services.async_call('light', 'turn_on')",
            "def f():\n    hass = object()\nresult = await hass.services.async_call('light', 'turn_on')",
            {AWAITED_ASYNC_CALLS},
            id="function-local-hass-shadow-does-not-leak",
        ),
        pytest.param(
            "result = x.get('foo')",
            "result = x.get('foo')",
            set(),
            id="local-variable-not-rooted",
        ),
        pytest.param(
            "result = hass.services.async_call(",
            "result = hass.services.async_call(",
            set(),
            id="syntax-error-fail-open",
        ),
        pytest.param(
            "result = hass.services.async_call('light', 'turn_on', {'brightness_pct': 80}, {'entity_id': 'light.bedroom'})",
            "result = await hass.services.async_call('light', 'turn_on', {'brightness_pct': 80}, target={'entity_id': 'light.bedroom'})",
            {REWROTE_POSITIONAL_SERVICE_TARGET, AWAITED_ASYNC_CALLS},
            id="mapping-fourth-argument-becomes-target",
        ),
        pytest.param(
            "result = hass.services.async_call('light', 'turn_on', {}, {'entity_id': 'light.bedroom'}, blocking=True)",
            "result = await hass.services.async_call('light', 'turn_on', {}, target={'entity_id': 'light.bedroom'}, blocking=True)",
            {REWROTE_POSITIONAL_SERVICE_TARGET, AWAITED_ASYNC_CALLS},
            id="mapping-fourth-argument-with-blocking-keyword",
        ),
    ],
)
def test_rewrite_awaits_rooted_facade_operations(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            "result = await hass.services.async_call(*args, 'light', 'turn_on', {'entity_id': 'light.bedroom'})",
            id="starred-argument",
        ),
        pytest.param(
            "result = await hass.services.async_call('light', 'turn_on', {}, {'entity_id': 'light.bedroom'}, **options)",
            id="keyword-unpacking",
        ),
    ],
)
def test_rewrite_positional_service_target_leaves_unpacking_alone(code: str) -> None:
    normalized, labels = rewrite(code)

    assert normalized == code
    assert set(labels) == set()


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("type = my_fn\nresult = type(states).__name__", id="rebound-type-not-resolved"),
        pytest.param("states = obj\nresult = type(states).__name__", id="shadowed-receiver-not-resolved"),
        pytest.param("next = my_fn\nresult = next(xs)", id="rebound-next-not-wrapped"),
        pytest.param("iter = my_fn\nresult = next(xs)", id="rebound-iter-not-wrapped"),
        pytest.param("er = something_local\nresult = await er.async_get(hass)", id="facade-root-shadow-not-awaited"),
        pytest.param("states = obj\nresult = await states.get('x')", id="stripper-respects-shadow"),
        pytest.param("datetime = 5\nfrom datetime import datetime\nresult = 1", id="from-import-drop-respects-shadow"),
        pytest.param(
            "vals = [(states := item) for item in items]\nresult = len(states)",
            id="walrus-in-comprehension-shadows-enclosing",
        ),
        pytest.param(
            "import datetime as dt\nimport json as dt\nresult = dt.datetime.now()",
            id="import-rebind-invalidates-datetime-alias",
        ),
        pytest.param(
            "import datetime as dt\nfrom json import dumps as dt\nresult = dt.datetime.now()",
            id="from-import-rebind-invalidates-datetime-alias",
        ),
        pytest.param(
            "from homeassistant.helpers import entity_registry as registry\nresult = registry.async_get(hass)",
            id="non-facade-registry-alias-left-alone",
        ),
        pytest.param(
            "def get_registry():\n    from homeassistant.helpers import entity_registry as er\n    return er.async_get(hass)",
            id="function-registry-import-left-alone",
        ),
        pytest.param(
            "er = object()\nfrom homeassistant.helpers import entity_registry as er\nresult = er.async_get(hass)",
            id="shadowed-registry-import-left-alone",
        ),
        pytest.param(
            "result = await hass.services.async_call('light', 'turn_on', {}, target={'entity_id': 'light.bedroom'})",
            id="existing-service-target-left-alone",
        ),
        pytest.param(
            "result = await hass.services.async_call('light', 'turn_on', {}, target_value)",
            id="dynamic-fourth-service-argument-left-alone",
        ),
    ],
)
def test_rewrite_shadow_fixes_do_not_rewrite(code: str) -> None:
    normalized, labels = rewrite(code)

    assert normalized == code
    assert set(labels) == set()


@pytest.mark.parametrize(
    ("code", "expected_code", "expected_labels"),
    [
        pytest.param(
            "ids = [states for states in groups]\nresult = len(states)",
            "ids = [states for states in groups]\nresult = len(states.async_entity_ids())",
            {REWROTE_SYNC_SUBSCRIPT},
            id="comprehension-target-no-leak",
        ),
    ],
)
def test_rewrite_positive_fix_adjacent_anchors(
    code: str,
    expected_code: str,
    expected_labels: set[str],
) -> None:
    normalized, labels = rewrite(code)

    assert normalized == expected_code
    assert set(labels) == expected_labels


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("from datetime import datetime\nresult = datetime.now()", id="datetime-from"),
        pytest.param("import datetime as dt\nresult = dt.datetime.now()", id="datetime-alias"),
        pytest.param("result = type(floor_registry).__name__", id="type-name"),
        pytest.param("result = next(xs)", id="next-iter"),
        pytest.param("result = states['light.bedroom']", id="state-subscript"),
        pytest.param("result = len(states)", id="state-len"),
        pytest.param("result = await hass.states.get('light.bedroom')", id="strip-await"),
        pytest.param("result = hass.services.async_call('light', 'turn_on')", id="insert-await"),
        pytest.param("states = obj\nresult = await states.get('x')", id="shadow-noop"),
        pytest.param("x = 1\nresult = x + 2", id="plain-noop"),
    ],
)
def test_rewrite_is_idempotent(code: str) -> None:
    once, _ = rewrite(code)
    twice, _ = rewrite(once)

    assert twice == once


def test_rewrite_noop_purity() -> None:
    code = "x = 1\nresult = x + 2"

    assert rewrite(code) == (code, ())


def test_rewrite_labels_follow_registry_order() -> None:
    code = "x = hass.services.async_call('light', 'turn_on')\nfrom datetime import datetime as dt"

    normalized, labels = rewrite(code)

    # Datetime rules are registered before await rules, so the datetime label
    # stays first even though the await rewrite occurs on an earlier source line.
    assert labels == ("datetime_imports_resolved", "awaited_async_calls")
    assert normalized == "x = await hass.services.async_call('light', 'turn_on')\ndt = datetime"
