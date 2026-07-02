"""Monty-backed code executor for the LLM Sandbox facade runtime."""

import ast
import asyncio

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import HomeAssistantError

from ..const import DOMAIN
from ..snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeCategoryEntry,
    SafeConfig,
    SafeConfigEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeIssueEntry,
    SafeLabelEntry,
    SafeNotificationEntry,
    SafeRegistryEntry,
    SafeState,
    SafeUnitSystem,
)
from . import await_normalization, result_binding
from .builtin_normalization import normalize_builtins
from .contracts import MONTY_TYPE_STUBS
from .datetime_normalization import normalize_datetime_imports
from .errors import CodeErrorPayload, HelperErrorPayload, HelperExecutionError, helper_error_from_exception
from .executor_support import (
    ExecutionState,
    MontyFactory,
    MontyRunner,
    code_error_payload_for_state,
    helper_error_payload_for_state,
    json_safe,
    load_monty_factory,
    refine_code_error,
    underlying_exception,
    validation_error,
)
from .facade_views import (
    SafeAreaRegistry,
    SafeCategoryRegistry,
    SafeConfigEntries,
    SafeDate,
    SafeDateFacade,
    SafeDateTime,
    SafeDateTimeFacade,
    SafeDeviceRegistry,
    SafeEntityRegistry,
    SafeFloorRegistry,
    SafeHass,
    SafeIssueRegistry,
    SafeLabelRegistry,
    SafeLLMContext,
    SafeNotificationRegistry,
    SafeServiceRegistry,
    SafeStateMachine,
    build_facades,
)
from .resolution import available_hint
from .runtime import RuntimeContext, activate_runtime, clear_runtime

MAX_MONTY_CODE_CHARS = 8000
MONTY_MAX_ALLOCATIONS = 250_000
MONTY_MAX_MEMORY_BYTES = 64 * 1024 * 1024
MONTY_GC_INTERVAL = 1000
MONTY_MAX_RECURSION_DEPTH = 1000
_REFERENCE_TYPE_ERRORS = ("unresolved-reference", "used-when-not-defined")

# View classes registered with Monty for type-checking and dataclass binding.
# Also reused by ``await_normalization`` to derive async/sync method names.
# Every facade and record type the LLM can receive from a method call must be
# registered so Monty knows how to type-check attribute access on it.
DATACLASS_REGISTRY: list[type] = [
    # Root + state machine + service registry
    SafeHass,
    SafeStateMachine,
    SafeServiceRegistry,
    # Registry instance facades
    SafeEntityRegistry,
    SafeDeviceRegistry,
    SafeAreaRegistry,
    SafeFloorRegistry,
    SafeLabelRegistry,
    SafeCategoryRegistry,
    SafeIssueRegistry,
    SafeNotificationRegistry,
    SafeConfigEntries,
    # LLM context
    SafeLLMContext,
    # Date/datetime facades (value types + class facades)
    SafeDate,
    SafeDateTime,
    SafeDateFacade,
    SafeDateTimeFacade,
    # Record types returned by facade methods
    SafeContext,
    SafeState,
    SafeConfig,
    SafeUnitSystem,
    SafeRegistryEntry,
    SafeDeviceEntry,
    SafeAreaEntry,
    SafeFloorEntry,
    SafeLabelEntry,
    SafeCategoryEntry,
    SafeIssueEntry,
    SafeNotificationEntry,
    SafeConfigEntry,
]


def _build_monty(
    monty_factory: MontyFactory,
    code: str,
    input_names: list[str],
    state: ExecutionState,
) -> MontyRunner:
    """Construct the Monty program, failing open on non-reference type errors.

    Reference errors (undefined names) are surfaced so they refine into a
    familiar NameError. Other type-check strictness the runtime accepts
    (e.g. ``invalid-assignment`` from heterogeneous dict seeding) is relaxed
    by rebuilding with ``type_check=False`` and recording ``type_check_relaxed``.
    """
    try:
        return monty_factory(
            code,
            inputs=input_names,
            script_name="llm_sandbox_agent.py",
            type_check=True,
            type_check_stubs=MONTY_TYPE_STUBS,
            dataclass_registry=DATACLASS_REGISTRY,
        )
    except Exception as err:
        message = str(err)
        # Reference errors (undefined names) and non-diagnostic construction
        # failures (e.g. SyntaxError) must surface, not be silently relaxed.
        # Diagnostic tokens track pydantic-monty==0.0.18; this branch is
        # fail-open for other type-check diagnostics the runtime can tolerate.
        if any(token in message for token in _REFERENCE_TYPE_ERRORS) or "error[" not in message:
            raise
        # Type-check strictness the runtime tolerates (e.g. invalid-assignment
        # from heterogeneous dict seeding): relax and continue.
        state.normalizations.append("type_check_relaxed")
        return monty_factory(
            code,
            inputs=input_names,
            script_name="llm_sandbox_agent.py",
            type_check=False,
            type_check_stubs=MONTY_TYPE_STUBS,
            dataclass_registry=DATACLASS_REGISTRY,
        )


def _is_empty_output(result: object) -> bool:
    """True for ``None`` or an empty collection, but not scalar 0/False outputs."""
    if result is None:
        return True
    if isinstance(result, list | dict | tuple | str | set):
        return len(result) == 0
    return False


def _states_root(node: ast.AST) -> bool:
    """Whether ``node`` is the ``states`` or ``hass.states`` expression."""
    if isinstance(node, ast.Name):
        return node.id == "states"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "states"
        and isinstance(node.value, ast.Name)
        and node.value.id == "hass"
    )


def _referenced_missing(code: str, snapshot: HomeSnapshot) -> list[str]:
    """Entity ids read via literal ``states.get``/``states[...]`` that are absent.

    Static analysis: Monty copies input objects and does not propagate the
    runtime contextvar to synchronous methods, so absent reads cannot be
    recorded inside ``states.get``. Scanning the submitted code for literal
    entity-id reads catches the dominant case (an LLM-typed integration id) and
    lets the executor self-describe available entities on an empty result.
    """
    try:
        module = ast.parse(code)
    except SyntaxError:
        return []
    missing: list[str] = []
    seen: set[str] = set()

    def _consider(literal: object) -> None:
        if isinstance(literal, str) and literal not in seen and literal not in snapshot.states:
            seen.add(literal)
            missing.append(literal)

    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and _states_root(node.func.value) and node.args:
                _consider(_literal_str(node.args[0]))
        elif isinstance(node, ast.Subscript) and _states_root(node.value):
            _consider(_literal_str(node.slice))
    return missing


def _literal_str(node: ast.AST) -> object:
    """Return the Python value of a string-literal AST node, else a sentinel."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return object()  # non-string / non-literal: never matches an entity id


async def async_execute_home_code(
    code: str,
    *,
    snapshot: HomeSnapshot,
    llm_context: SafeLLMContext,
    runtime: RuntimeContext,
) -> dict[str, object] | HelperErrorPayload | CodeErrorPayload:
    """Run bounded Monty code against the LLM Sandbox facade runtime."""
    if not code.strip():
        raise validation_error("monty_code_required", {})
    if len(code) > MAX_MONTY_CODE_CHARS:
        raise validation_error("monty_code_too_long", {"max_length": str(MAX_MONTY_CODE_CHARS)})

    try:
        monty_factory = load_monty_factory()
    except ImportError as err:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="monty_execution_failed",
            translation_placeholders={"error": err.__class__.__name__},
        ) from err

    # Forgiveness pipeline (Postel's law): each pass is independent and
    # fails open. Order matters: datetime import normalization first resolves
    # supported datetime/date imports to sandbox facades, then builtin
    # normalization resolves safe reflection syntax, then await normalization,
    # then last-expression promotion before append_result_expression checks
    # explicit ``result``.
    datetime_code, datetime_labels = normalize_datetime_imports(code)
    builtin_code, builtin_labels = normalize_builtins(datetime_code)
    normalized_code, await_labels = await_normalization.normalize_awaits(builtin_code, DATACLASS_REGISTRY)
    promoted_code, promote_labels = result_binding.promote_last_expression_to_result(normalized_code)
    executable_code = result_binding.append_result_expression(promoted_code)
    runtime.state.normalizations = [*datetime_labels, *builtin_labels, *await_labels, *promote_labels]

    # Build facade globals from the snapshot.
    facade_inputs = build_facades(snapshot)
    facade_inputs["llm_context"] = llm_context
    input_names = list(facade_inputs.keys())

    # CollectString exposes one entry per print() call via .output (concatenated).
    # We split on newlines to give the LLM one printable line per list entry.
    print_collector = pydantic_monty.CollectString()

    def _capture_printed() -> None:
        """Persist captured print() lines before building any response payload."""
        runtime.state.printed = [line for line in print_collector.output.splitlines() if line]

    try:
        activate_runtime(runtime, snapshot)
        limits: pydantic_monty.ResourceLimits = {
            "max_allocations": MONTY_MAX_ALLOCATIONS,
            "max_duration_secs": runtime.settings.execution_timeout_seconds,
            "max_memory": MONTY_MAX_MEMORY_BYTES,
            "gc_interval": MONTY_GC_INTERVAL,
            "max_recursion_depth": MONTY_MAX_RECURSION_DEPTH,
        }
        monty = _build_monty(monty_factory, executable_code, input_names, runtime.state)
        output = await asyncio.wait_for(
            monty.run_async(
                inputs=facade_inputs,
                limits=limits,
                external_functions={},
                print_callback=print_collector,
            ),
            timeout=runtime.settings.execution_timeout_seconds,
        )
    except TimeoutError:
        _capture_printed()
        return code_error_payload_for_state(
            kind="TimeoutError",
            message=f"Code execution timed out after {runtime.settings.execution_timeout_seconds} seconds.",
            state=runtime.state,
        )
    except HelperExecutionError as err:
        _capture_printed()
        return helper_error_payload_for_state(err, runtime.state)
    except Exception as err:  # noqa: BLE001
        # Monty wraps input-object method exceptions into MontyRuntimeError
        # without preserving __cause__, so helper_error_from_exception cannot
        # recover the structured error from the exception chain. Fall back to
        # the per-run state where helper_response stored the last error, but
        # only when Monty's generic wrapper carries that exact opaque marker.
        _capture_printed()
        if helper_error := helper_error_from_exception(err):
            return helper_error_payload_for_state(helper_error, runtime.state)
        if (candidate := runtime.state.last_helper_error) is not None:
            specific = underlying_exception(err)
            if specific.__class__ is Exception and specific.args == (candidate.marker,):
                return helper_error_payload_for_state(candidate, runtime.state)
        specific = underlying_exception(err)
        refined_kind, refined_message, available_attributes = refine_code_error(
            specific.__class__.__name__, str(specific) or str(err), code
        )
        return code_error_payload_for_state(
            kind=refined_kind,
            message=refined_message,
            state=runtime.state,
            available_attributes=available_attributes,
        )
    finally:
        # Capture print output even on success; CollectString.output is a
        # concatenation, so we strip the trailing newline of each print() call.
        _capture_printed()
        clear_runtime()

    result = json_safe(output)
    # Static scan for literal entity-id reads that are absent from the snapshot.
    # An unguessable integration-specific id often surfaces as an empty result;
    # when that happens, name the visible entities in the first referenced
    # domain so the next call can recover. (Monty copies inputs and does not
    # propagate the runtime contextvar to sync methods, so this is a code scan,
    # not a runtime record.)
    referenced_missing = _referenced_missing(code, snapshot)
    execution_payload: dict[str, object] = {
        "status": "ok",
        "helper_calls": runtime.state.helper_calls,
        "helper_call_limit": runtime.state.helper_call_limit,
        "adjustments": list(runtime.state.adjustments),
        "referenced_missing": referenced_missing,
    }
    if referenced_missing and _is_empty_output(result):
        domain = referenced_missing[0].split(".", 1)[0] if "." in referenced_missing[0] else referenced_missing[0]
        execution_payload["available_hint"] = available_hint(snapshot, domain)
    return {
        "execution": execution_payload,
        "output": result,
        "printed": list(runtime.state.printed),
        "actions": json_safe(runtime.state.actions),
    }
