"""Monty-backed code executor for the LLM Sandbox facade runtime."""

import asyncio

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import HomeAssistantError

from ..const import DOMAIN
from ..snapshot.models import (
    HomeSnapshot,
    SafeAreaEntry,
    SafeContext,
    SafeDeviceEntry,
    SafeFloorEntry,
    SafeRegistryEntry,
    SafeState,
)
from . import await_normalization, result_binding
from .contracts import MONTY_TYPE_STUBS
from .errors import CodeErrorPayload, HelperErrorPayload, HelperExecutionError, helper_error_from_exception
from .executor_support import (
    code_error_payload_for_state,
    helper_error_payload_for_state,
    json_safe,
    load_monty_factory,
    underlying_exception,
    validation_error,
)
from .facade_views import (
    SafeAreaModule,
    SafeAreaRegistry,
    SafeDeviceModule,
    SafeDeviceRegistry,
    SafeEntityModule,
    SafeEntityRegistry,
    SafeFloorModule,
    SafeFloorRegistry,
    SafeHass,
    SafeLLMContext,
    SafeServiceRegistry,
    SafeStateMachine,
    build_facades,
)
from .runtime import RuntimeContext, activate_runtime, clear_runtime

MAX_MONTY_CODE_CHARS = 8000
MONTY_MAX_ALLOCATIONS = 250_000
MONTY_MAX_MEMORY_BYTES = 64 * 1024 * 1024
MONTY_GC_INTERVAL = 1000
MONTY_MAX_RECURSION_DEPTH = 1000

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
    # Module facades (er/dr/ar/fr)
    SafeEntityModule,
    SafeDeviceModule,
    SafeAreaModule,
    SafeFloorModule,
    # LLM context
    SafeLLMContext,
    # Record types returned by facade methods
    SafeContext,
    SafeState,
    SafeRegistryEntry,
    SafeDeviceEntry,
    SafeAreaEntry,
    SafeFloorEntry,
]


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
    # fails open. Order matters: await normalization first (cheap, syntactic),
    # then last-expression promotion, then append_result_expression which
    # depends on knowing whether ``result`` is explicitly assigned.
    normalized_code, await_labels = await_normalization.normalize_awaits(code, DATACLASS_REGISTRY)
    promoted_code, promote_labels = result_binding.promote_last_expression_to_result(normalized_code)
    executable_code = result_binding.append_result_expression(promoted_code)
    runtime.state.normalizations = [*await_labels, *promote_labels]

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
        activate_runtime(runtime)
        limits: pydantic_monty.ResourceLimits = {
            "max_allocations": MONTY_MAX_ALLOCATIONS,
            "max_duration_secs": runtime.settings.execution_timeout_seconds,
            "max_memory": MONTY_MAX_MEMORY_BYTES,
            "gc_interval": MONTY_GC_INTERVAL,
            "max_recursion_depth": MONTY_MAX_RECURSION_DEPTH,
        }
        monty = monty_factory(
            executable_code,
            inputs=input_names,
            script_name="llm_sandbox_agent.py",
            type_check=True,
            type_check_stubs=MONTY_TYPE_STUBS,
            dataclass_registry=DATACLASS_REGISTRY,
        )
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
        return code_error_payload_for_state(
            kind=specific.__class__.__name__,
            message=str(specific) or str(err),
            state=runtime.state,
        )
    finally:
        # Capture print output even on success; CollectString.output is a
        # concatenation, so we strip the trailing newline of each print() call.
        _capture_printed()
        clear_runtime()

    result = json_safe(output)
    return {
        "execution": {
            "status": "ok",
            "helper_calls": runtime.state.helper_calls,
            "helper_call_limit": runtime.state.helper_call_limit,
            "normalizations": list(runtime.state.normalizations),
        },
        "output": result,
        "printed": list(runtime.state.printed),
        "proposed_actions": json_safe(runtime.state.proposed_actions),
    }
