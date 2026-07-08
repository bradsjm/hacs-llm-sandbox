"""Monty-backed code executor for the LLM Sandbox facade runtime."""

import asyncio
import re
from collections.abc import Sequence

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import HomeAssistantError

from ..const import DOMAIN
from ..snapshot.models import HomeSnapshot
from .contracts import MONTY_TYPE_STUBS
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
from .facade_registry import MONTY_DATACLASS_REGISTRY
from .facades import SafeLLMContext, build_facades
from .guidance import FailureContext, Intent, advise
from .legacy_notes import (
    LegacyNoteContext,
    _referenced_missing,
    _referenced_visible_state_id,
    compute_legacy_note,
)
from .literal_resolution import substitute_remembered_literals
from .normalization import await_normalization, result_binding
from .normalization.builtin_normalization import normalize_builtins
from .normalization.datetime_normalization import normalize_datetime_imports
from .sandbox_context import RuntimeContext, activate_runtime, clear_runtime

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
DATACLASS_REGISTRY = MONTY_DATACLASS_REGISTRY


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


def _strip_monty_diagnostic(message: str) -> str:
    """Remove Monty code-frame lines and filenames from an executor-surfaced message."""
    lines: list[str] = []
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped or "llm_sandbox_agent.py" in stripped or stripped.startswith("-->"):
            continue
        if set(stripped) <= {"^", "|", " ", "-"}:
            continue
        lines.append(stripped.replace("llm_sandbox_agent.py", ""))
    return " ".join(" ".join(lines).split()) or "Code execution failed."


def _read_path_fix(
    kind: str,
    message: str,
    code: str,
    snapshot: HomeSnapshot,
    fallback_guidance: dict[str, object] | None,
) -> tuple[str, str, dict[str, object] | None]:
    """Augment code-error guidance with snapshot-aware state/entity facts."""
    missing = _referenced_missing(code, snapshot)
    if kind == "AttributeError" and missing and ("NoneType" in message or "value is None" in message):
        requested = missing[0]
        domain = requested.split(".", 1)[0] if "." in requested else ""
        guidance = advise(
            snapshot,
            FailureContext(intent=Intent.READ_STATE, requested=requested, domain=domain),
        ).to_payload()
        return (
            kind,
            f"State '{requested}' was not found; choose a visible entity id before reading its state.",
            guidance,
        )
    if (
        kind in {"AttributeError", "KeyError"}
        and (entity_id := _referenced_visible_state_id(code, snapshot)) is not None
    ):
        state = snapshot.states[entity_id]
        valid_names = ("state", *sorted(state.attributes))
        requested = _missing_attribute_name(message)
        guidance = advise(
            snapshot,
            FailureContext(
                intent=Intent.CODE_ATTRIBUTE,
                requested=requested,
                domain=state.domain,
                available_attributes=valid_names,
            ),
        ).to_payload()
        return (
            kind,
            f"Read '{entity_id}' using one of these valid fields or attributes: {', '.join(valid_names)}.",
            guidance,
        )
    return kind, message, fallback_guidance


def _missing_attribute_name(message: str) -> str:
    """Extract the missing attribute/key name from common Python error messages."""
    # AttributeError/KeyError wording differs by runtime; fall back to the whole message for ranking only.
    if match := re.search(r"(?:attribute|has no attribute) ['\"]([^'\"]+)['\"]", message):
        return match.group(1)
    if match := re.search(r"['\"]([^'\"]+)['\"]", message):
        return match.group(1)
    return message


def _failed_action_summary(actions: Sequence[object]) -> str | None:
    """Return an unmissable note for blocked/failed service calls, if any."""
    failed: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or action.get("status") != "error":
            continue
        error = action.get("error")
        failed.append(str(error.get("key", "unknown_error")) if isinstance(error, dict) else "unknown_error")
    if not failed:
        return None
    names = ", ".join(failed)
    return (
        f"{len(failed)} of {len(actions)} service calls were blocked or failed ({names}); "
        "the code output alone does not confirm these actions completed."
    )


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
    resolved_code, resolutions = substitute_remembered_literals(code, snapshot, runtime.memory)
    datetime_code, datetime_labels = normalize_datetime_imports(resolved_code)
    builtin_code, builtin_labels = normalize_builtins(datetime_code)
    normalized_code, await_labels = await_normalization.normalize_awaits(builtin_code, DATACLASS_REGISTRY)
    promoted_code, promote_labels = result_binding.promote_last_expression_to_result(normalized_code)
    executable_code, append_labels = result_binding.append_result_expression(promoted_code)
    runtime.state.normalizations = [*datetime_labels, *builtin_labels, *await_labels, *promote_labels, *append_labels]

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
        refined_kind, refined_message, guidance = refine_code_error(
            specific.__class__.__name__, str(specific) or str(err), resolved_code, snapshot
        )
        clean_message = _strip_monty_diagnostic(refined_message)
        refined_kind, clean_message, guidance = _read_path_fix(
            refined_kind, clean_message, resolved_code, snapshot, guidance
        )
        return code_error_payload_for_state(
            kind=refined_kind,
            message=clean_message,
            state=runtime.state,
            guidance=guidance,
        )
    finally:
        # Capture print output even on success; CollectString.output is a
        # concatenation, so we strip the trailing newline of each print() call.
        _capture_printed()
        if runtime.state.home_db is not None:
            # Close per-run SQLite state before clearing contextvars so no
            # snapshot-derived database survives past this execute_home_code call.
            runtime.state.home_db.close()
            runtime.state.home_db = None
        clear_runtime()

    result = json_safe(output)
    execution_payload: dict[str, object] = {"status": "ok"}
    payload: dict[str, object] = {"execution": execution_payload, "output": result}
    if resolutions:
        payload["resolutions"] = [{"requested": item.requested, "applied": item.applied} for item in resolutions]
    # Success-side notes are selected by an ordered static-analysis registry so
    # future legacy HA patterns can be added without growing executor branches.
    if note := compute_legacy_note(
        LegacyNoteContext(code=resolved_code, snapshot=snapshot, result=result, memory=runtime.memory)
    ):
        payload["note"] = note
    if runtime.state.printed:
        payload["printed"] = list(runtime.state.printed)
    if runtime.state.actions:
        actions = json_safe(runtime.state.actions)
        payload["actions"] = actions
        # Policy blocks are action outcomes, not code failures, so execution.status remains ok.
        if isinstance(actions, list) and (summary := _failed_action_summary(actions)) is not None:
            execution_payload["action_status"] = "error"
            execution_payload["action_failures"] = _failed_action_keys(actions)
            payload["notes"] = [summary, *runtime.state.notes]
    if runtime.state.notes:
        payload.setdefault("notes", list(runtime.state.notes))
    return payload


def _failed_action_keys(actions: Sequence[object]) -> list[str]:
    """Return stable error keys for failed service actions."""
    failed: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or action.get("status") != "error":
            continue
        error = action.get("error")
        failed.append(str(error.get("key", "unknown_error")) if isinstance(error, dict) else "unknown_error")
    return failed
