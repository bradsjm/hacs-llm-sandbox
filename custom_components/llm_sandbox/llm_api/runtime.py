"""Runtime context helpers for LLM Sandbox facade methods."""

from contextvars import ContextVar, Token
from dataclasses import dataclass

from ..runtime import SandboxSettings
from .executor_support import ExecutionState


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Private runtime dependencies for facade methods.

    Reads against the frozen snapshot are pure and need no runtime access.
    The propose-only services facade reaches into ``state`` to record proposed
    actions and enforce helper-call budgets. There is no semantic index,
    scheduler, background manager, or live Home Assistant reference here.
    """

    state: ExecutionState
    settings: SandboxSettings


_ACTIVE_RUNTIME: ContextVar[RuntimeContext | None] = ContextVar("llm_sandbox_active_runtime", default=None)
_RUNTIME_TOKEN: ContextVar[Token[RuntimeContext | None] | None] = ContextVar("llm_sandbox_runtime_token", default=None)


def activate_runtime(runtime: RuntimeContext) -> None:
    """Register the active runtime for Monty-visible facade methods."""
    _RUNTIME_TOKEN.set(_ACTIVE_RUNTIME.set(runtime))


def clear_runtime() -> None:
    """Clear the active runtime after one execute_home_code run."""
    token = _RUNTIME_TOKEN.get()
    # Executor cleanup may run after setup failures that never activated a runtime.
    if token is None:
        return
    _RUNTIME_TOKEN.set(None)
    _ACTIVE_RUNTIME.reset(token)


def require_runtime(runtime: RuntimeContext | None) -> RuntimeContext:
    """Return the active runtime or fail when methods run out of band."""
    if (active := _ACTIVE_RUNTIME.get()) is not None:
        return active
    if runtime is not None:
        return runtime
    raise RuntimeError("LLM Sandbox runtime methods are unavailable outside execute_home_code")
