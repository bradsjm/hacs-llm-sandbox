"""Runtime context helpers for LLM Sandbox facade methods."""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.core import Context, HomeAssistant

from ..runtime import SandboxSettings
from .executor_support import ExecutionState

if TYPE_CHECKING:
    from ..snapshot.models import HomeSnapshot


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Private runtime dependencies for facade methods.

    Reads against the frozen snapshot are pure and need no runtime access, so
    only the propose-only services facade reaches into ``state`` (to record a
    proposed action) and ``context`` (future execution boundary). There is no
    semantic index, scheduler, or background manager here.
    """

    hass: HomeAssistant
    snapshot: HomeSnapshot
    state: ExecutionState
    context: Context
    settings: SandboxSettings


_ACTIVE_RUNTIME: ContextVar[RuntimeContext | None] = ContextVar("llm_sandbox_active_runtime", default=None)
_RUNTIME_TOKENS: ContextVar[tuple[Token[RuntimeContext | None], ...]] = ContextVar(
    "llm_sandbox_runtime_tokens",
    default=(),
)


def activate_runtime(runtime: RuntimeContext) -> None:
    """Register the active runtime for Monty-visible facade methods."""
    # Store the token so nested executions can restore the previous scope.
    token = _ACTIVE_RUNTIME.set(runtime)
    _RUNTIME_TOKENS.set((*_RUNTIME_TOKENS.get(), token))


def clear_runtime() -> None:
    """Clear the active runtime after one execute_home_code run."""
    tokens = _RUNTIME_TOKENS.get()
    # Executor cleanup may run after setup failures that never activated a runtime.
    if not tokens:
        return
    # Pop and reset the matching token instead of clobbering an outer scope.
    _RUNTIME_TOKENS.set(tokens[:-1])
    _ACTIVE_RUNTIME.reset(tokens[-1])


def require_runtime(runtime: RuntimeContext | None) -> RuntimeContext:
    """Return the active runtime or fail when methods run out of band."""
    if (active := _ACTIVE_RUNTIME.get()) is not None:
        return active
    if runtime is not None:
        return runtime
    raise RuntimeError("LLM Sandbox runtime methods are unavailable outside execute_home_code")
