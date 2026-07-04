"""Runtime context helpers for LLM Sandbox facade methods."""

import math
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Protocol

from ..runtime import SandboxSettings
from ..snapshot.models import HomeSnapshot
from ..types import ProposedAction
from .executor_support import ExecutionState
from .resolution_memory import ResolutionMemory


class ServiceInvoker(Protocol):
    """Private live service-call boundary kept outside Monty inputs."""

    async def __call__(self, action: ProposedAction) -> Any:  # noqa: ANN401
        """Execute one validated service action through live Home Assistant."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Private runtime dependencies for facade methods.

    Reads against the frozen snapshot are pure and need no runtime access.
    The services facade reaches into ``state`` to record action outcomes and
    enforce helper-call budgets. Live Home Assistant access remains behind the
    private ``invoke`` callable; no live HA object is exposed to Monty.
    """

    state: ExecutionState
    settings: SandboxSettings
    invoke: ServiceInvoker
    deadline: float = math.inf
    memory: ResolutionMemory | None = None


_ACTIVE_RUNTIME: ContextVar[RuntimeContext | None] = ContextVar("llm_sandbox_active_runtime", default=None)
_RUNTIME_TOKEN: ContextVar[Token[RuntimeContext | None] | None] = ContextVar("llm_sandbox_runtime_token", default=None)
_ACTIVE_SNAPSHOT: ContextVar[HomeSnapshot | None] = ContextVar("llm_sandbox_active_snapshot", default=None)
_SNAPSHOT_TOKEN: ContextVar[Token[HomeSnapshot | None] | None] = ContextVar("llm_sandbox_snapshot_token", default=None)


def activate_runtime(runtime: RuntimeContext, snapshot: HomeSnapshot) -> None:
    """Register the active runtime for Monty-visible facade methods."""
    _RUNTIME_TOKEN.set(_ACTIVE_RUNTIME.set(runtime))
    _SNAPSHOT_TOKEN.set(_ACTIVE_SNAPSHOT.set(snapshot))


def clear_runtime() -> None:
    """Clear the active runtime after one execute_home_code run."""
    token = _RUNTIME_TOKEN.get()
    # Executor cleanup may run after setup failures that never activated a runtime.
    if token is None:
        return
    _RUNTIME_TOKEN.set(None)
    _ACTIVE_RUNTIME.reset(token)
    snapshot_token = _SNAPSHOT_TOKEN.get()
    if snapshot_token is not None:
        _SNAPSHOT_TOKEN.set(None)
        _ACTIVE_SNAPSHOT.reset(snapshot_token)


def require_runtime(runtime: RuntimeContext | None) -> RuntimeContext:
    """Return the active runtime or fail when methods run out of band."""
    if (active := _ACTIVE_RUNTIME.get()) is not None:
        return active
    if runtime is not None:
        return runtime
    raise RuntimeError("LLM Sandbox runtime methods are unavailable outside execute_home_code")


def require_snapshot() -> HomeSnapshot:
    """Return the active frozen snapshot for copied Monty facade methods."""
    if (snapshot := _ACTIVE_SNAPSHOT.get()) is not None:
        return snapshot
    raise RuntimeError("LLM Sandbox snapshot is unavailable outside execute_home_code")
