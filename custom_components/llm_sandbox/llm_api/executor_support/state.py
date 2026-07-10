"""Execution state and helper-call support for the Monty executor."""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

import pydantic_monty  # Required manifest dependency; do not convert to a dynamic import.
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util.json import JsonValueType

from ...const import DEFAULT_SERVICE_CALL_LIMIT
from ...types import ActionRecord
from ..data.home_db import HomeDatabase
from ..errors import HelperExecutionError, RecoverableToolError
from .refinement import error_key, error_placeholders


class MontyRunner(Protocol):
    """Runtime shape required from the optional Monty dependency."""

    async def run_async(
        self,
        *,
        inputs: Mapping[str, object],
        limits: pydantic_monty.ResourceLimits | None = None,
        external_functions: Mapping[str, object],
        print_callback: object = None,
    ) -> object:
        """Execute Monty code with JSON inputs and helper functions."""
        ...


class MontyFactory(Protocol):
    """Callable factory exposed as pydantic_monty.Monty."""

    def __call__(
        self,
        code: str,
        *,
        inputs: list[str],
        script_name: str,
        type_check: bool,
        type_check_stubs: str,
        dataclass_registry: list[type[object]] | None,
    ) -> MontyRunner:
        """Construct a runnable Monty program instance."""
        ...


@dataclass(slots=True)
class ExecutionState:
    """Mutable per-run bookkeeping for service dispatch limits and outputs."""

    dispatched_service_calls: int = 0
    service_call_limit: int = DEFAULT_SERVICE_CALL_LIMIT
    # Internal forgiveness-layer labels. Payloads expose these as concise
    # adjustments that tell the model the change was already applied.
    normalizations: list[str] = field(default_factory=list)
    adjustments: list[dict[str, object]] = field(default_factory=list)
    # Captured print() output, one entry per call. Independent of the service
    # dispatch limit: print() routes through Monty's print_callback.
    printed: list[str] = field(default_factory=list)
    # Structured truncation metadata for large executor-owned surfaces. Individual
    # helpers can add surface-specific records while preserving existing payloads.
    overflow: dict[str, object] = field(default_factory=dict)
    # Service action outcomes recorded by the services facade. Actions execute
    # sequentially, and prior successful entries remain when a later call fails.
    actions: list[ActionRecord] = field(default_factory=list)
    # Last helper validation error raised inside a facade method. Monty wraps
    # such exceptions into MontyRuntimeError without preserving __cause__, so
    # the executor recovers the structured error from here only when Monty's
    # generic wrapper carries the error's opaque per-run marker. Cleared at
    # the start of each helper call so a user try/except that swallows a
    # helper error cannot shadow a later helper-path failure.
    last_helper_error: HelperExecutionError | None = None
    # Per-run SQL database and transparency notes. The database is created lazily
    # by hass.query() and closed by executor cleanup before runtime context reset.
    home_db: HomeDatabase | None = None
    notes: list[str] = field(default_factory=list)
    # Set when this run dispatched at least one live service call, so later
    # recorder-backed reads synchronize before reading (read-after-write).
    live_write_dispatched: bool = False


async def helper_response(
    state: ExecutionState,
    helper: str,
    callback: Callable[[], object],
) -> JsonValueType:
    """Run one approved helper with async, error, and JSON handling."""
    # Clear any prior helper error so a user try/except that swallowed a
    # previous helper error cannot shadow a later genuine code error.
    state.last_helper_error = None
    try:
        value = callback()
        if inspect.isawaitable(value):
            value = await cast(Awaitable[JsonValueType], value)
    except HelperExecutionError as err:
        state.last_helper_error = err
        raise
    except RecoverableToolError as err:
        helper_err = HelperExecutionError(helper, err.key, err.placeholders)
        state.last_helper_error = helper_err
        raise helper_err from err
    except ServiceValidationError as err:
        helper_err = HelperExecutionError(
            helper,
            error_key(err),
            error_placeholders(err),
        )
        state.last_helper_error = helper_err
        raise helper_err from err
    from .output import json_safe

    return json_safe(value)
