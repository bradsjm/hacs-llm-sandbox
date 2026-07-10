"""JSON-safe output and executor payload builders."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

from homeassistant.util.json import JsonValueType

from ...types import ActionRecord
from ..errors import (
    CodeErrorPayload,
    HelperErrorPayload,
    HelperExecutionError,
    code_error_payload,
    helper_error_payload,
    tool_error_message,
)
from .state import ExecutionState


def overflow_metadata(
    *,
    truncated: bool,
    returned: int,
    limit: int | None = None,
    omitted: int | None = None,
    next_cursor: str | None = None,
) -> dict[str, object]:
    """Return the shared structured overflow/truncation metadata shape."""
    metadata: dict[str, object] = {"truncated": truncated, "returned": returned}
    if limit is not None:
        metadata["limit"] = limit
    if omitted is not None:
        metadata["omitted"] = omitted
    if next_cursor is not None:
        metadata["next_cursor"] = next_cursor
    return metadata


def json_safe(value: object) -> JsonValueType:
    """Convert arbitrary values into JSON-safe structures."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    sandbox_json = getattr(value, "__llm_sandbox_json__", None)
    if callable(sandbox_json):
        return json_safe(sandbox_json())
    if isinstance(value, Mapping):
        mapping_items = cast(Mapping[object, object], value)
        return {str(key): json_safe(item) for key, item in mapping_items.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        sequence_items = cast(Sequence[object], value)
        return [json_safe(item) for item in sequence_items]
    if isinstance(value, set):
        set_items = cast(set[object], value)
        return [json_safe(item) for item in set_items]
    return str(value)


def helper_error_payload_for_state(
    err: HelperExecutionError,
    state: ExecutionState,
) -> HelperErrorPayload:
    """Build a helper-error response using current execution state."""
    payload = helper_error_payload(
        err,
        message=_helper_error_message(err, state),
        kind=err.key,
        guidance=err.guidance,
        adjustments=list(state.adjustments),
        printed=list(state.printed),
        actions=cast(list[ActionRecord], json_safe(state.actions)),
    )
    if state.notes:
        payload["notes"] = list(state.notes)
    if state.overflow:
        payload["overflow"] = dict(state.overflow)
    return payload


def code_error_payload_for_state(
    *,
    kind: str,
    message: str,
    state: ExecutionState,
    guidance: dict[str, object] | None = None,
) -> CodeErrorPayload:
    """Build a code-execution error response using current state."""
    payload = code_error_payload(
        kind=kind,
        message=message,
        adjustments=list(state.adjustments),
        printed=list(state.printed),
        actions=cast(list[ActionRecord], json_safe(state.actions)),
        guidance=guidance,
    )
    if state.notes:
        payload["notes"] = list(state.notes)
    if state.overflow:
        payload["overflow"] = dict(state.overflow)
    return payload


def _helper_error_message(err: HelperExecutionError, state: ExecutionState) -> str:
    """Return one actionable sentence for a helper execution error."""
    if err.key == "service_call_limit_exceeded":
        return f"Stopped after {state.service_call_limit} dispatched service calls; do not retry the same call."
    if reason := err.placeholders.get("reason"):
        return f"Fix the {err.helper} call failure: {reason}."
    if message := tool_error_message(err.key, err.placeholders):
        return message
    return f"Resolve the {err.helper} error '{err.key}' before retrying."
