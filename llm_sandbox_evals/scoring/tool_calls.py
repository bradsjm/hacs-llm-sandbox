"""Deterministic one-to-one scoring for successful tool-call contracts."""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal

from llm_sandbox_evals.schema import (
    ExpectedToolCall,
    ToolCallComparison,
    ToolCallResult,
    ToolEvent,
)
from llm_sandbox_evals.tool_events import tool_succeeded


def score_tool_calls(expected: Sequence[ExpectedToolCall], events: Sequence[ToolEvent]) -> ToolCallResult:
    """Match every expected call to one successful event by exact name and canonical args."""
    successful = tuple(event for event in events if tool_succeeded(event))
    if not expected:
        return ToolCallResult(False, "tool_calls_no_events", unmatched_events=successful)

    available = list(range(len(successful)))
    comparisons: list[ToolCallComparison] = []
    for expected_call in expected:
        matched_index = next(
            (
                index
                for index in available
                if successful[index].tool_name == expected_call.tool_name
                and _canonical(successful[index].args) == _canonical(expected_call.args)
            ),
            None,
        )
        # State mutation point: consume a successful event after its sole expected match.
        if matched_index is not None:
            available.remove(matched_index)
            comparisons.append(ToolCallComparison(expected_call, successful[matched_index]))
        else:
            comparisons.append(ToolCallComparison(expected_call, None))

    unmatched_events = tuple(successful[index] for index in available)
    if all(comparison.matched_event is not None for comparison in comparisons):
        return ToolCallResult(True, "tool_calls_matched", tuple(comparisons), unmatched_events)

    reason: Literal["tool_calls_mismatched", "tool_calls_missing"] = (
        "tool_calls_mismatched"
        if any(
            comparison.matched_event is None
            and any(successful[index].tool_name == comparison.expected.tool_name for index in available)
            for comparison in comparisons
        )
        else "tool_calls_missing"
    )
    return ToolCallResult(False, reason, tuple(comparisons), unmatched_events)


def _canonical(value: object) -> object:
    """Return a stable recursive representation with JSON numbers normalized by value."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return ("number", Decimal(str(value)))
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _canonical(item)) for key, item in sorted(value.items())))
    if isinstance(value, list | tuple):
        return ("sequence", tuple(_canonical(item) for item in value))
    return value
