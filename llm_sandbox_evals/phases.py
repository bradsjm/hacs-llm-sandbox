"""Safe non-UI observations emitted while an eval cell executes."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

type LanePhase = Literal[
    "queued",
    "awaiting_model",
    "thinking",
    "preparing_tool_call",
    "running_tool",
    "processing_tool_result",
    "responding",
    "scoring",
    "finished",
]


@dataclass(frozen=True, slots=True)
class PhaseObservation:
    """Payload-free execution phase that can be safely forwarded to observers."""

    phase: LanePhase
    tool_name: str | None = None


type PhaseObserver = Callable[[PhaseObservation], None]


class PhaseEmitter(Protocol):
    """Callback used internally to emit a phase with an optional tool name."""

    def __call__(self, phase: LanePhase, tool_name: str | None = None) -> None:
        """Emit one safe execution phase."""
