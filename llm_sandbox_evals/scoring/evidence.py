"""Successful-event selection and provenance-preserving evidence union."""

from collections.abc import Mapping, Sequence

from llm_sandbox_evals.schema import ToolEvent
from llm_sandbox_evals.scoring.automation import normalize_automation_output
from llm_sandbox_evals.scoring.contracts import EvidenceFact, NormalizedEvidence, Provenance
from llm_sandbox_evals.scoring.execute import normalize_execute_output
from llm_sandbox_evals.scoring.history import normalize_history_output
from llm_sandbox_evals.scoring.logbook import normalize_logbook_output
from llm_sandbox_evals.scoring.statistics import normalize_statistics_output

_TOOLS = frozenset({"execute_home_code", "get_history", "get_statistics", "get_logbook", "get_automation"})


def successful_events(events: Sequence[ToolEvent]) -> tuple[ToolEvent, ...]:
    """Return usable production events; failed calls remain diagnostics."""
    result: list[ToolEvent] = []
    for event in events:
        if event.tool_name not in _TOOLS or not isinstance(event.output, Mapping):
            continue
        execution = event.output.get("execution")
        if event.tool_name == "execute_home_code" and (
            not isinstance(execution, Mapping) or execution.get("status") != "ok"
        ):
            continue
        if event.output.get("status") == "error":
            continue
        if not _recognizable_success(event.tool_name, event.output):
            continue
        result.append(event)
    return tuple(result)


def _recognizable_success(tool_name: str, output: Mapping[str, object]) -> bool:
    """Require a production-shaped envelope while retaining valid empty results."""
    if tool_name == "execute_home_code":
        return "output" in output
    expected_keys = {
        "get_history": {"entities", "rows", "summary"},
        "get_statistics": {"statistics", "summary"},
        "get_logbook": {"entries"},
        "get_automation": {"automations"},
    }[tool_name]
    return not expected_keys.isdisjoint(output)


def normalize_events(events: Sequence[ToolEvent]) -> NormalizedEvidence:
    """Union successful facts independently of call order or execution path."""
    facts: list[EvidenceFact] = []
    for event in successful_events(events):
        provenance = Provenance(
            event.tool_name, event.call_index, event.turn_index, event.batch_index, event.tool_name
        )
        if event.tool_name == "execute_home_code":
            facts.extend(normalize_execute_output(event.output.get("output"), provenance))
        elif event.tool_name == "get_history":
            facts.extend(normalize_history_output(event.output, provenance))
        elif event.tool_name == "get_statistics":
            facts.extend(normalize_statistics_output(event.output, provenance))
        elif event.tool_name == "get_logbook":
            facts.extend(normalize_logbook_output(event.output, provenance))
        else:
            facts.extend(normalize_automation_output(event.output, provenance))
    return NormalizedEvidence(tuple(facts))
