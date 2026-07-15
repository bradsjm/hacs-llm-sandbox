"""Shared production-shaped tool-event classification."""

from llm_sandbox_evals.schema import ToolEvent

_TOOL_EXECUTE_HOME_CODE = "execute_home_code"


def tool_succeeded(event: ToolEvent) -> bool:
    """Return whether a captured event contains a successful production result envelope."""
    if event.output.get("status") == "error":
        return False
    if event.tool_name == _TOOL_EXECUTE_HOME_CODE:
        execution = event.output.get("execution")
        return isinstance(execution, dict) and execution.get("status") == "ok" and "output" in event.output
    expected_keys = {
        "get_history": {"entities", "rows", "summary"},
        "get_statistics": {"statistics", "summary"},
        "get_logbook": {"entries"},
        "get_automation": {"automations"},
    }.get(event.tool_name)
    return expected_keys is not None and not expected_keys.isdisjoint(event.output)
