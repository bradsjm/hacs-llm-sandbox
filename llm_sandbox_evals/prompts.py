"""Prompt assembly for the dev-only eval harness.

The baseline candidate reuses production prompt text and tool descriptions so
model evaluations see the same static instructions the integration sends. This
module adds only eval-harness framing: tool-spec presentation, JSON response
contract, and the concrete user request.
"""

import json
from pathlib import Path

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.prompts import (
    ACTIONS_DISABLED_PROMPT,
    ACTIONS_ENABLED_PROMPT,
    BASE_API_PROMPT,
    build_execute_home_code_description,
    build_get_history_description,
    build_get_logbook_description,
    build_get_statistics_description,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import EvalCase, PromptCandidate, ToolSpec

_BASELINE_ID = "baseline"


def baseline_candidate() -> PromptCandidate:
    """Return the production-baseline prompt candidate."""
    return PromptCandidate(
        id=_BASELINE_ID,
        api_prompt=BASE_API_PROMPT,
        execute_home_code_description=build_execute_home_code_description(),
        get_history_description=build_get_history_description(),
        get_statistics_description=build_get_statistics_description(),
        get_logbook_description=build_get_logbook_description(),
    )


def load_candidates(candidate_ids: list[str]) -> list[PromptCandidate]:
    """Load supported prompt candidates, rejecting unknown candidate ids."""
    candidates: list[PromptCandidate] = []
    for candidate_id in candidate_ids:
        # Branch boundary: the production baseline is the built-in candidate.
        if candidate_id == _BASELINE_ID:
            candidates.append(baseline_candidate())
            continue
        # Branch boundary: optimizer artifacts are explicitly loaded from JSON.
        if candidate_id.startswith("optimized:"):
            candidates.append(_load_optimized(candidate_id.removeprefix("optimized:")))
            continue
        # Candidate ids are configuration errors when unknown; do not silently fall back.
        raise ValueError(f"unknown prompt candidate id(s): {candidate_id}")
    return candidates


def render_prompt(candidate: PromptCandidate, case: EvalCase, snapshot: HomeSnapshot) -> str:
    """Render the complete single-message prompt sent to the model adapter."""
    return f"{candidate.api_prompt}\n\n{render_context(candidate, case, snapshot)}"


def render_context(candidate: PromptCandidate, case: EvalCase, snapshot: HomeSnapshot) -> str:
    """Render the production-like prompt context without the leading API prompt."""
    sections = [ACTIONS_ENABLED_PROMPT if case.actions_enabled else ACTIONS_DISABLED_PROMPT]
    location_section = _request_location_section(case.llm_context.device_id, snapshot)
    # The production API omits the location section when there is no initiating device.
    if location_section is not None:
        sections.append(location_section)
    sections.extend((_tools_section(candidate), _response_contract_section(), _user_request_section(case)))
    return "\n\n".join(sections)


def tool_specs(candidate: PromptCandidate) -> list[ToolSpec]:
    """Return tool specs in the production API's stable tool order."""
    return [
        ToolSpec(
            name=TOOL_EXECUTE_HOME_CODE,
            description=candidate.execute_home_code_description,
            args=("code",),
        ),
        ToolSpec(
            name=TOOL_GET_HISTORY,
            description=candidate.get_history_description,
            args=("entity_ids", "start", "end"),
        ),
        ToolSpec(
            name=TOOL_GET_STATISTICS,
            description=candidate.get_statistics_description,
            args=("statistic_ids", "start", "end", "period"),
        ),
        ToolSpec(
            name=TOOL_GET_LOGBOOK,
            description=candidate.get_logbook_description,
            args=("entity_ids", "start", "end"),
        ),
    ]


def _request_location_section(device_id: str | None, snapshot: HomeSnapshot) -> str | None:
    """Render the production request-location section from frozen snapshot records."""
    if device_id is None:
        return None

    device = snapshot.devices.get(device_id)
    area_id = device.area_id if device is not None else None
    area = snapshot.areas.get(area_id) if area_id is not None else None
    floor_id = area.floor_id if area is not None else None
    floor = snapshot.floors.get(floor_id) if floor_id is not None else None

    lines = [
        "## Request location",
        f"- device_id: {device_id}",
    ]
    # Area/floor lines exactly mirror production labels while guarding missing snapshot records.
    if area is not None:
        lines.append(f"- area_id: {area.id} ({area.name})")
    if floor is not None:
        lines.append(f"- floor_id: {floor.floor_id} ({floor.name})")
    if area is not None:
        lines.append(
            "For underspecified local questions, use this area as the default scope. "
            "If the user asks for the whole home or names another area/floor, follow that explicit scope."
        )
    return "\n".join(lines)


def _tools_section(candidate: PromptCandidate) -> str:
    """Render the eval adapter's plain-text tool list."""
    lines = ["## Available tools"]
    for spec in tool_specs(candidate):
        lines.extend(
            (
                f"### {spec.name}",
                spec.description,
                f"Arguments: {', '.join(spec.args)}",
            )
        )
    return "\n".join(lines)


def _response_contract_section() -> str:
    """Render the JSON-only model response contract."""
    tool_names = ", ".join(spec.name for spec in tool_specs(baseline_candidate()))
    return (
        "## Response contract\n"
        "Reply with ONLY a single JSON object of the form "
        '{"tool_name": "<one of the tool names>", "tool_args": {...}} '
        "and nothing else: no prose, no code fences. "
        f"tool_name must be one of: {tool_names}. "
        'For execute_home_code, tool_args must be {"code": "<python>"}. '
        "For recorder tools, entity_ids/statistic_ids must reference currently-visible entities."
    )


def _user_request_section(case: EvalCase) -> str:
    """Render the concrete user request being evaluated."""
    return f"## User request\n{case.user_request}"


def _load_optimized(path: str) -> PromptCandidate:
    """Load an optimized prompt candidate from a JSON artifact."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("optimized candidate JSON must contain an object")
    return PromptCandidate(
        id=_string_field(decoded, "id"),
        api_prompt=_string_field(decoded, "api_prompt"),
        execute_home_code_description=_string_field(decoded, "execute_home_code_description"),
        get_history_description=_string_field(decoded, "get_history_description"),
        get_statistics_description=_string_field(decoded, "get_statistics_description"),
        get_logbook_description=_string_field(decoded, "get_logbook_description"),
    )


def _string_field(data: dict[object, object], key: str) -> str:
    """Return a required string field from an optimizer candidate artifact."""
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"optimized candidate field {key!r} must be a string")
    return value
