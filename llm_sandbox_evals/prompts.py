"""Prompt and function-schema assembly for the dev-only eval harness."""

import json
from pathlib import Path

from custom_components.llm_sandbox.const import (
    DEFAULT_PROMPT_PROFILE,
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.prompts import (
    ACTIONS_DISABLED_PROMPT,
    ACTIONS_ENABLED_PROMPT,
    build_execute_home_code_description,
    build_get_history_description,
    build_get_logbook_description,
    build_get_statistics_description,
    resolve_profile,
)
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot

from llm_sandbox_evals.schema import EvalCase, PromptCandidate, ToolSpec

_BASELINE_ID = "baseline"


def baseline_candidate(prompt_profile_id: str = DEFAULT_PROMPT_PROFILE) -> PromptCandidate:
    """Return the production-baseline prompt candidate."""
    profile = resolve_profile(prompt_profile_id)
    return PromptCandidate(
        id=_BASELINE_ID,
        api_prompt=profile.base_prompt,
        execute_home_code_description=build_execute_home_code_description(),
        get_history_description=build_get_history_description(),
        get_statistics_description=build_get_statistics_description(),
        get_logbook_description=build_get_logbook_description(),
    )


def load_candidates(candidate_ids: list[str], prompt_profile_id: str) -> list[PromptCandidate]:
    """Load supported prompt candidates, rejecting unknown candidate ids."""
    candidates: list[PromptCandidate] = []
    for candidate_id in candidate_ids:
        # Branch boundary: the production baseline is the built-in candidate.
        if candidate_id == _BASELINE_ID:
            candidates.append(baseline_candidate(prompt_profile_id))
            continue
        # Branch boundary: optimizer artifacts are explicitly loaded from JSON.
        if candidate_id.startswith("optimized:"):
            candidates.append(_load_optimized(candidate_id.removeprefix("optimized:")))
            continue
        # Candidate ids are configuration errors when unknown; do not silently fall back.
        raise ValueError(f"unknown prompt candidate id(s): {candidate_id}")
    return candidates


def render_messages(candidate: PromptCandidate, case: EvalCase, snapshot: HomeSnapshot) -> list[dict[str, object]]:
    """Render provider messages for the native tool-calling agent loop."""
    sections = [ACTIONS_ENABLED_PROMPT if case.actions_enabled else ACTIONS_DISABLED_PROMPT]
    location_section = _request_location_section(case.llm_context.device_id, snapshot)
    # The production API omits the location section when there is no initiating device.
    if location_section is not None:
        sections.append(location_section)
    system = f"{candidate.api_prompt}\n\n{'\n\n'.join(sections)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": case.user_request}]


def function_schemas(candidate: PromptCandidate) -> list[dict[str, object]]:
    """Return OpenAI-compatible function tool schemas from the candidate specs."""
    return [
        {
            "type": "function",
            "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters},
        }
        for spec in tool_specs(candidate)
    ]


def tool_specs(candidate: PromptCandidate) -> list[ToolSpec]:
    """Return tool specs in the production API's stable tool order."""
    return [
        ToolSpec(
            name=TOOL_EXECUTE_HOME_CODE,
            description=candidate.execute_home_code_description,
            args=("code",),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name=TOOL_GET_HISTORY,
            description=candidate.get_history_description,
            args=("entity_ids", "area_id", "device_id", "floor_id", "label_id", "domain", "hours", "start", "end"),
            parameters=_recorder_parameters(id_key="entity_ids"),
        ),
        ToolSpec(
            name=TOOL_GET_STATISTICS,
            description=candidate.get_statistics_description,
            args=(
                "statistic_ids",
                "area_id",
                "device_id",
                "floor_id",
                "label_id",
                "domain",
                "hours",
                "start",
                "end",
                "period",
            ),
            parameters=_recorder_parameters(id_key="statistic_ids", include_period=True),
        ),
        ToolSpec(
            name=TOOL_GET_LOGBOOK,
            description=candidate.get_logbook_description,
            args=("entity_ids", "area_id", "device_id", "floor_id", "label_id", "domain", "hours", "start", "end"),
            parameters=_recorder_parameters(id_key="entity_ids"),
        ),
    ]


def _recorder_parameters(*, id_key: str, include_period: bool = False) -> dict[str, object]:
    """Build the shared recorder JSON Schema accepted by native function calling."""
    properties: dict[str, object] = {
        id_key: {"type": "array", "items": {"type": "string"}},
        "area_id": {"type": "string"},
        "device_id": {"type": "string"},
        "floor_id": {"type": "string"},
        "label_id": {"type": "string"},
        "domain": {"type": "string"},
        "hours": {"type": "number"},
        "start": {"type": "string"},
        "end": {"type": "string"},
    }
    # Branch boundary: statistics adds one aggregation-period enum.
    if include_period:
        properties["period"] = {"type": "string", "enum": ["5minute", "hour", "day"]}
    return {"type": "object", "properties": properties, "additionalProperties": False}


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
