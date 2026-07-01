"""Model adapters for eval prompt completion."""

import asyncio
import importlib
import json
import re
from collections.abc import Callable
from typing import Protocol, cast

from llm_sandbox_evals.schema import ModelResult

_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z_]+\b")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FIXED_END = "2026-06-29T12:00:00+00:00"
_FIXED_START = "2026-06-28T12:00:00+00:00"


class ModelAdapter(Protocol):
    """Protocol implemented by all model completion adapters."""

    async def complete(self, model_id: str, prompt: str) -> ModelResult:
        """Complete a rendered eval prompt and return the normalized tool call."""


class StubAdapter:
    """Deterministic offline adapter used to validate the eval pipeline."""

    async def complete(self, model_id: str, prompt: str) -> ModelResult:
        """Return a deterministic tool call derived from the user request."""
        _ = model_id
        # Keyword detection must inspect only the user request: the full prompt
        # always lists every tool name/description, which would otherwise match
        # recorder keywords (e.g. "logbook") on every case.
        user_request = _extract_user_request(prompt)
        tool_name = _select_stub_tool(user_request)
        tool_args = _build_stub_tool_args(tool_name, user_request)
        tool_call: dict[str, object] = {"tool_name": tool_name, "tool_args": tool_args}
        return ModelResult(raw_text=json.dumps(tool_call, sort_keys=True), tool_call=tool_call, error=None)


class LiteLLMAdapter:
    """Adapter for real models routed through LiteLLM."""

    def __init__(self, reasoning_effort: str | None = None) -> None:
        """Store the optional reasoning effort level forwarded to LiteLLM."""
        self._reasoning_effort = reasoning_effort

    async def complete(self, model_id: str, prompt: str) -> ModelResult:
        """Call LiteLLM and parse the returned JSON tool call.

        The blocking LiteLLM call runs in a worker thread so the event loop
        stays free for concurrent matrix cells. Provider timeout plus a hard
        ``wait_for`` bound hung or rate-limited requests so a single call cannot
        stall the whole run; any failure becomes a ``ModelResult.error``.
        """
        try:
            litellm = importlib.import_module("litellm")
            completion = cast(Callable[..., object], litellm.__dict__["completion"])
            # Reasoning models generally ignore or reject an explicit temperature,
            # so let the provider default it when a reasoning effort is requested.
            kwargs: dict[str, object] = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 60,
                "num_retries": 1,
            }
            if self._reasoning_effort:
                kwargs["reasoning_effort"] = self._reasoning_effort
            else:
                kwargs["temperature"] = 0.0
            response = await asyncio.wait_for(
                asyncio.to_thread(completion, **kwargs),
                timeout=90,
            )
            content = _extract_litellm_content(response)
        except Exception as err:  # noqa: BLE001 - adapter boundary converts provider/runtime failures into data.
            return ModelResult(raw_text="", tool_call=None, error=f"{type(err).__name__}: {err}")

        return ModelResult(raw_text=content, tool_call=parse_tool_call(content), error=None)


def get_adapter(model_id: str, reasoning_effort: str | None = None) -> ModelAdapter:
    """Return the adapter for a model identifier."""
    # The stub adapter is a first-class offline model for keyless verification.
    if model_id == "stub":
        return StubAdapter()
    return LiteLLMAdapter(reasoning_effort=reasoning_effort)


def parse_tool_call(raw_text: str) -> dict[str, object] | None:
    """Extract and validate a JSON tool call from model text."""
    # Prefer fenced JSON because many models wrap their final answer in Markdown.
    for match in _FENCED_JSON_RE.finditer(raw_text):
        parsed = _loads_tool_call(match.group(1))
        if parsed is not None:
            return parsed

    json_object = _first_json_object(raw_text)
    if json_object is None:
        return None
    return _loads_tool_call(json_object)


def _extract_user_request(prompt: str) -> str:
    """Isolate the user request from the rendered eval prompt.

    The render places the request in a trailing ``## User request`` section; the
    stub analyzes only that text so tool names in the prompt body do not pollute
    keyword detection.
    """
    marker = "## User request\n"
    index = prompt.rfind(marker)
    if index == -1:
        return prompt
    return prompt[index + len(marker) :].strip()


def _select_stub_tool(user_request: str) -> str:
    """Choose a tool name with deterministic keyword matching."""
    lowered = user_request.lower()
    # Recorder keywords are ordered by the requested priority.
    if "logbook" in lowered or "happened" in lowered:
        return "get_logbook"
    if "statistic" in lowered or "hourly" in lowered or "average over" in lowered:
        return "get_statistics"
    if "history" in lowered or "last 24" in lowered or "over the last" in lowered:
        return "get_history"
    return "execute_home_code"


def _build_stub_tool_args(tool_name: str, user_request: str) -> dict[str, object]:
    """Build deterministic, runnable tool arguments for the selected stub tool."""
    # Execute code is intentionally read-only and only inspects the frozen facade.
    # ``SafeStateMachine`` exposes ``entity_ids()`` (HA parity), not ``.keys()``.
    if tool_name == "execute_home_code":
        return {"code": "states.entity_ids()"}

    entity_ids = _ENTITY_ID_RE.findall(user_request)
    ids: list[str] = entity_ids[:1]
    if tool_name == "get_statistics":
        return {"statistic_ids": ids, "start": _FIXED_START, "end": _FIXED_END, "period": "hour"}
    return {"entity_ids": ids, "start": _FIXED_START, "end": _FIXED_END}


def _loads_tool_call(raw_json: str) -> dict[str, object] | None:
    """Parse and validate the provider-neutral tool-call contract."""
    try:
        decoded = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    # Only a JSON object with the expected fields is a valid tool call.
    if not isinstance(decoded, dict):
        return None

    tool_name = decoded.get("tool_name")
    tool_args = decoded.get("tool_args")
    if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
        return None

    return {"tool_name": tool_name, "tool_args": dict(tool_args)}


def _first_json_object(raw_text: str) -> str | None:
    """Return the first balanced JSON-object-looking substring."""
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw_text[start:], start=start):
        if in_string:
            # JSON strings can contain escaped quotes and brace characters.
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : index + 1]

    return None


def _extract_litellm_content(response: object) -> str:
    """Extract ``choices[0].message.content`` from a LiteLLM response object."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("LiteLLM response did not include choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("LiteLLM response did not include message content")
    return content
