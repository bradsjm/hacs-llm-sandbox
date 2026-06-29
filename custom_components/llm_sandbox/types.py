"""Shared typing primitives for LLM Sandbox JSON boundaries."""

from homeassistant.util.json import JsonValueType

type TranslationPlaceholders = dict[str, str]
type ToolArgs = dict[str, JsonValueType]
type ProposedAction = dict[str, object]
