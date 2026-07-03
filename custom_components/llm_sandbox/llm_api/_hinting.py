"""Shared formatting helpers for LLM-facing remediation hints."""

from collections.abc import Mapping, Sequence


class SafeHintDict(dict[str, str]):
    """dict that keeps unknown ``{placeholder}`` tokens verbatim instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def error_guidance(
    guidance: Mapping[str, tuple[str, Sequence[str]]],
    key: str,
    placeholders: Mapping[str, str],
) -> tuple[str | None, list[str] | None]:
    """Return (message, fix) for an error key, formatting placeholders safely."""
    entry = guidance.get(key)
    if entry is None:
        return None, None
    message, templates = entry
    values = SafeHintDict({str(k): str(v) for k, v in placeholders.items()})
    return message, [template.format_map(values) for template in templates]
