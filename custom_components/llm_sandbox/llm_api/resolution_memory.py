"""Conversation-scoped advisory memory for entity-id resolution."""

from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic

from homeassistant.helpers import llm

_LITERAL_LIMIT = 32
_CONVERSATION_LIMIT = 16
_CONVERSATION_TTL_SECONDS = 7200.0
_KEY_SENTINEL = ""

type ConversationKey = tuple[str, str, str]


@dataclass(slots=True)
class ResolutionMemory:
    """Small LRU of requested literals that resolved to visible entity ids."""

    _resolved_by_request: OrderedDict[str, str] = field(default_factory=OrderedDict)

    def lookup(self, requested: str) -> str | None:
        """Return the remembered entity id for ``requested``, touching the LRU."""
        key = _normalize_request(requested)
        if key not in self._resolved_by_request:
            return None
        resolved = self._resolved_by_request.pop(key)
        # Touch successful reads so recently useful resolutions survive LRU trim.
        self._resolved_by_request[key] = resolved
        return resolved

    def record(self, requested: str, resolved: str) -> None:
        """Remember that ``requested`` should resolve to ``resolved``."""
        key = _normalize_request(requested)
        if not key:
            return
        if key in self._resolved_by_request:
            self._resolved_by_request.pop(key)
        # Mutate the per-conversation LRU only after the caller validated the
        # resolved id against its fresh snapshot.
        self._resolved_by_request[key] = resolved
        while len(self._resolved_by_request) > _LITERAL_LIMIT:
            self._resolved_by_request.popitem(last=False)

    def remembered_entity_ids(self) -> tuple[str, ...]:
        """Return recently remembered entity ids, newest first, without touching LRU."""
        seen: set[str] = set()
        entity_ids: list[str] = []
        for entity_id in reversed(self._resolved_by_request.values()):
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entity_ids.append(entity_id)
        return tuple(entity_ids)


@dataclass(slots=True)
class ResolutionMemoryStore:
    """Per-entry LRU of surrogate conversation resolution memories."""

    _conversations: OrderedDict[ConversationKey, tuple[float, ResolutionMemory]] = field(default_factory=OrderedDict)

    def for_context(self, llm_context: llm.LLMContext) -> ResolutionMemory:
        """Return the memory bucket for a Home Assistant LLM context surrogate."""
        now = monotonic()
        self._evict_expired(now)
        key = _surrogate_key(llm_context)
        if key in self._conversations:
            _, memory = self._conversations.pop(key)
        else:
            memory = ResolutionMemory()
        # Touch or create the per-conversation bucket for LRU/TTL accounting.
        self._conversations[key] = (now, memory)
        while len(self._conversations) > _CONVERSATION_LIMIT:
            self._conversations.popitem(last=False)
        return memory

    def _evict_expired(self, now: float) -> None:
        """Drop stale surrogate conversations before lookup or creation."""
        for key, (last_seen, _) in tuple(self._conversations.items()):
            if now - last_seen <= _CONVERSATION_TTL_SECONDS:
                continue
            self._conversations.pop(key, None)


def _surrogate_key(llm_context: llm.LLMContext) -> ConversationKey:
    """Build the selected server-side conversation surrogate."""
    context = llm_context.context
    return (
        llm_context.assistant or _KEY_SENTINEL,
        llm_context.device_id or _KEY_SENTINEL,
        context.user_id if context is not None and context.user_id is not None else _KEY_SENTINEL,
    )


def _normalize_request(requested: str) -> str:
    """Normalize requested literals for memory lookup."""
    return requested.strip().lower()
