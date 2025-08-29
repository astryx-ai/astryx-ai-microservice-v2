from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class MemoryStore:
    """Abstract interface for chat memory persistence keyed by chat_id/user_id."""

    def load(self, chat_id: Optional[str], user_id: Optional[str], ttl_seconds: int | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, chat_id: Optional[str], user_id: Optional[str], memory: Dict[str, Any]) -> None:
        raise NotImplementedError

    def clear(self, chat_id: Optional[str], user_id: Optional[str]) -> None:
        """Clear memory for the given chat_id or user_id."""
        raise NotImplementedError


class _InMemoryStore(MemoryStore):
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def _key(self, chat_id: Optional[str], user_id: Optional[str]) -> Optional[str]:
        return chat_id or user_id

    def load(self, chat_id: Optional[str], user_id: Optional[str], ttl_seconds: int | None = None) -> Dict[str, Any]:
        key = self._key(chat_id, user_id)
        if not key:
            return {}
        rec = self._data.get(key)
        if not rec:
            return {}
        if ttl_seconds:
            ts = rec.get("_ts")
            try:
                fresh = bool(ts and (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() <= ttl_seconds)
            except Exception:
                fresh = False
            if not fresh:
                return {}
        d = {k: v for k, v in rec.items() if not k.startswith("_")}
        return d

    def save(self, chat_id: Optional[str], user_id: Optional[str], memory: Dict[str, Any]) -> None:
        key = self._key(chat_id, user_id)
        if not key:
            return
        m = dict(memory or {})
        m["_ts"] = datetime.now(timezone.utc).isoformat()
        m["_chat_id"] = chat_id
        m["_user_id"] = user_id
        self._data[key] = m

    def clear(self, chat_id: Optional[str], user_id: Optional[str]) -> None:
        key = self._key(chat_id, user_id)
        if not key:
            return
        if key in self._data:
            del self._data[key]


_GLOBAL: MemoryStore | None = None


def global_memory_store() -> MemoryStore:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    _GLOBAL = _InMemoryStore()
    return _GLOBAL
