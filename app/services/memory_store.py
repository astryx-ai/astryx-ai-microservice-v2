from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from .config import settings

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


class MemoryStore:
    """Abstract interface for chat memory persistence keyed by chat_id/user_id."""

    def load(self, chat_id: Optional[str], user_id: Optional[str], ttl_seconds: int | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, chat_id: Optional[str], user_id: Optional[str], memory: Dict[str, Any]) -> None:
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
        # return copy without internal fields
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


class _SupabaseStore(MemoryStore):
    """Persist memory in Supabase table `chat_memory`.

    Expected schema (create if not exists outside of code):
    - id: bigint (PK) or use chat_id as PK
    - chat_id: text
    - user_id: text
    - memory: jsonb
    - updated_at: timestamptz default now()
    Unique index recommended on (chat_id) and optionally (user_id).
    """

    def __init__(self):
        self._sb = None
        if create_client:
            try:
                self._sb = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            except Exception:
                self._sb = None

    def load(self, chat_id: Optional[str], user_id: Optional[str], ttl_seconds: int | None = None) -> Dict[str, Any]:
        if not self._sb:
            return {}
        try:
            q = None
            if chat_id:
                q = self._sb.table("chat_memory").select("memory,updated_at").eq("chat_id", chat_id).limit(1)
            elif user_id:
                q = self._sb.table("chat_memory").select("memory,updated_at").eq("user_id", user_id).order("updated_at", desc=True).limit(1)
            if not q:
                return {}
            res = q.execute()
            row = res.data[0] if res and res.data else None
            if not row:
                return {}
            if ttl_seconds:
                ts = row.get("updated_at")
                try:
                    fresh = bool(ts and (datetime.now(timezone.utc) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() <= ttl_seconds)
                except Exception:
                    fresh = False
                if not fresh:
                    return {}
            mem = (row.get("memory") or {})
            if not isinstance(mem, dict):
                return {}
            return mem
        except Exception:
            return {}

    def save(self, chat_id: Optional[str], user_id: Optional[str], memory: Dict[str, Any]) -> None:
        if not self._sb:
            return
        try:
            payload = {
                "chat_id": chat_id,
                "user_id": user_id,
                "memory": dict(memory or {}),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if chat_id:
                # Prefer upsert on chat_id if unique
                try:
                    self._sb.table("chat_memory").upsert(payload, on_conflict="chat_id").execute()
                    return
                except Exception:
                    pass
                # Fallback to manual update/insert
                existing = self._sb.table("chat_memory").select("chat_id").eq("chat_id", chat_id).limit(1).execute()
                if existing.data:
                    self._sb.table("chat_memory").update(payload).eq("chat_id", chat_id).execute()
                else:
                    self._sb.table("chat_memory").insert(payload).execute()
            elif user_id:
                # No chat_id, use user_id only (may create multiple rows; latest one will be loaded)
                self._sb.table("chat_memory").insert(payload).execute()
        except Exception:
            return


_GLOBAL: MemoryStore | None = None


def global_memory_store() -> MemoryStore:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    # Try Supabase first; fallback to in-memory
    sb_store = _SupabaseStore()
    if getattr(sb_store, "_sb", None) is not None:
        _GLOBAL = sb_store
    else:
        _GLOBAL = _InMemoryStore()
    return _GLOBAL
