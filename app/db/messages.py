from typing import List, Dict, Any

from .supabase import get_supabase_client, get_psycopg_connection


def fetch_recent_messages(chat_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return most recent messages for a chat in reverse chronological order (newest first)."""
    try:
        client = get_supabase_client()
        resp = (
            client
            .table("messages")
            .select("id, content, is_ai, created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(max(1, int(limit)))
            .execute()
        )
        return list(resp.data or [])
    except Exception:
        return []


def fetch_relevant_messages(chat_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Return semantically relevant messages using pgvector similarity within the chat."""
    conn = None
    try:
        conn = get_psycopg_connection()
        if conn is None:
            return []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, is_ai, created_at
                FROM messages
                WHERE chat_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (chat_id, query_embedding, max(1, int(limit))),
            )
            rows = cur.fetchall() or []
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append({
                    "id": r[0],
                    "content": r[1],
                    "is_ai": r[2],
                    "created_at": r[3],
                })
            return out
    except Exception:
        return []
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


