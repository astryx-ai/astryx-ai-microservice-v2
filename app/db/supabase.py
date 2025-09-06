from supabase import create_client, Client
from app.config import settings
import psycopg
from typing import Optional


def get_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def get_psycopg_connection() -> Optional[psycopg.Connection]:
    dsn = getattr(settings, "DATABASE_URL", None)
    if not dsn:
        return None
    return psycopg.connect(dsn)


