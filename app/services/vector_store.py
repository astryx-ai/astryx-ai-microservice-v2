from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from .config import settings
from .azure_openai import embedder

_supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

def news_store():
    return SupabaseVectorStore(
        client=_supabase,
        table_name="news_documents",
        query_name="match_news_wrapper",  # Updated to use wrapper function
        embedding=embedder(),
    )

def stock_store():
    return SupabaseVectorStore(
        client=_supabase,
        table_name="stock_documents",
        query_name="match_stocks_wrapper",  # Updated to use wrapper function
        embedding=embedder(),
    )