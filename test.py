from supabase import create_client

supabase_url = "URI"
supabase_key = "SUPABASE_KEY"
client = create_client(supabase_url, supabase_key)

try:
    response = client.table("news_documents").select("*").limit(1).execute()
    print("Supabase connection successful:", response)
except Exception as e:
    print("Supabase error:", str(e))
