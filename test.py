from supabase import create_client

supabase_url = "https://axxmgwykqdaosbxbbihi.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF4eG1nd3lrcWRhb3NieGJiaWhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTYzMjA5NCwiZXhwIjoyMDcxMjA4MDk0fQ.umMdP5u9kiRL0agcdFvCIUpQR3Hyph4oPE3cnsmt_Ac"
client = create_client(supabase_url, supabase_key)

try:
    response = client.table("news_documents").select("*").limit(1).execute()
    print("Supabase connection successful:", response)
except Exception as e:
    print("Supabase error:", str(e))