import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise ValueError("🚨 啟動失敗：請確認已在 .env 檔案中設定 SUPABASE_URL")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("🚨 啟動失敗：請確認已在 .env 檔案中設定 SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
