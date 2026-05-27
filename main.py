import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from collections import defaultdict, deque
import hashlib
import random
import re
import string
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
import google.generativeai as genai
import numpy as np
import json
import os
import shutil
import tempfile
import traceback
import uuid
import asyncio  # 引入 asyncio 以處理並行任務
import httpx
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
try:
    # Supabase 把 socket/連線層的暫時錯誤（如 Windows WSAEWOULDBLOCK）包成 AuthRetryableError。
    # 必須納入暫時性錯誤清單，不然第一次 auth 打 Supabase 遇到短暫網路抖動就會直接 401。
    # 優先用新名 supabase_auth，舊名 gotrue 已標示棄用但仍可當 fallback。
    try:
        from supabase_auth.errors import AuthRetryableError as _AuthRetryableError
    except ImportError:
        from gotrue.errors import AuthRetryableError as _AuthRetryableError
except Exception:  # 保底，理論上不會走到
    _AuthRetryableError = type("_NoSuchExc", (Exception,), {})

logger = logging.getLogger("uvicorn.error")
from tavily import AsyncTavilyClient
from database import supabase
from youtube_utils import extract_video_id, get_youtube_transcript

# --- 1. 初始化設定與安全防護 ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("🚨 啟動失敗：請確認已在環境變數或 .env 檔案中設定 GEMINI_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("🚨 啟動失敗：請確認已在環境變數或 .env 檔案中設定 TAVILY_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# --- NewebPay 藍新金流 ---
NEWEBPAY_MERCHANT_ID = os.getenv("NEWEBPAY_MERCHANT_ID")
NEWEBPAY_HASH_KEY = os.getenv("NEWEBPAY_HASH_KEY")
NEWEBPAY_HASH_IV = os.getenv("NEWEBPAY_HASH_IV")
# 測試環境預設 https://ccore.newebpay.com/MPG/mpg_gateway，正式環境是 https://core.newebpay.com/MPG/mpg_gateway
NEWEBPAY_URL = os.getenv("NEWEBPAY_URL", "https://ccore.newebpay.com/MPG/mpg_gateway")
NEWEBPAY_RETURN_URL = os.getenv("NEWEBPAY_RETURN_URL", "http://localhost:8000/api/newebpay/webhook")


def _newebpay_require_keys() -> None:
    if not (NEWEBPAY_MERCHANT_ID and NEWEBPAY_HASH_KEY and NEWEBPAY_HASH_IV):
        raise HTTPException(status_code=500, detail="藍新環境變數未設定完整")


def newebpay_encrypt(params: dict) -> str:
    """AES-256-CBC 加密：URL-encoded query string → hex 字串（TradeInfo）。
    HashKey 必須 32 bytes，HashIV 必須 16 bytes，否則 PyCryptodome 會丟 ValueError。"""
    _newebpay_require_keys()
    raw = urllib.parse.urlencode(params)
    cipher = AES.new(
        NEWEBPAY_HASH_KEY.encode("utf-8"),
        AES.MODE_CBC,
        NEWEBPAY_HASH_IV.encode("utf-8"),
    )
    encrypted = cipher.encrypt(pad(raw.encode("utf-8"), AES.block_size))
    return encrypted.hex()


def newebpay_decrypt(hex_str: str) -> dict:
    """AES-256-CBC 解密 webhook 的 TradeInfo（hex）→ JSON dict。
    由於下單時 RespondType=JSON，藍新回拋的明文也是 JSON 字串。"""
    _newebpay_require_keys()
    cipher = AES.new(
        NEWEBPAY_HASH_KEY.encode("utf-8"),
        AES.MODE_CBC,
        NEWEBPAY_HASH_IV.encode("utf-8"),
    )
    decrypted = unpad(cipher.decrypt(bytes.fromhex(hex_str)), AES.block_size)
    return json.loads(decrypted.decode("utf-8"))


def newebpay_trade_sha(trade_info_hex: str) -> str:
    """SHA256("HashKey={KEY}&{TradeInfo}&HashIV={IV}") 後轉大寫；下單與驗 webhook 都用同一條規則。"""
    _newebpay_require_keys()
    raw = f"HashKey={NEWEBPAY_HASH_KEY}&{trade_info_hex}&HashIV={NEWEBPAY_HASH_IV}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()

# 初始化模型 (維持原本的 Gemini 2.5 Flash)
model_text = genai.GenerativeModel('gemini-2.5-flash')
model_json = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

app = FastAPI(title="專屬課程建構室 API (Async 版)")


# 任何非預期例外都用通用訊息回客戶端，避免把 SQL/內部錯誤原文外洩；完整 traceback 仍記錄到 log。
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# 設定 CORS：由環境變數 ALLOWED_ORIGINS 控制（逗號分隔），未設定時 fallback 到本機開發。
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 定義資料格式 ---
class ChatMessage(BaseModel):
    role: str
    content: str

class TAChatRequest(BaseModel):
    question: str
    course_id: str | None = None
    history: list[ChatMessage]

class NoteUpdate(BaseModel):
    content: str | None = None
    pos_x: float | None = None
    pos_y: float | None = None
    width: float | None = None
    height: float | None = None
    is_transparent: bool | None = None

class WhiteboardCreate(BaseModel):
    title: str | None = None
    mode: str | None = None  # 'board' | 'document'（不給就由 DB 預設）

class WhiteboardUpdate(BaseModel):
    title: str | None = None
    mode: str | None = None
    document_content: str | None = None

class DocumentContentUpdate(BaseModel):
    content: str

class ConnectionCreate(BaseModel):
    whiteboard_id: str
    source_note_id: str
    target_note_id: str


def calculate_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def split_into_chunks(text: str, target_size: int = 500, max_size: int = 800) -> list[str]:
    """段落感知切塊：優先在段落邊界切；段落過長才退而求其次在句末標點切；避免在句中硬斷。"""
    text = text.strip()
    if not text:
        return []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        s = buf.strip()
        if len(s) > 10:
            chunks.append(s)
        buf = ""

    for para in paragraphs:
        if len(para) > max_size:
            flush()
            sep_pattern = r"(?<=[。！？!?.\n])"
            pieces = [p for p in re.split(sep_pattern, para) if p.strip()]
            piece_buf = ""
            for piece in pieces:
                if len(piece_buf) + len(piece) > max_size and piece_buf:
                    chunks.append(piece_buf.strip())
                    piece_buf = piece
                else:
                    piece_buf += piece
            if piece_buf.strip():
                if len(piece_buf) < target_size // 2:
                    buf = piece_buf
                else:
                    chunks.append(piece_buf.strip())
            continue

        if len(buf) + len(para) + 2 > target_size and buf:
            flush()
        buf = f"{buf}\n\n{para}" if buf else para

    flush()
    return chunks


# --- 每使用者簡易速率限制（in-memory 滑動視窗） ---
# 注意：僅在單一 uvicorn worker / 單機部署下有效；多 worker 或多機請改用 Redis。
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "generate_course": (5, 3600),   # 每小時 5 次
    "chat":            (60, 3600),  # 每小時 60 次
    "ta_chat":         (60, 3600),  # 每小時 60 次
}
_rate_buckets: dict[tuple[str, str], deque] = defaultdict(deque)
_RATE_SWEEP_INTERVAL = 600.0  # 每 10 分鐘掃一次清空 bucket，避免無限成長
_last_rate_sweep = 0.0


def _maybe_sweep_rate_buckets(now: float) -> None:
    global _last_rate_sweep
    if now - _last_rate_sweep < _RATE_SWEEP_INTERVAL:
        return
    _last_rate_sweep = now
    empty_keys: list[tuple[str, str]] = []
    for k, bucket in _rate_buckets.items():
        _, window = RATE_LIMITS[k[1]]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if not bucket:
            empty_keys.append(k)
    for k in empty_keys:
        _rate_buckets.pop(k, None)


def check_rate_limit(user_id: str, key: str) -> None:
    limit, window = RATE_LIMITS[key]
    now = time.time()
    _maybe_sweep_rate_buckets(now)
    bucket = _rate_buckets[(user_id, key)]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        retry_after = max(1, int(window - (now - bucket[0])))
        raise HTTPException(
            status_code=429,
            detail=f"請求過於頻繁，請於 {retry_after} 秒後再試。",
            headers={"Retry-After": str(retry_after)},
        )
    bucket.append(now)


# 被視為「暫時性網路錯誤」的例外：退讓重試，不要當成 auth 失敗往外丟。
# 典型來源：Windows WinError 10035 (WSAEWOULDBLOCK) / 10054 (RST)、連線逾時、httpx 的 transport 層錯誤、
# gotrue 顯式標為可重試的 AuthRetryableError（會包住底層 socket 錯誤）。
_TRANSIENT_EXC = (OSError, httpx.TransportError, _AuthRetryableError)


async def sb_execute(fn, *, retries: int = 3):
    """把 Supabase 同步呼叫丟進 thread pool，並對暫時性網路錯誤自動退讓重試。其他錯誤直接向外拋。"""
    backoff = 0.1
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(fn)
        except _TRANSIENT_EXC as e:
            last_exc = e
            if attempt == retries - 1:
                break
            await asyncio.sleep(backoff)
            backoff *= 2
    assert last_exc is not None
    raise last_exc


# Gemini gRPC 暫時性錯誤：504 timeout / 503 unavailable / 500 internal / 429 限流。
_GEMINI_TRANSIENT_EXC = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
    google_exceptions.ResourceExhausted,
)


async def gemini_call(coro_factory, *, retries: int = 2, base_backoff: float = 1.5):
    """執行一個 Gemini async 呼叫的 factory，遇到 gRPC 暫時性錯誤自動退讓重試。"""
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except _GEMINI_TRANSIENT_EXC as e:
            last_exc = e
            if attempt == retries:
                break
            wait = base_backoff * (2 ** attempt)
            logger.warning(f"[Gemini] {type(e).__name__} attempt {attempt+1}/{retries+1}, retry in {wait:.1f}s: {e}")
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


async def upload_to_gemini(file: UploadFile | None):
    """把 UploadFile 存到暫存檔，上傳到 Gemini，再刪掉本地檔。回傳 Gemini File 物件或 None。"""
    if file is None:
        return None
    suffix = os.path.splitext(file.filename or "")[1] or ""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        uploaded = await asyncio.to_thread(
            lambda: genai.upload_file(path=temp_path, mime_type=file.content_type or None)
        )
        return uploaded
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


async def delete_gemini_file(uploaded) -> None:
    """刪掉 Gemini 端的上傳檔（避免累積儲存成本）；失敗只記 log，不影響流程。"""
    if uploaded is None:
        return
    name = getattr(uploaded, "name", None)
    if not name:
        return
    try:
        await asyncio.to_thread(lambda: genai.delete_file(name))
    except Exception as e:
        logger.warning(f"[Gemini] 刪除上傳檔案 {name} 失敗：{e}")


# --- Auth 依賴：驗證 Bearer Token ---
bearer_scheme = HTTPBearer(auto_error=False)

# 短期用戶查詢快取：驟減對 Supabase /auth/v1/user 的打擊面。
# JWT 本身有 exp，60s 內同 token 幾乎不可能剛好失效；若使用者被刪除/權限變動，最多延遲 60s 生效。
_AUTH_CACHE_TTL = 60.0
_auth_cache: dict[str, tuple[float, object]] = {}
_AUTH_SWEEP_INTERVAL = 300.0  # 每 5 分鐘清一次過期 token，避免 dict 無限成長
_last_auth_sweep = 0.0


def _maybe_sweep_auth_cache(now: float) -> None:
    global _last_auth_sweep
    if now - _last_auth_sweep < _AUTH_SWEEP_INTERVAL:
        return
    _last_auth_sweep = now
    expired = [k for k, (ts, _) in _auth_cache.items() if now - ts > _AUTH_CACHE_TTL]
    for k in expired:
        _auth_cache.pop(k, None)


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is None or not credentials.credentials:
        logger.warning("[Auth] 401 A: 沒帶 Bearer Token (credentials is None)")
        raise HTTPException(status_code=401, detail="未提供 Bearer Token")
    token = credentials.credentials

    now = time.time()
    _maybe_sweep_auth_cache(now)

    # 快取命中就不打 Supabase
    cached = _auth_cache.get(token)
    if cached is not None:
        ts, cached_user = cached
        if now - ts < _AUTH_CACHE_TTL:
            return cached_user
        _auth_cache.pop(token, None)

    try:
        user_resp = await sb_execute(lambda: supabase.auth.get_user(token))
    except _TRANSIENT_EXC as e:
        logger.warning(f"[Auth] 503 B: 暫時性錯誤 ({type(e).__name__}): {e}")
        raise HTTPException(status_code=503, detail=f"Auth 服務暫時不可用：{e}")
    except Exception as e:
        logger.warning(
            f"[Auth] 401 C: 非預期錯誤 ({type(e).__module__}.{type(e).__name__}): {e!r}"
        )
        raise HTTPException(status_code=401, detail=f"Token 驗證失敗：{e}")
    user = getattr(user_resp, "user", None)
    if user is None:
        logger.warning(f"[Auth] 401 D: Supabase 回應沒有 user ({user_resp!r})")
        raise HTTPException(status_code=401, detail="無效的 Token 或使用者不存在")

    _auth_cache[token] = (now, user)
    return user

# --- 3. API 端點 ---

@app.post("/api/chat")
async def chat_endpoint(
    messages: str = Form(...),       # JSON-encoded list of ChatMessage
    new_prompt: str = Form(...),
    file: UploadFile | None = File(None),
    user=Depends(get_current_user),
):
    """階段一：與課程顧問聊天（串流，可選擇附加檔案）"""
    check_rate_limit(user.id, "chat")
    # Free 版禁止附檔，沒帶檔就略過這個查詢以免每個訊息都打 DB
    if file is not None:
        await require_premium(user.id, "PREMIUM_FEATURE_FILE_UPLOAD")
    try:
        try:
            parsed_messages = [ChatMessage(**m) for m in json.loads(messages)]
        except Exception:
            raise HTTPException(status_code=400, detail="messages 欄位必須是有效的 JSON 陣列")

        gemini_history = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [m.content]}
            for m in parsed_messages
        ]
        chat_session = model_text.start_chat(history=gemini_history)

        uploaded_file = await upload_to_gemini(file)
        content = [new_prompt, uploaded_file] if uploaded_file else new_prompt

        async def token_stream():
            try:
                try:
                    response = await chat_session.send_message_async(content, stream=True)
                    async for chunk in response:
                        text = getattr(chunk, "text", None)
                        if text:
                            yield text
                except _GEMINI_TRANSIENT_EXC as e:
                    # 串流迭代中才會觸發的 503/504 等暫時錯誤：gemini_call 無法覆蓋到這層。
                    # 若直接 raise 客戶端會看到 connection reset（網頁顯示 network error），
                    # 改成把錯誤當成一段文字送出去，讓聊天氣泡顯示可讀訊息。
                    logger.warning(f"[chat stream] Gemini transient: {type(e).__name__}: {e}")
                    yield f"\n\n⚠️ 模型目前服務繁忙（{type(e).__name__}），請稍後再試。"
                except Exception as e:
                    logger.exception("[chat stream] Unexpected error")
                    yield f"\n\n⚠️ 發生未預期錯誤：{type(e).__name__}"
            finally:
                # 串流完成 / client 提前斷線都要清 Gemini 端暫存檔
                await delete_gemini_file(uploaded_file)

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.get("/api/courses")
async def list_courses(
    user=Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """列出當前使用者的課程（支援 limit / offset 分頁）。"""
    try:
        res = await sb_execute(
            lambda: supabase.table("courses")
                .select("id, course_title, created_at")
                .eq("user_id", user.id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
        )
        return res.data or []
    except Exception:
        raise


@app.delete("/api/courses/{course_id}")
async def delete_course(course_id: str, user=Depends(get_current_user)):
    """刪除課程（course_chunks 透過 FK CASCADE 自動清除）。關聯白板不會被動到，使用者可另外刪除。"""
    try:
        res = await sb_execute(
            lambda: supabase.table("courses")
                .delete()
                .eq("id", course_id)
                .eq("user_id", user.id)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Course not found")
        return {"ok": True, "id": course_id}
    except HTTPException:
        raise
    except Exception:
        raise


@app.post("/api/courses/{course_id}/to-note")
async def course_to_note(course_id: str, user=Depends(get_current_user)):
    """把課程講義交給 Gemini 萃取成「重點筆記」，再以 document 模式存入 whiteboards。
    不再直接複製 markdown，而是 AI 整理後的精華。"""
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        res = await sb_execute(
            lambda: supabase.table("courses")
                .select("course_title, markdown")
                .eq("id", course_id)
                .eq("user_id", user.id)
                .limit(1)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Course not found")

        title = (rows[0].get("course_title") or "").strip() or "未命名筆記"
        markdown = rows[0].get("markdown") or ""
        if not markdown.strip():
            raise HTTPException(status_code=400, detail="此課程沒有可萃取的內容")

        # 交給 Gemini 萃取重點。用 model_text（純文字）而非 model_json。
        summary_prompt = (
            "你是一位專業的助教。請將以下課程講義濃縮成條理分明的「重點筆記」。"
            "請大量使用 Markdown 的列點（Bullet points）、粗體來標示關鍵字，"
            "並幫忙整理出「課後複習重點」與「核心概念懶人包」。"
            "請直接輸出 Markdown 純文字，不要使用 ```markdown 標籤包覆。"
            f"講義內容：\n{markdown}"
        )
        try:
            resp = await gemini_call(
                lambda: model_text.generate_content_async(
                    summary_prompt,
                    request_options={"timeout": 180},
                ),
            )
        except (google_exceptions.RetryError, google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded, google_exceptions.ResourceExhausted) as e:
            logger.warning(f"[to-note] Gemini busy: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=503,
                detail="模型服務目前繁忙或暫時不可用，請稍後再試。",
            )
        notes_md = (resp.text or "").strip()

        # 清掉模型可能自帶的 ```markdown ... ``` 包覆（即使提示要求過也常常還是會包）
        if notes_md.startswith("```"):
            # 砍掉開頭的 ```xxx\n
            first_newline = notes_md.find("\n")
            if first_newline != -1:
                notes_md = notes_md[first_newline + 1:]
            # 砍掉結尾的 ```
            if notes_md.rstrip().endswith("```"):
                notes_md = notes_md.rstrip()[:-3].rstrip()
        notes_md = notes_md.strip()

        if not notes_md:
            raise HTTPException(status_code=502, detail="AI 無法產生重點筆記，請稍後重試")

        ins = await sb_execute(
            lambda: supabase.table("whiteboards").insert({
                "user_id": user.id,
                "title": title,
                "mode": "document",
                "document_content": notes_md,
            }).execute()
        )
        new_rows = ins.data or []
        if not new_rows:
            raise HTTPException(status_code=500, detail="建立筆記失敗")
        return {"whiteboard_id": new_rows[0]["id"]}
    except HTTPException:
        raise
    except Exception:
        raise


@app.get("/api/courses/{course_id}")
async def get_course(course_id: str, user=Depends(get_current_user)):
    """取得單一課程完整內容（僅限擁有者）。"""
    try:
        res = await sb_execute(
            lambda: supabase.table("courses")
                .select("id, course_title, course_data, markdown, refs, created_at")
                .eq("id", course_id)
                .eq("user_id", user.id)
                .limit(1)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Course not found")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise


FREE_COURSE_LIMIT = 5


async def get_user_plan(user_id: str) -> dict:
    """回傳 { is_premium, course_count, free_limit }。course_count 只算「本月（台北時區）」新生成的課程，
    每個月 1 號 00:00 (TPE) 重新計數。"""
    is_premium = False
    try:
        prof_res = await sb_execute(
            lambda: supabase.table("profiles").select("is_premium").eq("id", user_id).limit(1).execute()
        )
        rows = prof_res.data or []
        if rows:
            is_premium = bool(rows[0].get("is_premium"))
    except Exception as e:
        print(f"[Plan] 查詢 profile 失敗：{e}")

    # 本月第一天 00:00 (TPE) 的 ISO 字串。created_at 是 timestamptz，Postgres 會自動處理時區轉換。
    tpe = timezone(timedelta(hours=8))
    month_start_iso = datetime.now(tpe).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

    try:
        c_res = await sb_execute(
            lambda: supabase.table("courses")
                .select("id")
                .eq("user_id", user_id)
                .gte("created_at", month_start_iso)
                .execute()
        )
        course_count = len(c_res.data or [])
    except Exception as e:
        print(f"[Plan] 查詢 courses 數量失敗：{e}")
        course_count = 0

    return {"is_premium": is_premium, "course_count": course_count, "free_limit": FREE_COURSE_LIMIT}


async def require_premium(user_id: str, feature_code: str) -> None:
    """不是 Premium 就拋 403，detail 是一個給前端辨識的 code（如 PREMIUM_FEATURE_WHITEBOARD）。"""
    plan = await get_user_plan(user_id)
    if not plan.get("is_premium"):
        raise HTTPException(status_code=403, detail=feature_code)


@app.get("/api/user/plan")
async def user_plan(user=Depends(get_current_user)):
    return await get_user_plan(user.id)


@app.post("/api/generate-course")
async def generate_course(
    syllabus: str = Form(...),
    teaching_style: str = Form(...),
    video_url: Optional[str] = Form(None),
    file: UploadFile | None = File(None),
    user=Depends(get_current_user),
):
    """階段二：依照使用者填寫的課綱與風格，正式生成課程與助教向量庫（可附參考檔案 / YouTube 影片）"""
    check_rate_limit(user.id, "generate_course")
    # --- 額度檢查 ---
    plan = await get_user_plan(user.id)
    if not plan["is_premium"] and plan["course_count"] >= FREE_COURSE_LIMIT:
        raise HTTPException(status_code=403, detail="FREE_LIMIT_REACHED")
    # Free 版禁止附檔（重用剛才查過的 plan，不再 round-trip）
    if file is not None and not plan["is_premium"]:
        raise HTTPException(status_code=403, detail="PREMIUM_FEATURE_FILE_UPLOAD")

    try:
        # 1. 定義授課風格指南
        style_guide = {
            "條理": "請以條理分明、步驟清晰的方式進行教學，多使用列點與結構化的排版。",
            "輕鬆": "請以幽默、輕鬆、口語化的語氣進行教學，多使用生活化的比喻，甚至可以帶點趣味。",
            "簡單介紹": "請以最精簡、易懂的方式概述核心概念，不要過度深入艱澀的技術細節，適合初學者。",
            "專業講解": "請以學術、專業的角度深入探討，正確使用專業術語，並提供嚴謹的定義與原理解釋。"
        }
        current_style = style_guide.get(teaching_style, style_guide["條理"])

        # 2. 若有附 YouTube 網址，抓字幕作為額外參考（最多 ~20000 字避免 token 超載）
        # Free 版禁用 YouTube：複用前面已查過的 plan，不再多打一次 DB
        # YouTube 付費 gate 留在最上面：免費版若帶了 video_url 就立刻 403，不要白白打 Tavily。
        if video_url and video_url.strip() and not plan["is_premium"]:
            raise HTTPException(status_code=403, detail="PREMIUM_FEATURE_YOUTUBE")

        # 2. 先向 Tavily 搜尋最新網路資料
        # 優先用「主題：」那行作為 query，否則用整段課綱
        query_source = syllabus.strip()
        first_line = query_source.splitlines()[0] if query_source else ""
        if first_line.startswith("主題："):
            tavily_query = first_line.replace("主題：", "", 1).strip() or query_source
        else:
            tavily_query = query_source[:300]

        tavily_context_block = "（本次未取得網路參考資料）"
        tavily_urls: list[str] = []
        try:
            tavily_result = await tavily_client.search(
                query=tavily_query,
                search_depth="basic",
                max_results=5,
            )
            hits = tavily_result.get("results", []) if isinstance(tavily_result, dict) else []
            if hits:
                tavily_context_block = "\n\n".join(
                    f"[{i+1}] {h.get('title','(無標題)')}\nURL: {h.get('url','')}\n內容摘要：{h.get('content','')}"
                    for i, h in enumerate(hits)
                )
                tavily_urls = [h.get("url", "") for h in hits if h.get("url")]
        except Exception as tav_e:
            print(f"Tavily 搜尋失敗（不中斷課程生成）：{tav_e}")

        # 給 Gemini 的「可用網址清單」：先放 Tavily 結果，YouTube 字幕成功取得後再 append video_url。
        # 這樣模型才看得到 YouTube 網址當候選，並能放進 references。
        all_reference_urls = tavily_urls.copy()

        # 3. 若有附 YouTube 網址，抓字幕作為額外參考（最多 ~20000 字避免 token 超載）
        video_context = ""
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                transcript = await get_youtube_transcript(video_id)
                if transcript:
                    video_context = transcript[:20000]
                    all_reference_urls.append(video_url)
                else:
                    print(f"[YouTube] 影片 {video_id} 無可用字幕或抓取失敗，略過。")
            else:
                print(f"[YouTube] 無法解析網址 Video ID：{video_url}")

        video_block = (
            f"\n        【YouTube 影片逐字稿】（重要參考資料，請依其實際內容豐富課程細節）：\n"
            f"        {video_context}\n"
            if video_context else ""
        )

        # 4. 組裝給 Gemini 的 Prompt
        url_list_text = "\n".join(f"- {u}" for u in all_reference_urls) if all_reference_urls else "（無）"
        final_prompt = f"""
        你是一位專業的課程講師。請依照以下【指定課綱】與【授課風格】撰寫課程。
        絕對遵守 JSON 格式輸出。不要包含任何 Markdown 標籤（如 ```json）。

        【授課風格】：{teaching_style}
        {current_style}

        【指定課綱】：
        {syllabus}

        【最新網路參考資料】：
        {tavily_context_block}

        【可用網址清單】（你必須從中挑選至少一個你實際參考的 URL 放入 references 欄位）：
        {url_list_text}
        如果清單中包含 YouTube 網址，且你有參考其逐字稿，請務必將該 YouTube 網址包含在 references 陣列中。
{video_block}
        【視覺化要求】：
        為了讓課程更生動，請在適當的段落加入視覺輔助：
        - 圖表與關係圖：若遇到流程、架構或比較，請務必使用 Mermaid.js 語法（以 ```mermaid 開頭的程式碼區塊）來繪製圖表。
          當你生成 Mermaid 圖表語法時，請嚴格遵守語法規範。最重要的一點：節點的文字描述若包含**任何空白、括號或特殊符號（含中文標點、冒號、連字號、斜線等）**，請務必使用**雙引號**將其包起來（例如 `A["這是一個測試(附註)"]` 或 `A["步驟 1：設定"]`）。請盡量使用最基本且穩定的 `graph TD`、`flowchart LR` 或 `mindmap` 結構，避免罕見語法。節點 ID 只用英數字（A, B, C1, step2 這類），不要用中文當 ID。
        - 情境配圖：請在每個主要章節的開頭或需要具體畫面的地方，插入 1~2 張情境圖片。請使用 Markdown 語法，網址格式為：![圖片替代文字](https://image.pollinations.ai/prompt/{{用英文精準描述圖片畫面,以逗號分隔}}?width=800&height=400&nologo=true)
        （請把 {{...}} 換成實際的英文描述，整段網址不要有空白。）
        這些視覺化元素請放進 modules 的 detailed_content 欄位（仍為 Markdown 字串）。

        === 輸出格式（強制）===
        必須輸出一個 JSON 物件，**所有欄位都不可省略**：
        - course_title: string
        - learning_objectives: string 或 string 陣列
        - introduction: string
        - modules: 陣列，每個元素包含 module_title (string), detailed_content (string), key_takeaways (string 或 string 陣列)
        - conclusion: string
        - next_learning_steps: string
        - references: string 陣列。**這個欄位是必填**，請從上方【可用網址清單】中挑出你實際用於撰寫本課程的 URL（通常 2~5 個），照原樣放入。絕對不可省略此欄位，也不可回傳空陣列（除非清單為空）。

        請務必參考【最新網路參考資料】與（若有）【YouTube 影片逐字稿】來豐富課程細節（例如最新案例、工具、版本、實務建議）。
        """

        # 5. 生成課程 JSON 內容（若有附檔，一併傳給 Gemini）
        # timeout 180s：Gemini SDK 內建會自動在 503 期間持續重試直到 timeout，
        # 原本 600s 會讓使用者乾等 10 分鐘後才收到失敗，縮短到 3 分鐘比較合理。
        uploaded_file = await upload_to_gemini(file)
        try:
            gemini_content = [final_prompt, uploaded_file] if uploaded_file else final_prompt
            raw_response_obj = await gemini_call(
                lambda: model_json.generate_content_async(
                    gemini_content,
                    request_options={"timeout": 180},
                ),
            )
            raw_response = raw_response_obj.text
        finally:
            # 不管成功失敗都清 Gemini 端暫存檔
            await delete_gemini_file(uploaded_file)

        # 清理 JSON 字串防呆
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        try:
            course_data = json.loads(clean_json)
        except json.JSONDecodeError as je:
            # Gemini 雖然設了 response_mime_type=application/json 但偶爾還是會吐出帶語法錯誤的 JSON
            # （例如長文 array 漏逗號）。轉成 502 給前端顯示友善訊息，並把原文記到 log 以便除錯。
            logger.warning(f"[generate_course] Gemini 回傳無效 JSON：{je}")
            logger.warning(f"[generate_course] Raw output（前 1500 字）：{clean_json[:1500]}")
            raise HTTPException(
                status_code=502,
                detail="AI 回應格式錯誤，請重新生成（這通常是模型一時抽風，再試一次就好）。",
            )

        # 保底：若 Gemini 沒給 references 或給空陣列，回填 Tavily 的 URL
        existing_refs = course_data.get("references")
        if (not isinstance(existing_refs, list) or not existing_refs) and all_reference_urls:
            print("⚠️ Gemini 未輸出 references，回填 Tavily + YouTube URL 作為保底。")
            course_data["references"] = all_reference_urls

        # 強制防呆：先確保 references 是合法 list；再檢查 YouTube 網址。
        # 即使 Gemini 只回了 Tavily 那幾條而漏了 YouTube，只要我們前面確實抓到字幕，
        # 就強制 append 進去——不依賴模型自律。
        # 用 video_context 作為「字幕成功取得」的信號（成功時被指派為 transcript[:20000]，否則保持空字串）。
        if not isinstance(course_data.get("references"), list):
            course_data["references"] = []
        if video_url and video_context and video_url not in course_data["references"]:
            print(f"⚠️ Gemini 漏掉 YouTube 網址，強制補回：{video_url}")
            course_data["references"].append(video_url)

        # 6. 組裝 Markdown 講義
        md_export = f"# {course_data.get('course_title', '')}\n\n## 課程前言\n{course_data.get('introduction', '')}\n\n"
        for i, m in enumerate(course_data.get("modules", [])):
            md_export += f"## 章節 {i+1}：{m.get('module_title')}\n\n{m.get('detailed_content')}\n\n"
        md_export += f"## 課程總結\n{course_data.get('conclusion', '')}\n\n"
        md_export += f"## 下一步的學習建議\n{course_data.get('next_learning_steps', '')}\n\n"

        references = course_data.get("references") or []
        if isinstance(references, list) and references:
            md_export += "## 參考資料\n"
            for url in references:
                if isinstance(url, str) and url.strip():
                    md_export += f"- [{url}]({url})\n"
            md_export += "\n"

        # 7. 建立文字片段 (Chunk) 以供助教 RAG 系統使用
        chunks = split_into_chunks(md_export, target_size=500, max_size=800)

        # 8. 並行產生 Embeddings (建議未來可升級為 text-embedding-004)
        async def get_embedding(text_chunk):
            try:
                res = await gemini_call(
                    lambda: genai.embed_content_async(
                        model="models/gemini-embedding-001",
                        content=text_chunk,
                        request_options={"timeout": 60},
                    ),
                )
                return {"text": text_chunk, "embedding": res['embedding']}
            except Exception as emb_e:
                logger.warning(f"Embedding 錯誤 (段落前20字: {text_chunk[:20]}...): {emb_e}")
                return None

        embedding_tasks = [get_embedding(c) for c in chunks]
        embedding_results = await asyncio.gather(*embedding_tasks)

        # 過濾掉失敗的 Embedding
        ta_db = [res for res in embedding_results if res is not None]

        # 9. 寫入 Supabase（失敗不中斷回傳）
        course_id = None
        try:
            insert_res = await sb_execute(
                lambda: supabase.table("courses").insert({
                    "course_title": course_data.get("course_title", ""),
                    "course_data": course_data,
                    "markdown": md_export,
                    "refs": course_data.get("references") or [],
                    "user_id": user.id,
                }).execute()
            )
            rows = getattr(insert_res, "data", None) or []
            if rows:
                course_id = rows[0].get("id")
        except Exception as db_e:
            print(f"⚠️ 寫入 Supabase courses 失敗：{db_e}")

        # 9.5 寫入 course_chunks（RAG 向量庫，僅在 course 已建立後）
        if course_id and ta_db:
            try:
                chunk_rows = [
                    {
                        "course_id": course_id,
                        "user_id": user.id,
                        "text": item["text"],
                        "embedding": item["embedding"],
                    }
                    for item in ta_db
                ]
                await sb_execute(
                    lambda: supabase.table("course_chunks").insert(chunk_rows).execute()
                )
            except Exception as db_e:
                print(f"⚠️ 寫入 course_chunks 失敗：{db_e}")

        # 不再於課程生成後自動建立白板/筆記。
        # 使用者要產生筆記時，透過 /api/courses/{id}/to-note 端點手動觸發。
        return {
            "course_id": course_id,
            "course_data": course_data,
            "markdown": md_export,
        }
    except HTTPException:
        raise
    except (google_exceptions.RetryError, google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded, google_exceptions.ResourceExhausted) as e:
        # Gemini 真的繁忙／超時到我們這層了。翻成 503 讓前端可以辨識並顯示友善訊息。
        logger.warning(f"[generate_course] Gemini busy: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=503,
            detail="模型服務目前繁忙或暫時不可用，請稍後再試。",
        )
    except Exception:
        print("====== Generate Course 發生錯誤 ======")
        print(traceback.format_exc())
        raise


@app.post("/api/ta-chat")
async def ta_chat(request: TAChatRequest, user=Depends(get_current_user)):
    """階段三：隨堂 AI 助教問答 (RAG，串流)"""
    await require_premium(user.id, "PREMIUM_FEATURE_TA_CHAT")
    check_rate_limit(user.id, "ta_chat")
    try:
        top_context = "無參考講義。"

        # 1. 從 DB 撈取此課程的 chunks，計算餘弦相似度取 top-2
        if request.course_id:
            try:
                chunks_res = await sb_execute(
                    lambda: supabase.table("course_chunks")
                        .select("text, embedding")
                        .eq("course_id", request.course_id)
                        .eq("user_id", user.id)
                        .execute()
                )
                chunks = chunks_res.data or []
            except Exception as db_e:
                print(f"⚠️ 讀取 course_chunks 失敗：{db_e}")
                chunks = []

            if chunks:
                emb_res = await gemini_call(
                    lambda: genai.embed_content_async(
                        model="models/gemini-embedding-001",
                        content=request.question,
                        request_options={"timeout": 60},
                    ),
                )
                question_emb = emb_res['embedding']
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: calculate_similarity(x["embedding"], question_emb),
                    reverse=True,
                )
                top_context = "\n\n---\n\n".join(c["text"] for c in ranked_chunks[:2])

        # 2. 重建 Gemini 歷史對話紀錄
        gemini_ta_history = []
        for m in request.history:
            if "我是隨堂助教" in m.content:
                continue
            role = "model" if m.role == "assistant" else "user"
            gemini_ta_history.append({"role": role, "parts": [m.content]})

        # 3. 設定助教的 System Prompt 並發送問題
        ta_sys_inst = "你是一位專業且熱心的隨堂助教。回答問題時，請【優先】根據使用者提供的【參考講義片段】來解答。如果講義片段中沒有包含相關資訊，或者不足以完整回答，請運用你自身的知識來為學生解答。但使用自身知識時，請務必在回答中稍微提一下（例如：「這部分雖然不在目前的講義範圍內，但...」或「講義雖然沒特別提到，不過...」），以便讓學生清楚區分資訊的來源。"
        ta_chat_session = genai.GenerativeModel('gemini-2.5-flash', system_instruction=ta_sys_inst).start_chat(history=gemini_ta_history)

        user_msg = f"【參考講義片段】\n{top_context}\n\n【學生的問題】\n{request.question}"

        async def token_stream():
            parts: list[str] = []
            try:
                response = await ta_chat_session.send_message_async(user_msg, stream=True)
                async for chunk in response:
                    text = getattr(chunk, "text", None)
                    if text:
                        parts.append(text)
                        yield text
            except _GEMINI_TRANSIENT_EXC as e:
                logger.warning(f"[ta stream] Gemini transient: {type(e).__name__}: {e}")
                err_text = f"\n\n⚠️ 助教暫時無法回應（{type(e).__name__}），請稍後再試。"
                parts.append(err_text)
                yield err_text
            except Exception as e:
                logger.exception("[ta stream] Unexpected error")
                err_text = f"\n\n⚠️ 發生未預期錯誤：{type(e).__name__}"
                parts.append(err_text)
                yield err_text
            # 串流結束後（含錯誤降級）寫入 Supabase chat_history；失敗不影響已送出的回覆
            try:
                await sb_execute(
                    lambda: supabase.table("chat_history").insert({
                        "question": request.question,
                        "answer": "".join(parts),
                        "user_id": user.id,
                    }).execute()
                )
            except Exception as db_e:
                print(f"⚠️ 寫入 Supabase chat_history 失敗：{db_e}")

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")

    except HTTPException:
        raise
    except Exception as e:
        print("====== TA Chat 發生錯誤 ======")
        print(traceback.format_exc())
        raise


# --- 白板 Whiteboards ---

@app.get("/api/whiteboards")
async def list_whiteboards(
    user=Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        res = await sb_execute(
            lambda: supabase.table("whiteboards")
                .select("id, title, mode, created_at")
                .eq("user_id", user.id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
        )
        return res.data or []
    except Exception:
        raise


@app.post("/api/whiteboards")
async def create_whiteboard(payload: WhiteboardCreate, user=Depends(get_current_user)):
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        row: dict = {"user_id": user.id}
        if payload.title and payload.title.strip():
            row["title"] = payload.title.strip()
        if payload.mode in ("board", "document"):
            row["mode"] = payload.mode
        res = await sb_execute(
            lambda: supabase.table("whiteboards").insert(row).execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=500, detail="建立白板失敗")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.patch("/api/whiteboards/{whiteboard_id}")
async def update_whiteboard(
    whiteboard_id: str,
    payload: WhiteboardUpdate,
    user=Depends(get_current_user),
):
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    update: dict = {}
    if payload.title is not None:
        t = payload.title.strip()
        if not t:
            raise HTTPException(status_code=400, detail="title 不可為空")
        update["title"] = t
    if payload.mode is not None:
        if payload.mode not in ("board", "document"):
            raise HTTPException(status_code=400, detail="mode 必須是 board 或 document")
        update["mode"] = payload.mode
    if payload.document_content is not None:
        update["document_content"] = payload.document_content
    if not update:
        raise HTTPException(status_code=400, detail="沒有要更新的欄位")
    try:
        res = await sb_execute(
            lambda: supabase.table("whiteboards")
                .update(update)
                .eq("id", whiteboard_id)
                .eq("user_id", user.id)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Whiteboard not found")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.patch("/api/whiteboards/{whiteboard_id}/document")
async def update_whiteboard_document(
    whiteboard_id: str,
    payload: DocumentContentUpdate,
    user=Depends(get_current_user),
):
    """專門更新文字模式的 document_content（BlockNote JSON 字串）。"""
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        res = await sb_execute(
            lambda: supabase.table("whiteboards")
                .update({"document_content": payload.content})
                .eq("id", whiteboard_id)
                .eq("user_id", user.id)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Whiteboard not found")
        return {"ok": True, "id": whiteboard_id}
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.delete("/api/whiteboards/{whiteboard_id}")
async def delete_whiteboard(whiteboard_id: str, user=Depends(get_current_user)):
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        # 先抓屬於這個白板、這個使用者的 notes 中有圖片的 URL，才能一併清 Storage
        notes_res = await sb_execute(
            lambda: supabase.table("notes")
                .select("image_url")
                .eq("user_id", user.id)
                .eq("whiteboard_id", whiteboard_id)
                .execute()
        )
        image_paths: list[str] = []
        for n in (notes_res.data or []):
            url = n.get("image_url")
            if not url:
                continue
            marker = "/note_images/"
            idx = url.find(marker)
            if idx != -1:
                image_paths.append(url[idx + len(marker):].split("?", 1)[0])

        if image_paths:
            try:
                await sb_execute(
                    lambda: supabase.storage.from_("note_images").remove(image_paths)
                )
            except Exception as se:
                print(f"⚠️ Storage 批次刪除失敗：{se}")

        # 刪除白板；notes 透過 ON DELETE CASCADE 自動清除
        del_res = await sb_execute(
            lambda: supabase.table("whiteboards")
                .delete()
                .eq("id", whiteboard_id)
                .eq("user_id", user.id)
                .execute()
        )
        rows = del_res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Whiteboard not found")
        return {"ok": True, "id": whiteboard_id}
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.get("/api/whiteboards/{whiteboard_id}/bundle")
async def get_whiteboard_bundle(whiteboard_id: str, user=Depends(get_current_user)):
    """一次取回白板、底下所有 notes 與 connections。"""
    await require_premium(user.id, "PREMIUM_FEATURE_WHITEBOARD")
    try:
        wb = await sb_execute(
            lambda: supabase.table("whiteboards")
                .select("id, title, mode, document_content, created_at")
                .eq("id", whiteboard_id).eq("user_id", user.id).limit(1).execute()
        )
        wb_rows = wb.data or []
        if not wb_rows:
            raise HTTPException(status_code=404, detail="Whiteboard not found")

        notes = await sb_execute(
            lambda: supabase.table("notes")
                .select("id, content, image_url, pos_x, pos_y, width, height, is_transparent, created_at")
                .eq("user_id", user.id).eq("whiteboard_id", whiteboard_id)
                .order("created_at", desc=False).execute()
        )
        conns = await sb_execute(
            lambda: supabase.table("note_connections")
                .select("id, source_note_id, target_note_id, created_at")
                .eq("whiteboard_id", whiteboard_id).execute()
        )
        return {
            "whiteboard": wb_rows[0],
            "notes": notes.data or [],
            "connections": conns.data or [],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.post("/api/connections")
async def create_connection(payload: ConnectionCreate, user=Depends(get_current_user)):
    if payload.source_note_id == payload.target_note_id:
        raise HTTPException(status_code=400, detail="不能連到自己")
    try:
        # 確認白板擁有權
        wb = await sb_execute(
            lambda: supabase.table("whiteboards").select("id")
                .eq("id", payload.whiteboard_id).eq("user_id", user.id).limit(1).execute()
        )
        if not (wb.data or []):
            raise HTTPException(status_code=404, detail="Whiteboard not found")

        res = await sb_execute(
            lambda: supabase.table("note_connections").insert({
                "whiteboard_id": payload.whiteboard_id,
                "source_note_id": payload.source_note_id,
                "target_note_id": payload.target_note_id,
            }).execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=500, detail="建立連線失敗")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        # 可能是 unique (source, target) 衝突
        raise


@app.delete("/api/connections/{connection_id}")
async def delete_connection(connection_id: str, user=Depends(get_current_user)):
    try:
        # 只能刪到自己白板上的連線：透過 join 的方式先查 whiteboard_id 再驗證
        existing = await sb_execute(
            lambda: supabase.table("note_connections").select("id, whiteboard_id")
                .eq("id", connection_id).limit(1).execute()
        )
        rows = existing.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Connection not found")

        wb = await sb_execute(
            lambda: supabase.table("whiteboards").select("id")
                .eq("id", rows[0]["whiteboard_id"]).eq("user_id", user.id).limit(1).execute()
        )
        if not (wb.data or []):
            raise HTTPException(status_code=404, detail="Connection not found")

        await sb_execute(
            lambda: supabase.table("note_connections").delete().eq("id", connection_id).execute()
        )
        return {"ok": True, "id": connection_id}
    except HTTPException:
        raise
    except Exception as e:
        raise


# --- 白板筆記 Notes ---

@app.post("/api/notes")
async def create_note(
    content: str = Form(""),
    whiteboard_id: str = Form(...),
    pos_x: float = Form(0.0),
    pos_y: float = Form(0.0),
    width: float = Form(200.0),
    height: float = Form(150.0),
    is_transparent: bool = Form(False),
    file: UploadFile | None = File(None),
    user=Depends(get_current_user),
):
    image_url: str | None = None
    if file is not None:
        try:
            file_bytes = await file.read()
            ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
            storage_path = f"{user.id}/{uuid.uuid4().hex}{ext}"
            await sb_execute(
                lambda: supabase.storage.from_("note_images").upload(
                    path=storage_path,
                    file=file_bytes,
                    file_options={"content-type": file.content_type or "application/octet-stream"},
                )
            )
            image_url = supabase.storage.from_("note_images").get_public_url(storage_path)
        except Exception as e:
            logger.exception("圖片上傳失敗")
            raise HTTPException(status_code=500, detail="圖片上傳失敗")

    try:
        res = await sb_execute(
            lambda: supabase.table("notes").insert({
                "user_id": user.id,
                "whiteboard_id": whiteboard_id,
                "content": content or "",
                "image_url": image_url,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "width": width,
                "height": height,
                "is_transparent": is_transparent,
            }).execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=500, detail="寫入 notes 失敗")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.get("/api/notes/{whiteboard_id}")
async def list_notes(whiteboard_id: str, user=Depends(get_current_user)):
    try:
        res = await sb_execute(
            lambda: supabase.table("notes")
                .select("id, content, image_url, pos_x, pos_y, width, height, is_transparent, created_at")
                .eq("user_id", user.id)
                .eq("whiteboard_id", whiteboard_id)
                .order("created_at", desc=False)
                .execute()
        )
        return res.data or []
    except Exception as e:
        raise


@app.patch("/api/notes/{note_id}")
async def update_note(
    note_id: str,
    payload: NoteUpdate,
    user=Depends(get_current_user),
):
    update_fields = payload.model_dump(exclude_none=True)
    if not update_fields:
        raise HTTPException(status_code=400, detail="沒有要更新的欄位")
    try:
        res = await sb_execute(
            lambda: supabase.table("notes")
                .update(update_fields)
                .eq("id", note_id)
                .eq("user_id", user.id)
                .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Note not found")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, user=Depends(get_current_user)):
    try:
        # 先撈出這筆筆記（確認擁有權 + 取 image_url）
        get_res = await sb_execute(
            lambda: supabase.table("notes")
                .select("id, image_url")
                .eq("id", note_id)
                .eq("user_id", user.id)
                .limit(1)
                .execute()
        )
        rows = get_res.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Note not found")

        image_url = rows[0].get("image_url")
        # 嘗試從 Storage 移除對應圖片（失敗不阻擋刪除）
        if image_url:
            marker = "/note_images/"
            idx = image_url.find(marker)
            if idx != -1:
                storage_path = image_url[idx + len(marker):].split("?", 1)[0]
                try:
                    await sb_execute(
                        lambda: supabase.storage.from_("note_images").remove([storage_path])
                    )
                except Exception as se:
                    print(f"⚠️ Storage 刪除失敗（繼續刪 row）：{se}")

        await sb_execute(
            lambda: supabase.table("notes")
                .delete()
                .eq("id", note_id)
                .eq("user_id", user.id)
                .execute()
        )
        return {"ok": True, "id": note_id}
    except HTTPException:
        raise
    except Exception as e:
        raise


# --- NewebPay 藍新金流 ---

@app.post("/api/create-newebpay-order")
async def create_newebpay_order(user=Depends(get_current_user)):
    _newebpay_require_keys()

    tpe_now = datetime.now(timezone(timedelta(hours=8)))
    # MerchantOrderNo 規範：英數，最長 30 字
    merchant_order_no = "CB" + tpe_now.strftime("%y%m%d%H%M%S") + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=4)
    )

    order = {
        "MerchantID": NEWEBPAY_MERCHANT_ID,
        "RespondType": "JSON",
        "TimeStamp": str(int(tpe_now.timestamp())),
        "Version": "2.0",
        "MerchantOrderNo": merchant_order_no,
        "Amt": 300,
        "ItemDesc": "專業版 AI 課程",
        "Email": getattr(user, "email", "") or "",
        # NotifyURL 是 server-to-server 回呼，會更新 is_premium；
        # ReturnURL 是使用者付款成功後的瀏覽器導回頁面，先用同一個 URL 簡化設定。
        "NotifyURL": NEWEBPAY_RETURN_URL,
        "Custom1": str(user.id),
    }
    trade_info = newebpay_encrypt(order)
    trade_sha = newebpay_trade_sha(trade_info)

    return {
        "action_url": NEWEBPAY_URL,
        "MerchantID": NEWEBPAY_MERCHANT_ID,
        "TradeInfo": trade_info,
        "TradeSha": trade_sha,
        "Version": "2.0",
    }


@app.post("/api/newebpay/webhook")
async def newebpay_webhook(request: Request):
    """接收藍新付款結果通知（server-to-server，不經 JWT 驗證）。"""
    raw = await request.body()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1")
    form = {k: v for k, v in urllib.parse.parse_qsl(decoded, keep_blank_values=True)}

    received_trade_info = form.get("TradeInfo", "")
    received_trade_sha = form.get("TradeSha", "")
    if not received_trade_info or not received_trade_sha:
        logger.warning("[NewebPay] webhook 缺 TradeInfo / TradeSha")
        return PlainTextResponse("ERR")

    # 1) 驗 TradeSha（防竄改）
    try:
        expected_sha = newebpay_trade_sha(received_trade_info)
    except HTTPException:
        # 環境變數沒設好就不可能驗證成功，直接回 ERR
        logger.warning("[NewebPay] HashKey/HashIV 未設定，無法驗 webhook")
        return PlainTextResponse("ERR")

    if expected_sha != received_trade_sha.upper():
        logger.warning(
            f"[NewebPay] TradeSha mismatch: received={received_trade_sha} expected={expected_sha}"
        )
        return PlainTextResponse("ERR")

    # 2) 解密 TradeInfo
    try:
        info = newebpay_decrypt(received_trade_info)
    except Exception as e:
        logger.warning(f"[NewebPay] 解密失敗：{e}")
        return PlainTextResponse("ERR")

    status = info.get("Status")
    result = info.get("Result") or {}
    merchant_order_no = result.get("MerchantOrderNo")

    if status != "SUCCESS":
        logger.info(
            f"[NewebPay] 付款未成功 status={status} order={merchant_order_no}"
        )
        return PlainTextResponse("OK")

    user_id = result.get("Custom1")
    if not user_id:
        logger.warning("[NewebPay] webhook 缺 Custom1 (user_id)")
        return PlainTextResponse("OK")

    try:
        await sb_execute(
            lambda: supabase.table("profiles").upsert({
                "id": user_id,
                "is_premium": True,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        )
        logger.info(
            f"✅ 用戶 {user_id} 升級為 Premium，MerchantOrderNo={merchant_order_no}"
        )
    except Exception as e:
        logger.warning(f"[NewebPay] 更新 profiles 失敗：{e}")

    return PlainTextResponse("OK")
