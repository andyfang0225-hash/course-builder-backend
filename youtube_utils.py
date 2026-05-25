"""YouTube 網址解析與字幕萃取工具（async 介面，內部以 to_thread 包裝 sync 呼叫）。

本模組對應 youtube-transcript-api 1.x：
  - API 已改為實例化：`YouTubeTranscriptApi().fetch(video_id, languages=[...])`
  - 回傳 `FetchedTranscript`（iterable 產出 `FetchedTranscriptSnippet(text, start, duration)`）
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi

# 涵蓋常見網址格式：
#   https://www.youtube.com/watch?v=XXXXXXXXXXX
#   https://youtu.be/XXXXXXXXXXX
#   https://www.youtube.com/embed/XXXXXXXXXXX
#   https://www.youtube.com/shorts/XXXXXXXXXXX
#   https://www.youtube.com/v/XXXXXXXXXXX
#   https://m.youtube.com/watch?v=XXXXXXXXXXX
_VIDEO_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|v/))"
    r"([A-Za-z0-9_-]{11})"
)

# 語言優先序：繁中 → 簡中 → 英文
_PREFERRED_LANGUAGES = ["zh-TW", "zh-Hant", "zh-CN", "zh-Hans", "zh", "en"]


def extract_video_id(url: str) -> Optional[str]:
    """從常見 YouTube 網址萃取 11 碼 Video ID；抓不到回 None。"""
    if not url:
        return None
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


async def get_youtube_transcript(video_id: str) -> str:
    """依語言優先序抓字幕並合併成單一字串；失敗（無字幕、私人影片、被移除等）回空字串。"""

    def _fetch() -> str:
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=_PREFERRED_LANGUAGES)
        except Exception:
            return ""
        return " ".join(
            snippet.text.strip()
            for snippet in transcript
            if snippet.text
        ).strip()

    return await asyncio.to_thread(_fetch)
