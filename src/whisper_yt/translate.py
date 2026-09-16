import json
import os
import time
import uuid
from collections.abc import Iterable

import httpx

from .models import Subtitle

DEFAULT_API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_MODEL = "glm-5.3-flash"


class OpenCodeGoTranslator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        timeout: float = 120,
        session_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENCODE_GO_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("未設定 OPENCODE_GO_API_KEY，無法呼叫 OpenCode Go 翻譯。")
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.session_id = session_id or str(uuid.uuid4())

    def translate(self, subtitles: list[Subtitle], batch_chars: int = 6000) -> None:
        for batch in _make_batches(subtitles, batch_chars):
            self._translate_batch(batch)

    def _translate_batch(self, batch: list[Subtitle]) -> None:
        try:
            translations = self._request(batch)
            for subtitle in batch:
                subtitle.translated_text = translations[subtitle.id]
        except (KeyError, ValueError, httpx.HTTPError):
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            self._translate_batch(batch[:middle])
            self._translate_batch(batch[middle:])

    def _request(self, batch: list[Subtitle]) -> dict[int, str]:
        items = [{"id": item.id, "text": item.text} for item in batch]
        system = (
            "你是專業字幕譯者。將輸入字幕翻譯成自然、精簡的繁體中文（台灣用語）。"
            "保留人名、品牌、程式碼、數字與語氣；不要增添解說。"
            "只輸出有效 JSON，格式必須是 {\"translations\":[{\"id\":1,\"text\":\"譯文\"}]}，"
            "每個輸入 id 必須恰好出現一次。"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "whisper-yt/0.1.0",
            "x-opencode-session": self.session_id,
        }
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = httpx.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return _parse_translations(content, {item.id for item in batch})
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt < 3:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error


def _make_batches(subtitles: Iterable[Subtitle], max_chars: int) -> list[list[Subtitle]]:
    batches: list[list[Subtitle]] = []
    current: list[Subtitle] = []
    current_chars = 0
    for subtitle in subtitles:
        size = len(subtitle.text)
        if current and (current_chars + size > max_chars or len(current) >= 40):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(subtitle)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _parse_translations(content: str, expected_ids: set[int]) -> dict[int, str]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("翻譯 API 未回傳 JSON。")
    data = json.loads(text[start : end + 1])
    items = data["translations"]
    translations = {int(item["id"]): str(item["text"]).strip() for item in items}
    if (
        len(items) != len(expected_ids)
        or set(translations) != expected_ids
        or any(not value for value in translations.values())
    ):
        raise ValueError("翻譯 API 回傳的字幕 id 不完整或譯文為空。")
    return translations
