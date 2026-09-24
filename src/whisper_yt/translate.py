import json
import os
import time
import uuid
from collections.abc import Iterable
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from .models import Subtitle

DEFAULT_API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_MODEL = "glm-5.3-flash"
MODELS_URL = "https://models.dev/api.json"
NON_CHAT_PREFIXES = (
    "grok-",
    "gpt-5.6-luna",
    "muse-spark-",
    "ox-alpha",
    "union-alpha",
)
USER_AGENT = "whisper-yt/0.1.0"

INSTRUCTIONS = (
    "你是專業字幕譯者。將輸入字幕翻譯成自然、精簡的繁體中文（台灣用語）。"
    "保留人名、品牌、程式碼、數字與語氣；不要增添解說。"
    "只輸出有效 JSON，格式必須是 {\"translations\":[{\"id\":1,\"text\":\"譯文\"}]}，"
    "每個輸入 id 必須恰好出現一次。"
)


def list_models(api_url: str = DEFAULT_API_URL) -> list[str]:
    base = api_url.removesuffix("/chat/completions").rstrip("/")
    try:
        response = httpx.get(
            f"{base}/models", timeout=20, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    models: list[str] = []
    for item in data.get("data") or []:
        slug = str(item.get("id") or "") if isinstance(item, dict) else ""
        if slug and not slug.startswith(NON_CHAT_PREFIXES):
            models.append(slug)
    return models


def model_variants() -> dict[str, list[str]]:
    try:
        response = httpx.get(MODELS_URL, timeout=30, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return {}
    provider = data.get("opencode-go") if isinstance(data, dict) else None
    if not isinstance(provider, dict):
        return {}
    variants: dict[str, list[str]] = {}
    for slug, model in (provider.get("models") or {}).items():
        if not isinstance(model, dict):
            continue
        efforts: list[str] = []
        for option in model.get("reasoning_options") or []:
            if isinstance(option, dict) and option.get("type") == "effort":
                efforts.extend(str(value) for value in option.get("values") or [])
        if efforts:
            variants[str(slug)] = efforts
    return variants


class SubtitleTranslator:
    def translate(
        self,
        subtitles: list[Subtitle],
        batch_chars: int = 6000,
        workers: int = 4,
        on_batch_complete: Callable[[int, int], None] | None = None,
    ) -> None:
        batches = _make_batches(subtitles, batch_chars)
        if not batches:
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._translate_batch, batch): batch for batch in batches}
            completed = 0
            first_error: Exception | None = None
            for future in as_completed(futures):
                try:
                    translations = future.result()
                except Exception as error:
                    if first_error is None:
                        first_error = error
                    continue
                for subtitle in futures[future]:
                    subtitle.translated_text = translations[subtitle.id]
                completed += 1
                if on_batch_complete is not None:
                    on_batch_complete(completed, len(batches))
            if first_error is not None:
                raise first_error

    def _translate_batch(self, batch: list[Subtitle]) -> dict[int, str]:
        try:
            return self._request(batch)
        except (KeyError, ValueError, httpx.HTTPError) as error:
            # Splitting malformed responses may help, but rate limits and server
            # failures should not multiply the number of requests.
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code != 413:
                raise
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            return self._translate_batch(batch[:middle]) | self._translate_batch(batch[middle:])

    def _request(self, batch: list[Subtitle]) -> dict[int, str]:
        raise NotImplementedError


class OpenCodeGoTranslator(SubtitleTranslator):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        effort: str | None = None,
        timeout: float = 120,
        session_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENCODE_GO_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("未設定 OPENCODE_GO_API_KEY，無法呼叫 OpenCode Go 翻譯。")
        self.model = model
        self.api_url = api_url
        self.effort = effort
        self.timeout = timeout
        self.session_id = session_id or str(uuid.uuid4())

    def _request(self, batch: list[Subtitle]) -> dict[int, str]:
        items = [{"id": item.id, "text": item.text} for item in batch]
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            "temperature": 0.1,
        }
        if self.effort:
            payload["reasoning_effort"] = self.effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
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
