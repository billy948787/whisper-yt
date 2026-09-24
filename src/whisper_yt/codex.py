import base64
import json
import os
import time
import tomllib
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx

from .models import Subtitle
from .translate import INSTRUCTIONS, SubtitleTranslator, _parse_translations

CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_MODEL = "gpt-5.6-sol"
USER_AGENT = "whisper-yt/0.1.0"


@dataclass(slots=True)
class CodexCredentials:
    access_token: str
    refresh_token: str
    account_id: str
    path: Path | None = None


@dataclass(slots=True)
class CodexModel:
    slug: str
    display_name: str
    default_effort: str
    efforts: list[str]


def codex_home() -> Path:
    return Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser()


def load_credentials(path: Path | None = None) -> CodexCredentials:
    auth_path = (path or codex_home() / "auth.json").expanduser()
    if not auth_path.is_file():
        raise RuntimeError(f"找不到 Codex 登入資訊：{auth_path}。請先執行 `codex login`。")
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    tokens = data.get("tokens") or {}
    access_token = str(tokens.get("access_token") or "")
    refresh_token = str(tokens.get("refresh_token") or "")
    if not access_token or not refresh_token:
        raise RuntimeError("Codex auth.json 缺少 OAuth token，請重新執行 `codex login`。")
    account_id = str(tokens.get("account_id") or _account_id_from_token(access_token))
    if not account_id:
        raise RuntimeError("Codex auth.json 缺少 account_id，請重新執行 `codex login`。")
    return CodexCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        account_id=account_id,
        path=auth_path,
    )


def default_model() -> str:
    config = codex_home() / "config.toml"
    if config.is_file():
        try:
            model = tomllib.loads(config.read_text(encoding="utf-8")).get("model")
        except tomllib.TOMLDecodeError:
            model = None
        if isinstance(model, str) and model.strip():
            return model.strip()
    return DEFAULT_MODEL


def available_models() -> list[CodexModel]:
    cache = codex_home() / "models_cache.json"
    if not cache.is_file():
        return []
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    models: list[CodexModel] = []
    for item in data.get("models") or []:
        if not isinstance(item, dict) or not item.get("supported_in_api"):
            continue
        if item.get("visibility", "list") != "list":
            continue
        slug = str(item.get("slug") or "")
        if not slug:
            continue
        efforts = [
            str(level["effort"])
            for level in item.get("supported_reasoning_levels") or []
            if isinstance(level, dict) and level.get("effort")
        ]
        default = str(item.get("default_reasoning_level") or "")
        models.append(
            CodexModel(
                slug=slug,
                display_name=str(item.get("display_name") or slug),
                default_effort=default if default in efforts else (efforts[0] if efforts else "low"),
                efforts=efforts,
            )
        )
    return models


def default_effort(model: str) -> str:
    for item in available_models():
        if item.slug == model:
            return item.default_effort
    return "low"


def refresh_credentials(credentials: CodexCredentials) -> CodexCredentials:
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    access_token = str(data.get("access_token") or "")
    if not access_token:
        raise RuntimeError("Codex token 更新失敗，請重新執行 `codex login`。")
    updated = replace(
        credentials,
        access_token=access_token,
        refresh_token=str(data.get("refresh_token") or credentials.refresh_token),
    )
    _save_credentials(updated)
    return updated


class CodexTranslator(SubtitleTranslator):
    def __init__(
        self,
        model: str | None = None,
        credentials: CodexCredentials | None = None,
        auth_file: Path | None = None,
        effort: str | None = None,
        timeout: float = 300,
        session_id: str | None = None,
    ) -> None:
        self.model = model or default_model()
        self.effort = effort or default_effort(self.model)
        self.credentials = credentials or load_credentials(auth_file)
        self._credential_lock = Lock()
        self.timeout = timeout
        self.session_id = session_id or str(uuid.uuid4())

    def _request(self, batch: list[Subtitle]) -> dict[int, str]:
        items = [{"id": item.id, "text": item.text} for item in batch]
        payload = {
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(items, ensure_ascii=False),
                        }
                    ],
                }
            ],
            "store": False,
            "stream": True,
            "reasoning": {"effort": self.effort},
            "include": ["reasoning.encrypted_content"],
        }
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                content = self._post(payload)
                return _parse_translations(content, {item.id for item in batch})
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt < 3:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def _post(self, payload: dict[str, object]) -> str:
        response = self._send(payload)
        token = self.credentials.access_token
        if response.status_code == 401:
            with self._credential_lock:
                if self.credentials.access_token == token:
                    self.credentials = refresh_credentials(self.credentials)
            response = self._send(payload)
        response.raise_for_status()
        return _collect_response_text(response.text)

    def _send(self, payload: dict[str, object]) -> httpx.Response:
        with self._credential_lock:
            if _token_expired(self.credentials.access_token):
                self.credentials = refresh_credentials(self.credentials)
            credentials = self.credentials
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "chatgpt-account-id": credentials.account_id,
            "originator": "codex_cli_rs",
            "OpenAI-Beta": "responses=experimental",
            "session_id": self.session_id,
            "Accept": "text/event-stream",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
        return httpx.post(CODEX_API_URL, headers=headers, json=payload, timeout=self.timeout)


def _save_credentials(credentials: CodexCredentials) -> None:
    if credentials.path is None or not credentials.path.is_file():
        return
    data = json.loads(credentials.path.read_text(encoding="utf-8"))
    tokens = data.setdefault("tokens", {})
    tokens["access_token"] = credentials.access_token
    tokens["refresh_token"] = credentials.refresh_token
    data["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    temporary = credentials.path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(credentials.path)


def _jwt_claims(token: str) -> dict[str, object]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _account_id_from_token(token: str) -> str:
    auth = _jwt_claims(token).get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        return str(auth.get("chatgpt_account_id") or "")
    return ""


def _token_expired(token: str, leeway: float = 120) -> bool:
    expires = _jwt_claims(token).get("exp")
    if not isinstance(expires, (int, float)):
        return True
    return expires <= time.time() + leeway


def _collect_response_text(sse_text: str) -> str:
    completed: dict[str, object] | None = None
    item_texts: list[str] = []
    deltas: list[str] = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:") :].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            event = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in {"response.completed", "response.done"}:
            response = event.get("response")
            completed = response if isinstance(response, dict) else None
        elif kind == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "message":
                text = _message_text(item)
                if text:
                    item_texts.append(text)
        elif kind == "response.output_text.delta":
            deltas.append(str(event.get("delta") or ""))
        elif kind in {"response.failed", "error"}:
            raise ValueError(f"Codex 翻譯失敗：{_error_message(event)}")
    if completed is not None:
        text = _response_text(completed)
        if text:
            return text
    if item_texts:
        return "".join(item_texts)
    return "".join(deltas)


def _response_text(response: dict[str, object]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    return "".join(
        _message_text(item)
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    )


def _message_text(item: dict[str, object]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"output_text", "text"}
    )


def _error_message(event: dict[str, object]) -> str:
    error = event.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    response = event.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(event.get("message") or "未知錯誤")
