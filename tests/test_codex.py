import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from whisper_yt.codex import (
    CODEX_API_URL,
    DEFAULT_MODEL,
    TOKEN_URL,
    CodexCredentials,
    CodexTranslator,
    _collect_response_text,
    default_model,
    load_credentials,
    refresh_credentials,
)
from whisper_yt.models import Subtitle


def _token(claims: dict[str, object]) -> str:
    def segment(payload: dict[str, object]) -> str:
        raw = json.dumps(payload).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment({'alg': 'none'})}.{segment(claims)}.signature"


def _future_token(account_id: str | None = None) -> str:
    claims: dict[str, object] = {"exp": time.time() + 3600}
    if account_id:
        claims["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id}
    return _token(claims)


def _sse(*events: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)


def _completed(text: str) -> dict[str, object]:
    return {
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": text}]}
            ]
        },
    }


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        data: dict[str, object] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._data = data

    def json(self) -> dict[str, object]:
        assert self._data is not None
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", CODEX_API_URL),
                response=httpx.Response(self.status_code),
            )


def _credentials() -> CodexCredentials:
    return CodexCredentials(
        access_token=_future_token(),
        refresh_token="refresh",
        account_id="account",
    )


def _write_auth(path: Path, access_token: str, refresh_token: str = "refresh") -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "account_id": "account",
                },
                "last_refresh": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_collect_response_text_prefers_completed_event() -> None:
    text = _sse(
        {"type": "response.output_text.delta", "delta": "你"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "你好"}],
            },
        },
        _completed("你好"),
    )
    assert _collect_response_text(text) == "你好"


def test_collect_response_text_falls_back_to_deltas() -> None:
    text = _sse(
        {"type": "response.output_text.delta", "delta": "你"},
        {"type": "response.output_text.delta", "delta": "好"},
    )
    assert _collect_response_text(text) == "你好"


def test_collect_response_text_raises_on_failed_event() -> None:
    text = _sse({"type": "response.failed", "response": {"error": {"message": "boom"}}})
    with pytest.raises(ValueError, match="boom"):
        _collect_response_text(text)


def test_codex_translator_sends_codex_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    translations = {"translations": [{"id": 1, "text": "你好"}]}
    body = _sse(_completed(json.dumps(translations, ensure_ascii=False)))

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return FakeResponse(text=body)

    monkeypatch.setattr("whisper_yt.codex.httpx.post", fake_post)
    subtitles = [Subtitle(1, 0, 1, "hello")]

    CodexTranslator(model="gpt-test", credentials=_credentials()).translate(subtitles)

    assert subtitles[0].translated_text == "你好"
    assert captured["url"] == CODEX_API_URL
    headers = captured["headers"]
    assert headers["chatgpt-account-id"] == "account"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["Accept"] == "text/event-stream"
    payload = captured["payload"]
    assert payload["model"] == "gpt-test"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["instructions"]
    assert payload["input"][0]["content"][0]["text"] == json.dumps(
        [{"id": 1, "text": "hello"}], ensure_ascii=False
    )


def test_codex_translator_refreshes_expired_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "auth.json"
    _write_auth(path, _token({"exp": time.time() - 10}))
    refreshed = _future_token()
    calls: list[str] = []
    translations = {"translations": [{"id": 1, "text": "嗨"}]}
    body = _sse(_completed(json.dumps(translations, ensure_ascii=False)))

    def fake_post(
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        data: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        calls.append(url)
        if url == TOKEN_URL:
            return FakeResponse(data={"access_token": refreshed, "refresh_token": "new"})
        return FakeResponse(text=body)

    monkeypatch.setattr("whisper_yt.codex.httpx.post", fake_post)
    subtitles = [Subtitle(1, 0, 1, "hi")]

    CodexTranslator(model="gpt-test", credentials=load_credentials(path)).translate(subtitles)

    assert calls[0] == TOKEN_URL
    assert subtitles[0].translated_text == "嗨"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == refreshed


def test_load_credentials_reads_auth_file(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth(path, _future_token())

    credentials = load_credentials(path)

    assert credentials.account_id == "account"
    assert credentials.refresh_token == "refresh"
    assert credentials.path == path


def test_load_credentials_reads_account_id_from_token(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    access_token = _future_token(account_id="from-jwt")
    path.write_text(
        json.dumps({"tokens": {"access_token": access_token, "refresh_token": "refresh"}}),
        encoding="utf-8",
    )
    assert load_credentials(path).account_id == "from-jwt"


def test_refresh_credentials_updates_auth_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "auth.json"
    _write_auth(path, _token({"exp": time.time() - 10}))
    refreshed = _future_token()

    def fake_post(url: str, **kwargs: object):
        assert url == TOKEN_URL
        return FakeResponse(data={"access_token": refreshed, "refresh_token": "new"})

    monkeypatch.setattr("whisper_yt.codex.httpx.post", fake_post)
    updated = refresh_credentials(load_credentials(path))

    assert updated.access_token == refreshed
    assert updated.refresh_token == "new"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == refreshed
    assert saved["tokens"]["refresh_token"] == "new"


def test_default_model_reads_codex_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.toml").write_text('model = "gpt-custom"\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert default_model() == "gpt-custom"


def test_default_model_falls_back_when_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert default_model() == DEFAULT_MODEL
