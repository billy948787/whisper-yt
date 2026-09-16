import json

import httpx
import pytest

from whisper_yt.models import Subtitle
from whisper_yt.subtitles import ass_timestamp, srt_timestamp
from whisper_yt.translate import (
    OpenCodeGoTranslator,
    _make_batches,
    _parse_translations,
    list_models,
    model_variants,
)


class FakeResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._data


def test_timestamps_round_correctly() -> None:
    assert srt_timestamp(3661.2346) == "01:01:01,235"
    assert ass_timestamp(3661.236) == "1:01:01.24"


def test_translation_parser_accepts_fenced_json() -> None:
    content = "```json\n" + json.dumps(
        {"translations": [{"id": 1, "text": "你好"}, {"id": 2, "text": "世界"}]},
        ensure_ascii=False,
    ) + "\n```"
    assert _parse_translations(content, {1, 2}) == {1: "你好", 2: "世界"}


def test_batches_keep_every_subtitle() -> None:
    subtitles = [Subtitle(id=index, start=0, end=1, text="abcd") for index in range(1, 5)]
    batches = _make_batches(subtitles, max_chars=8)
    assert [[item.id for item in batch] for batch in batches] == [[1, 2], [3, 4]]


def test_translation_parser_rejects_duplicate_ids() -> None:
    content = json.dumps(
        {"translations": [{"id": 1, "text": "甲"}, {"id": 1, "text": "乙"}]},
        ensure_ascii=False,
    )
    with pytest.raises(ValueError):
        _parse_translations(content, {1, 2})


def test_list_models_filters_models_without_chat_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(
        {"data": [{"id": "glm-5.3-flash"}, {"id": "grok-4.6"}, {"id": "kimi-k3"}]}
    )
    monkeypatch.setattr("whisper_yt.translate.httpx.get", lambda *args, **kwargs: response)
    assert list_models() == ["glm-5.3-flash", "kimi-k3"]


def test_list_models_returns_empty_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("whisper_yt.translate.httpx.get", fail)
    assert list_models() == []


def test_model_variants_reads_effort_options(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        {
            "opencode-go": {
                "models": {
                    "glm-5.3-flash": {
                        "reasoning_options": [
                            {"type": "effort", "values": ["low", "high", "max"]}
                        ]
                    },
                    "kimi-k3": {},
                }
            }
        }
    )
    monkeypatch.setattr("whisper_yt.translate.httpx.get", lambda *args, **kwargs: response)
    assert model_variants() == {"glm-5.3-flash": ["low", "high", "max"]}


def test_opencode_translator_sends_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        payloads.append(payload)
        items = json.loads(payload["messages"][1]["content"])
        translations = {"translations": [{"id": item["id"], "text": "你好"} for item in items]}
        return FakeResponse(
            {"choices": [{"message": {"content": json.dumps(translations, ensure_ascii=False)}}]}
        )

    monkeypatch.setattr("whisper_yt.translate.httpx.post", fake_post)
    OpenCodeGoTranslator(api_key="key").translate([Subtitle(1, 0, 1, "hello")])
    OpenCodeGoTranslator(api_key="key", effort="max").translate([Subtitle(2, 0, 1, "hi")])

    assert "reasoning_effort" not in payloads[0]
    assert payloads[1]["reasoning_effort"] == "max"
