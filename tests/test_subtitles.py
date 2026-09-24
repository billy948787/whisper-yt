import json
from threading import Barrier

import httpx
import pytest

from whisper_yt.models import Subtitle
from whisper_yt.subtitles import ass_timestamp, srt_timestamp
from whisper_yt.translate import (
    OpenCodeGoTranslator,
    SubtitleTranslator,
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


def test_translation_runs_batches_concurrently_and_checkpoints() -> None:
    barrier = Barrier(2, timeout=2)

    class ParallelTranslator(SubtitleTranslator):
        def _request(self, batch: list[Subtitle]) -> dict[int, str]:
            barrier.wait()
            return {item.id: f"譯文 {item.id}" for item in batch}

    subtitles = [Subtitle(i, 0, 1, f"text {i}") for i in range(1, 3)]
    snapshots: list[tuple[int, int, int]] = []
    ParallelTranslator().translate(
        subtitles,
        batch_chars=1,
        workers=2,
        on_batch_complete=lambda done, total: snapshots.append(
            (done, total, sum(bool(item.translated_text) for item in subtitles))
        ),
    )
    assert snapshots == [(1, 2, 1), (2, 2, 2)]
    assert [item.translated_text for item in subtitles] == ["譯文 1", "譯文 2"]


def test_translation_keeps_successful_batches_after_failure() -> None:
    class FailingTranslator(SubtitleTranslator):
        def _request(self, batch: list[Subtitle]) -> dict[int, str]:
            if batch[0].id == 1:
                raise RuntimeError("failed")
            return {item.id: "成功" for item in batch}

    subtitles = [Subtitle(i, 0, 1, "text") for i in range(1, 3)]
    checkpoints: list[int] = []
    with pytest.raises(RuntimeError, match="failed"):
        FailingTranslator().translate(
            subtitles,
            batch_chars=1,
            on_batch_complete=lambda done, total: checkpoints.append(done),
        )
    assert checkpoints == [1]
    assert [item.translated_text for item in subtitles] == ["", "成功"]


def test_rate_limit_does_not_split_into_more_requests() -> None:
    calls = 0

    class RateLimitedTranslator(SubtitleTranslator):
        def _request(self, batch: list[Subtitle]) -> dict[int, str]:
            nonlocal calls
            calls += 1
            response = httpx.Response(429, request=httpx.Request("POST", "https://example.com"))
            raise httpx.HTTPStatusError("rate limited", request=response.request, response=response)

    subtitles = [Subtitle(i, 0, 1, "text") for i in range(1, 3)]
    with pytest.raises(httpx.HTTPStatusError):
        RateLimitedTranslator().translate(subtitles, workers=1)
    assert calls == 1
