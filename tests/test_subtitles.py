import json

import pytest

from whisper_yt.models import Subtitle
from whisper_yt.subtitles import ass_timestamp, srt_timestamp
from whisper_yt.translate import _make_batches, _parse_translations


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
