from pathlib import Path

import pytest

from whisper_yt.cli import _select_provider


@pytest.mark.parametrize(("answer", "expected"), [(1, "opencode"), (2, "codex")])
def test_select_provider_maps_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: int, expected: str
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("whisper_yt.cli.typer.prompt", lambda *args, **kwargs: answer)
    assert _select_provider() == expected
