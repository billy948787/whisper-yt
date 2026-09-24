import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from whisper_yt.cli import _select_provider, app


@pytest.mark.parametrize(("answer", "expected"), [(1, "opencode"), (2, "codex")])
def test_select_provider_maps_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: int, expected: str
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("whisper_yt.cli.typer.prompt", lambda *args, **kwargs: answer)
    assert _select_provider() == expected


@pytest.mark.parametrize(
    ("existing_key", "expected_key"),
    [(None, "from-dotenv"), ("from-shell", "from-shell")],
)
def test_cli_loads_dotenv_without_overriding_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_key: str | None, expected_key: str
) -> None:
    (tmp_path / ".env").write_text("OPENCODE_GO_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    if existing_key is None:
        monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENCODE_GO_API_KEY", existing_key)
    monkeypatch.setattr("whisper_yt.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("whisper_yt.cli.resolve_engine", lambda engine, device: ("pytorch", "CPU"))

    def stop_before_processing(source: str, work_dir: Path) -> None:
        assert os.environ["OPENCODE_GO_API_KEY"] == expected_key
        raise RuntimeError("configuration loaded")

    monkeypatch.setattr("whisper_yt.cli.acquire_video", stop_before_processing)
    result = CliRunner().invoke(app, ["video.mp4", "--provider", "opencode"])
    if existing_key is None:
        monkeypatch.setenv("OPENCODE_GO_API_KEY", expected_key)
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "configuration loaded"
