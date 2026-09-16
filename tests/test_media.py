import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from whisper_yt.media import acquire_video, burn_subtitles
from whisper_yt.models import Subtitle
from whisper_yt.subtitles import write_ass


def test_ffmpeg_can_burn_generated_ass(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    ass_file = tmp_path / "captions.ass"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-shortest",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    write_ass(
        ass_file,
        [Subtitle(1, 0, 0.2, "hello", "繁體中文字幕")],
        "sans-serif",
        52,
    )

    burn_subtitles(source, ass_file, output)

    assert output.stat().st_size > 0


def test_acquire_video_uses_path_reported_by_ytdlp(tmp_path: Path) -> None:
    downloaded = tmp_path / "yt-dlp-renamed-output.mp4"
    downloaded.write_bytes(b"video")
    downloader = MagicMock()
    downloader.__enter__.return_value = downloader
    downloader.extract_info.return_value = {
        "id": "abc123",
        "title": "Example",
        "filepath": str(downloaded),
    }
    downloader.prepare_filename.return_value = str(tmp_path / "source.webm")

    with patch("whisper_yt.media.YoutubeDL", return_value=downloader):
        path, title = acquire_video("https://youtu.be/abc123", tmp_path)

    assert path == downloaded
    assert title == "Example"
