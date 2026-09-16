import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝並確認它位於 PATH。")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def acquire_video(source: str, work_dir: Path) -> tuple[Path, str]:
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到影片檔案：{path}")
        return path, path.stem

    source = _normalize_url(source)
    template = str(work_dir / "source.%(ext)s")
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": template,
        "noplaylist": True,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(source, download=True)
        title = str(info.get("title") or "youtube-video")
        reported_paths = _reported_download_paths(info, downloader)

    candidates = [path for path in reported_paths if path.is_file()]
    candidates.extend(
        path
        for path in work_dir.iterdir()
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    )
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    if not candidates:
        raise RuntimeError(
            f"yt-dlp 執行完成，但找不到下載的影片（video id: {info.get('id', 'unknown')}）。"
        )
    return max(candidates, key=lambda path: path.stat().st_size), title


def _normalize_url(value: str) -> str:
    for character in "?=&#%":
        value = value.replace(f"\\{character}", character)
    return value


def _reported_download_paths(info: dict, downloader: YoutubeDL) -> list[Path]:
    values = [info.get("filepath"), info.get("filename")]
    values.extend(
        item.get(key)
        for item in info.get("requested_downloads") or []
        for key in ("filepath", "filename")
    )
    try:
        values.append(downloader.prepare_filename(info))
    except KeyError:
        pass
    return [Path(value).expanduser().resolve() for value in values if value]


def extract_audio(video: Path, audio: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio),
    ]
    subprocess.run(command, check=True)


def burn_subtitles(video: Path, ass_file: Path, output: Path) -> None:
    # A fixed local ASS filename avoids ffmpeg filter escaping issues in arbitrary paths.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"ass={ass_file.name}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, cwd=ass_file.parent, check=True)
