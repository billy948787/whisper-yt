import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from .media import acquire_video, burn_subtitles, extract_audio, require_ffmpeg
from .models import Subtitle
from .subtitles import write_ass, write_srt
from .transcribe import detect_device, transcribe
from .translate import DEFAULT_API_URL, DEFAULT_MODEL, OpenCodeGoTranslator

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    source: Annotated[str, typer.Argument(help="YouTube 網址或本機影片路徑")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("output"),
    model: Annotated[str, typer.Option(help="Whisper 模型名稱")] = "large-v3",
    device: Annotated[str, typer.Option(help="auto、cpu、cuda、mps 或裝置名稱")] = "auto",
    language: Annotated[str | None, typer.Option(help="來源語言代碼；留空自動偵測")] = None,
    translation_model: Annotated[str, typer.Option(help="OpenCode Go 模型 ID")] = DEFAULT_MODEL,
    api_url: Annotated[str, typer.Option(help="OpenCode Go chat completions 端點")] = DEFAULT_API_URL,
    font: Annotated[str, typer.Option(help="燒錄字幕字型")] = "Noto Sans CJK TC",
    font_size: Annotated[int, typer.Option(help="ASS 字幕大小（以 1080p 為基準）")] = 52,
    no_burn: Annotated[
        bool, typer.Option("--no-burn", help="只輸出字幕，不燒錄影片", is_flag=True)
    ] = False,
) -> None:
    """轉錄影片、翻譯為台灣繁體中文，並將字幕燒錄至 MP4。"""
    require_ffmpeg()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device, device_description = detect_device(device)
    typer.echo(f"Whisper 裝置：{device_description}")

    translator = OpenCodeGoTranslator(model=translation_model, api_url=api_url)
    cache_identity = json.dumps(
        {"source": source, "model": model, "language": language}, sort_keys=True
    )
    source_key = hashlib.sha256(cache_identity.encode()).hexdigest()[:12]
    cache_file = output_dir / f".{source_key}.json"
    with tempfile.TemporaryDirectory(prefix="whisper-yt-") as temp:
        work_dir = Path(temp)
        video, title = acquire_video(source, work_dir)
        basename = _safe_name(title)
        subtitles, metadata = _load_cache(cache_file)

        if metadata.get("translation_model") != translation_model:
            for subtitle in subtitles:
                subtitle.translated_text = ""

        if not subtitles:
            audio = work_dir / "audio.wav"
            typer.echo("擷取音訊並執行 Whisper 轉錄...")
            extract_audio(video, audio)
            subtitles, metadata = transcribe(
                audio, model, selected_device, language, output_dir / ".models"
            )
            _save_cache(cache_file, subtitles, metadata)
        else:
            typer.echo("使用既有轉錄快取。")

        untranslated = [item for item in subtitles if not item.translated_text]
        if untranslated:
            typer.echo(f"透過 OpenCode Go 翻譯 {len(untranslated)} 段字幕...")
            translator.translate(untranslated)
            metadata["translation_model"] = translation_model
            _save_cache(cache_file, subtitles, metadata)
        else:
            typer.echo("使用既有翻譯快取。")

        original_srt = output_dir / f"{basename}.original.srt"
        translated_srt = output_dir / f"{basename}.zh-TW.srt"
        ass_file = work_dir / "captions.ass"
        saved_ass = output_dir / f"{basename}.zh-TW.ass"
        write_srt(original_srt, subtitles, translated=False)
        write_srt(translated_srt, subtitles, translated=True)
        write_ass(ass_file, subtitles, font, font_size)
        saved_ass.write_bytes(ass_file.read_bytes())

        typer.echo(f"字幕：{translated_srt}")
        if not no_burn:
            output_video = output_dir / f"{basename}.zh-TW.mp4"
            typer.echo("燒錄字幕並輸出影片...")
            burn_subtitles(video, ass_file, output_video)
            typer.echo(f"影片：{output_video}")


def _safe_name(value: str) -> str:
    name = re.sub(r"[^\w. -]+", "_", value, flags=re.UNICODE).strip(" .")
    return name[:120] or "video"


def _load_cache(path: Path) -> tuple[list[Subtitle], dict[str, object]]:
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Subtitle.from_dict(item) for item in data["subtitles"]], data.get("metadata", {})


def _save_cache(path: Path, subtitles: list[Subtitle], metadata: dict[str, object]) -> None:
    data = {"metadata": metadata, "subtitles": [item.to_dict() for item in subtitles]}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    app()
