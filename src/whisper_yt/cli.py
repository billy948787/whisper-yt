import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from .codex import (
    CodexTranslator,
    available_models,
    default_effort,
    default_model as codex_default_model,
)
from .media import acquire_video, burn_subtitles, extract_audio, require_ffmpeg
from .models import Subtitle
from .progress import PhaseProgress
from .subtitles import write_ass, write_srt
from .transcribe import ENGINES, repeated_ratio, resolve_engine, transcribe
from .translate import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    OpenCodeGoTranslator,
    list_models,
    model_variants,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    source: Annotated[str, typer.Argument(help="YouTube 網址或本機影片路徑")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("output"),
    model: Annotated[str, typer.Option(help="Whisper 模型名稱")] = "large-v3",
    device: Annotated[
        str, typer.Option(help="PyTorch 引擎的裝置：auto、cpu、cuda、mps 或裝置名稱")
    ] = "auto",
    engine: Annotated[
        str, typer.Option(help="Whisper 引擎：auto、whisper-cpp（較快）或 pytorch")
    ] = "auto",
    language: Annotated[str | None, typer.Option(help="來源語言代碼；留空自動偵測")] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="翻譯服務：opencode 或 codex；留空會在互動模式問你")
    ] = None,
    translation_model: Annotated[
        str | None, typer.Option(help="翻譯模型 ID；codex 預設沿用 ~/.codex/config.toml 的 model")
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(help="variant／推理強度；codex 如 low…ultra，opencode 依模型（如 low/high/max）"),
    ] = None,
    api_url: Annotated[
        str, typer.Option(help="OpenCode Go chat completions 端點（僅 opencode）")
    ] = DEFAULT_API_URL,
    font: Annotated[str, typer.Option(help="燒錄字幕字型")] = "Noto Sans CJK TC",
    font_size: Annotated[int, typer.Option(help="ASS 字幕大小（以 1080p 為基準）")] = 52,
    no_burn: Annotated[
        bool, typer.Option("--no-burn", help="只輸出字幕，不燒錄影片", is_flag=True)
    ] = False,
) -> None:
    """轉錄影片、翻譯為台灣繁體中文，並將字幕燒錄至 MP4。"""
    load_dotenv(Path.cwd() / ".env")
    require_ffmpeg()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if engine not in ENGINES:
        raise typer.BadParameter(
            "engine 只支援 auto、whisper-cpp 或 pytorch。", param_hint="--engine"
        )
    if engine == "whisper-cpp" and device != "auto":
        raise typer.BadParameter(
            "--device 只適用於 PyTorch 引擎；whisper.cpp 會自動選擇後端與裝置。",
            param_hint="--device",
        )
    try:
        engine_name, device_description = resolve_engine(engine, device)
    except (RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Whisper 引擎：{device_description}")

    if provider is None:
        if translation_model is None and variant is None and sys.stdin.isatty():
            provider = _select_provider()
        else:
            provider = "opencode"
    if provider == "opencode":
        if translation_model is None and variant is None and sys.stdin.isatty():
            translation_model, variant = _select_opencode_model(DEFAULT_MODEL, api_url)
        translation_model = translation_model or DEFAULT_MODEL
        translator = OpenCodeGoTranslator(
            model=translation_model, api_url=api_url, effort=variant
        )
        service = "OpenCode Go"
    elif provider == "codex":
        if translation_model is None and variant is None and sys.stdin.isatty():
            translation_model, variant = _select_codex_model(codex_default_model())
        translation_model = translation_model or codex_default_model()
        translator = CodexTranslator(model=translation_model, effort=variant)
        service = "ChatGPT Codex"
    else:
        raise typer.BadParameter("provider 只支援 opencode 或 codex。", param_hint="--provider")

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

        if (
            metadata.get("translation_model") != translation_model
            or metadata.get("translation_provider", "opencode") != provider
        ):
            for subtitle in subtitles:
                subtitle.translated_text = ""

        if not subtitles:
            audio = work_dir / "audio.wav"
            typer.echo("擷取音訊並執行 Whisper 轉錄...")
            extract_audio(video, audio)
            progress = PhaseProgress({"download": "下載 Whisper 模型", "transcribe": "Whisper 轉錄"})
            try:
                subtitles, metadata = transcribe(
                    audio,
                    model,
                    engine_name,
                    device,
                    language,
                    output_dir / ".models",
                    log=typer.echo,
                    on_progress=progress,
                )
            finally:
                progress.finish()
            if len(subtitles) >= 20 and repeated_ratio(subtitles) >= 0.3:
                typer.echo(
                    "警告：轉錄結果有大量重複片段，可能是長靜音或音訊異常，"
                    "建議確認來源音訊後再刪除快取重跑。"
                )
            _save_cache(cache_file, subtitles, metadata)
        else:
            typer.echo("使用既有轉錄快取。")

        untranslated = [item for item in subtitles if not item.translated_text]
        if untranslated:
            typer.echo(f"透過 {service} 翻譯 {len(untranslated)} 段字幕...")
            metadata["translation_provider"] = provider
            metadata["translation_model"] = translation_model

            def save_progress(completed: int, total: int) -> None:
                _save_cache(cache_file, subtitles, metadata)
                typer.echo(f"翻譯進度：{completed}/{total} 批")

            translator.translate(untranslated, on_batch_complete=save_progress)
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


def _select_provider() -> str:
    typer.echo("選擇翻譯服務：")
    typer.echo(f"  1. OpenCode Go（預設 {DEFAULT_MODEL}，需要 API key）")
    typer.echo(f"  2. ChatGPT Codex（預設 {codex_default_model()}，用 codex login 的額度）")
    return "opencode" if _prompt_index("編號", 2) == 1 else "codex"


def _select_opencode_model(default_slug: str, api_url: str) -> tuple[str, str | None]:
    models = list_models(api_url)
    if not models:
        typer.echo(f"讀不到 OpenCode Go 模型清單，沿用 {default_slug}。")
        return default_slug, None
    variants = model_variants()
    typer.echo("選擇 OpenCode Go 翻譯模型：")
    typer.echo(f"  1. 沿用目前設定：{default_slug}")
    for index, slug in enumerate(models, start=2):
        efforts = variants.get(slug) or []
        label = f"（variant：{'、'.join(efforts)}）" if efforts else ""
        marker = "（目前設定）" if slug == default_slug else ""
        typer.echo(f"  {index}. {slug}{label}{marker}")
    choice = _prompt_index("編號", len(models) + 1)
    if choice == 1:
        return default_slug, None
    slug = models[choice - 2]
    efforts = variants.get(slug) or []
    if len(efforts) <= 1:
        return slug, None
    default = "low" if "low" in efforts else efforts[0]
    return slug, _prompt_variant(efforts, default)


def _select_codex_model(default_slug: str) -> tuple[str, str | None]:
    models = available_models()
    if not models:
        typer.echo(f"讀不到 Codex 模型清單，沿用 {default_slug}。")
        return default_slug, None
    typer.echo("選擇 Codex 翻譯模型：")
    typer.echo(f"  1. 沿用目前設定：{default_slug}（variant：{default_effort(default_slug)}）")
    for index, model in enumerate(models, start=2):
        marker = "（目前設定）" if model.slug == default_slug else ""
        typer.echo(
            f"  {index}. {model.slug} — {model.display_name}（variant：{model.default_effort}）{marker}"
        )
    choice = _prompt_index("編號", len(models) + 1)
    if choice == 1:
        return default_slug, None
    model = models[choice - 2]
    if len(model.efforts) <= 1:
        return model.slug, None
    return model.slug, _prompt_variant(model.efforts, model.default_effort)


def _prompt_index(prompt: str, count: int) -> int:
    while True:
        choice = typer.prompt(prompt, default=1, type=int)
        if 1 <= choice <= count:
            return choice
        typer.echo(f"請輸入 1 到 {count}。")


def _prompt_variant(efforts: list[str], default: str) -> str:
    while True:
        variant = typer.prompt(f"variant（{'、'.join(efforts)}）", default=default)
        if variant in efforts:
            return variant
        typer.echo(f"variant 只能是：{'、'.join(efforts)}")


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
