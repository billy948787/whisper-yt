from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import whisper_cpp
from .models import Subtitle

ENGINES = ("auto", "whisper-cpp", "pytorch")


def resolve_engine(engine: str, device: str = "auto") -> tuple[str, str]:
    """決定使用的引擎，回傳 (引擎名稱, 裝置描述)。

    順位：whisper.cpp（GPU）→ PyTorch（GPU）→ whisper.cpp（CPU）→ PyTorch（CPU）。
    """
    if engine not in ENGINES:
        raise ValueError(f"未知的 Whisper 引擎：{engine}")
    binary = None if engine == "pytorch" else whisper_cpp.binary_path()
    if engine == "whisper-cpp":
        if binary is None:
            raise RuntimeError(
                "找不到 whisper.cpp 執行檔，請先執行 `scripts/build_whisper_cpp.sh`，"
                "或改用 `--engine pytorch`。"
            )
        return "whisper-cpp", whisper_cpp.describe_device(binary)
    if engine == "pytorch" or device != "auto":
        return "pytorch", _pytorch_description(device)
    if binary is not None and whisper_cpp.has_gpu_backend(binary):
        return "whisper-cpp", whisper_cpp.describe_device(binary)
    if _pytorch_has_gpu():
        return "pytorch", _pytorch_description("auto")
    if binary is not None:
        return "whisper-cpp", whisper_cpp.describe_device(binary)
    return "pytorch", _pytorch_description("auto")


def detect_device(requested: str = "auto") -> tuple[str, str]:
    if requested != "auto":
        return requested, _device_description(requested)
    import torch

    if torch.cuda.is_available():
        return "cuda", _device_description("cuda")
    if torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS)"
    return "cpu", "CPU"


def _device_description(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError:
            return device.upper()
        if torch.cuda.is_available():
            runtime = "ROCm" if torch.version.hip else "CUDA"
            return f"{torch.cuda.get_device_name(0)} ({runtime})"
    return device.upper()


def _pytorch_has_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def _pytorch_description(device: str) -> str:
    _, description = detect_device(device)
    return f"PyTorch / {description}"


def transcribe(
    audio: Path,
    model_name: str,
    engine: str,
    device: str,
    language: str | None,
    model_dir: Path,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Subtitle], dict[str, Any]]:
    if engine not in ENGINES:
        raise ValueError(f"未知的 Whisper 引擎：{engine}")
    if engine == "whisper-cpp":
        return whisper_cpp.transcribe(audio, model_name, language, model_dir, log)
    selected_device, _ = detect_device(device)
    return _transcribe_pytorch(audio, model_name, selected_device, language, model_dir)


def _transcribe_pytorch(
    audio: Path,
    model_name: str,
    device: str,
    language: str | None,
    model_dir: Path,
) -> tuple[list[Subtitle], dict[str, Any]]:
    import whisper

    model = whisper.load_model(model_name, device=device, download_root=str(model_dir))
    result = model.transcribe(
        str(audio),
        language=language,
        task="transcribe",
        fp16=device.startswith("cuda"),
        verbose=False,
    )
    subtitles = [
        Subtitle(
            id=index,
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]).strip(),
        )
        for index, segment in enumerate(result["segments"], start=1)
        if str(segment["text"]).strip()
    ]
    metadata = {"language": result.get("language"), "model": model_name, "device": device}
    return subtitles, metadata
