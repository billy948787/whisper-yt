from pathlib import Path
from typing import Any

import torch
import whisper

from .models import Subtitle


def detect_device(requested: str = "auto") -> tuple[str, str]:
    if requested != "auto":
        return requested, _device_description(requested)
    if torch.cuda.is_available():
        return "cuda", _device_description("cuda")
    if torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS)"
    return "cpu", "CPU"


def _device_description(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        runtime = "ROCm" if torch.version.hip else "CUDA"
        return f"{torch.cuda.get_device_name(0)} ({runtime})"
    return device.upper()


def transcribe(
    audio: Path,
    model_name: str,
    device: str,
    language: str | None,
    model_dir: Path,
) -> tuple[list[Subtitle], dict[str, Any]]:
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
