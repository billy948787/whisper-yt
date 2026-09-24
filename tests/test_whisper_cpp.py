from pathlib import Path

import pytest

from whisper_yt import whisper_cpp
from whisper_yt.models import Subtitle
from whisper_yt.transcribe import resolve_engine, transcribe


def test_model_file_name_maps_to_ggml() -> None:
    assert whisper_cpp.model_file("large-v3", Path("models")).name == "ggml-large-v3.bin"
    assert whisper_cpp.model_file("small.en", Path("models")).name == "ggml-small.en.bin"


def test_ensure_model_uses_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "ggml-small.bin"
    existing.write_bytes(b"data")
    assert whisper_cpp.ensure_model("small", tmp_path) == existing


def test_parse_transcription_reads_offsets() -> None:
    data = {
        "result": {"language": "en"},
        "transcription": [
            {"offsets": {"from": 0, "to": 2140}, "text": " hello "},
            {"offsets": {"from": 2140, "to": 4000}, "text": "   "},
            {"offsets": {"from": 4000, "to": 5000}, "text": "world"},
        ],
    }
    subtitles = whisper_cpp.parse_transcription(data)
    assert [(item.id, item.start, item.end, item.text) for item in subtitles] == [
        (1, 0.0, 2.14, "hello"),
        (3, 4.0, 5.0, "world"),
    ]
    assert whisper_cpp.detected_language(data) == "en"


def test_parse_progress_reads_whisper_output() -> None:
    assert whisper_cpp.parse_progress("whisper_print_progress_callback: progress =  45%") == 45
    assert whisper_cpp.parse_progress("whisper_print_progress_callback: progress = 100%") == 100
    assert whisper_cpp.parse_progress("some other line") is None


def test_transcribe_raises_when_output_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("")

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = iter(["failed to read audio file\n"])
            self.returncode = 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(whisper_cpp, "binary_path", lambda: binary)
    monkeypatch.setattr(whisper_cpp, "backend", lambda path=None: "cpu")
    monkeypatch.setattr(
        whisper_cpp,
        "ensure_model",
        lambda model, model_dir, log=None, on_progress=None: tmp_path / "model.bin",
    )
    monkeypatch.setattr(whisper_cpp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(RuntimeError, match="沒有產生轉錄輸出"):
        whisper_cpp.transcribe(tmp_path / "a.wav", "large-v3", "en", tmp_path)


def test_vulkan_devices_and_pick_discrete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        "Devices:\n"
        "GPU0:\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
        "\tdeviceName         = AMD Ryzen 9 9950X3D (RADV RAPHAEL_MENDOCINO)\n"
        "GPU1:\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "\tdeviceName         = AMD Radeon RX 9070 XT (RADV GFX1201)\n"
    )

    class Completed:
        stdout = output

    monkeypatch.setattr(whisper_cpp.shutil, "which", lambda name: "/usr/bin/vulkaninfo")
    monkeypatch.setattr(whisper_cpp.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.delenv("WHISPER_YT_WHISPER_CPP_DEVICE", raising=False)

    assert whisper_cpp.vulkan_devices() == [
        ("AMD Ryzen 9 9950X3D (RADV RAPHAEL_MENDOCINO)", False),
        ("AMD Radeon RX 9070 XT (RADV GFX1201)", True),
    ]
    assert whisper_cpp.pick_vulkan_device() == 1


def test_vulkan_device_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_YT_WHISPER_CPP_DEVICE", "2")
    assert whisper_cpp.pick_vulkan_device() == 2


def test_backend_reads_marker(tmp_path: Path) -> None:
    build = tmp_path / "build"
    binary = build / "bin" / "whisper-cli"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    (build / whisper_cpp.BACKEND_MARKER).write_text("vulkan\n")
    assert whisper_cpp.backend(binary) == "vulkan"
    assert whisper_cpp.has_gpu_backend(binary)


def test_backend_falls_back_to_cmake_cache(tmp_path: Path) -> None:
    build = tmp_path / "build"
    binary = build / "bin" / "whisper-cli"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    (build / "CMakeCache.txt").write_text("GGML_VULKAN:BOOL=ON\n")
    assert whisper_cpp.backend(binary) == "vulkan"
    (build / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=OFF\nGGML_HIP:BOOL=OFF\n")
    assert whisper_cpp.backend(binary) == "cpu"


def test_binary_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "whisper-cli"
    fake.write_text("")
    monkeypatch.setenv("WHISPER_YT_WHISPER_CPP", str(fake))
    assert whisper_cpp.binary_path() == fake
    monkeypatch.setenv("WHISPER_YT_WHISPER_CPP", str(tmp_path / "missing"))
    assert whisper_cpp.binary_path() is None


def test_resolve_engine_prefers_whisper_cpp_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("")
    monkeypatch.setattr(whisper_cpp, "binary_path", lambda: binary)
    monkeypatch.setattr(whisper_cpp, "has_gpu_backend", lambda path=None: True)
    monkeypatch.setattr(
        whisper_cpp, "describe_device", lambda path=None: "whisper.cpp / Vulkan（測試卡）"
    )
    monkeypatch.setattr("whisper_yt.transcribe._pytorch_has_gpu", lambda: False)
    assert resolve_engine("auto") == ("whisper-cpp", "whisper.cpp / Vulkan（測試卡）")


def test_resolve_engine_prefers_pytorch_gpu_over_cpp_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("")
    monkeypatch.setattr(whisper_cpp, "binary_path", lambda: binary)
    monkeypatch.setattr(whisper_cpp, "has_gpu_backend", lambda path=None: False)
    monkeypatch.setattr("whisper_yt.transcribe._pytorch_has_gpu", lambda: True)
    monkeypatch.setattr(
        "whisper_yt.transcribe._pytorch_description", lambda device: "PyTorch / 測試 GPU"
    )
    assert resolve_engine("auto") == ("pytorch", "PyTorch / 測試 GPU")


def test_resolve_engine_falls_back_to_pytorch_without_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(whisper_cpp, "binary_path", lambda: None)
    monkeypatch.setattr(
        "whisper_yt.transcribe._pytorch_description", lambda device: "PyTorch / CPU"
    )
    assert resolve_engine("auto") == ("pytorch", "PyTorch / CPU")


def test_resolve_engine_explicit_device_forces_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "whisper_yt.transcribe._pytorch_description", lambda device: f"PyTorch / {device}"
    )
    assert resolve_engine("auto", "cpu") == ("pytorch", "PyTorch / cpu")


def test_resolve_engine_requires_binary_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(whisper_cpp, "binary_path", lambda: None)
    with pytest.raises(RuntimeError):
        resolve_engine("whisper-cpp")


def test_transcribe_routes_to_whisper_cpp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str, str | None, Path]] = []

    def fake_transcribe(
        audio: Path,
        model: str,
        language: str | None,
        model_dir: Path,
        log: object = None,
        on_progress: object = None,
    ) -> tuple[list[Subtitle], dict[str, object]]:
        calls.append((audio, model, language, model_dir))
        return [Subtitle(1, 0, 1, "hi")], {"language": "en"}

    monkeypatch.setattr(whisper_cpp, "transcribe", fake_transcribe)
    subtitles, metadata = transcribe(
        tmp_path / "a.wav", "large-v3", "whisper-cpp", "auto", "en", tmp_path
    )
    assert calls == [(tmp_path / "a.wav", "large-v3", "en", tmp_path)]
    assert subtitles[0].text == "hi"
    assert metadata == {"language": "en"}


def test_transcribe_rejects_unknown_engine(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        transcribe(tmp_path / "a.wav", "large-v3", "nope", "auto", None, tmp_path)
