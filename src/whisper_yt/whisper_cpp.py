"""whisper.cpp 整合：尋找執行檔、下載模型、呼叫 whisper-cli 並解析輸出。"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx

from .models import Subtitle

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"
BACKEND_MARKER = "whisper-yt-backend.txt"
GPU_BACKENDS = frozenset({"vulkan", "cuda", "hip"})
BACKEND_LABELS = {
    "vulkan": "Vulkan",
    "cuda": "CUDA",
    "hip": "ROCm (HIP)",
    "cpu": "CPU",
    "unknown": "未知",
}
USER_AGENT = "whisper-yt/0.1.0"
DOWNLOAD_CHUNK = 8 * 1024 * 1024

Log = Callable[[str], None]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def binary_path() -> Path | None:
    """回傳可用的 whisper-cli 路徑，找不到則回傳 None。"""
    override = os.getenv("WHISPER_YT_WHISPER_CPP")
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    executable = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    candidates = [
        Path.cwd() / "vendor" / "whisper.cpp" / "build" / "bin" / executable,
        project_root() / "vendor" / "whisper.cpp" / "build" / "bin" / executable,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def backend(binary: Path | None = None) -> str:
    """由建置標記或 CMakeCache 判斷執行檔的後端。"""
    binary = binary or binary_path()
    if binary is None:
        return "none"
    build_dir = binary.parent.parent
    marker = build_dir / BACKEND_MARKER
    if marker.is_file():
        value = marker.read_text(encoding="utf-8", errors="replace").strip().lower()
        if value:
            return value
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        text = cache.read_text(encoding="utf-8", errors="replace")
        for flag, name in (
            ("GGML_VULKAN:BOOL=ON", "vulkan"),
            ("GGML_CUDA:BOOL=ON", "cuda"),
            ("GGML_HIP:BOOL=ON", "hip"),
        ):
            if flag in text:
                return name
        return "cpu"
    return "unknown"


def has_gpu_backend(binary: Path | None = None) -> bool:
    return backend(binary) in GPU_BACKENDS


def describe_device(binary: Path | None = None) -> str:
    binary = binary or binary_path()
    kind = backend(binary)
    label = BACKEND_LABELS.get(kind, kind)
    if kind == "vulkan":
        devices = vulkan_devices()
        index = pick_vulkan_device(devices)
        if index is not None and index < len(devices):
            return f"whisper.cpp / {label}（{devices[index][0]}）"
    return f"whisper.cpp / {label}"


def vulkan_devices() -> list[tuple[str, bool]]:
    """回傳 [(裝置名稱, 是否為獨立顯卡)]，順序與 ggml 的 Vulkan 列舉一致。"""
    if shutil.which("vulkaninfo") is None:
        return []
    try:
        completed = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[tuple[str, bool]] = []
    name = ""
    discrete = False
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU") and stripped.endswith(":"):
            name = ""
            discrete = False
        elif stripped.startswith("deviceType"):
            discrete = "DISCRETE_GPU" in stripped
        elif stripped.startswith("deviceName"):
            name = stripped.split("=", 1)[1].strip() if "=" in stripped else ""
            devices.append((name, discrete))
    return devices


def pick_vulkan_device(devices: list[tuple[str, bool]] | None = None) -> int | None:
    override = os.getenv("WHISPER_YT_WHISPER_CPP_DEVICE")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    if devices is None:
        devices = vulkan_devices()
    if not devices:
        return None
    for index, (_, discrete) in enumerate(devices):
        if discrete:
            return index
    return 0


def physical_cores() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        physical: set[tuple[str, str]] = set()
        physical_id = core_id = None
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                if core_id is not None:
                    physical.add((physical_id or "", core_id))
                physical_id = core_id = None
            elif line.startswith("physical id"):
                physical_id = line.split(":", 1)[-1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[-1].strip()
        if core_id is not None:
            physical.add((physical_id or "", core_id))
        if physical:
            return len(physical)
    return os.cpu_count() or 4


def model_file(model: str, model_dir: Path) -> Path:
    return model_dir / f"ggml-{model}.bin"


def ensure_model(model: str, model_dir: Path, log: Log | None = None) -> Path:
    path = model_file(model, model_dir)
    if path.is_file():
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"下載 whisper.cpp 模型 {path.name}（首次使用時較久）...")
    temporary = path.with_suffix(".part")
    try:
        with httpx.stream(
            "GET",
            MODEL_URL.format(name=path.name),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(30, read=600),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            reported = 0
            step = total // 10 if total else 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(DOWNLOAD_CHUNK):
                    handle.write(chunk)
                    done += len(chunk)
                    if log and total and step and done - reported >= step and done < total:
                        reported = done
                        log(f"下載模型 {path.name}：{done * 100 // total}%")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
    if log:
        log(f"模型已就緒：{path}")
    return path


def transcribe(
    audio: Path,
    model: str,
    language: str | None,
    model_dir: Path,
    log: Log | None = None,
) -> tuple[list[Subtitle], dict[str, object]]:
    binary = binary_path()
    if binary is None:
        raise RuntimeError(
            "找不到 whisper.cpp 執行檔，請先執行 `scripts/build_whisper_cpp.sh`。"
        )
    kind = backend(binary)
    model_path = ensure_model(model, model_dir, log)
    with tempfile.TemporaryDirectory(prefix="whisper-yt-cpp-") as temp:
        output_base = Path(temp) / "transcript"
        command = [
            str(binary),
            "-m",
            str(model_path),
            "-f",
            str(audio),
            "-l",
            language or "auto",
            "-oj",
            "-of",
            str(output_base),
            "-np",
        ]
        if kind == "vulkan":
            index = pick_vulkan_device()
            if index is not None:
                command += ["-dev", str(index)]
        elif kind == "cpu":
            command += ["-t", str(physical_cores())]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"whisper.cpp 轉錄失敗（exit {completed.returncode}）：{details[-1500:]}"
            )
        data = json.loads(output_base.with_suffix(".json").read_text(encoding="utf-8"))
    subtitles = parse_transcription(data)
    metadata: dict[str, object] = {
        "language": detected_language(data) or language,
        "model": model,
        "device": describe_device(binary),
    }
    return subtitles, metadata


def parse_transcription(data: dict) -> list[Subtitle]:
    subtitles: list[Subtitle] = []
    for index, item in enumerate(data.get("transcription") or [], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets")
        offsets = offsets if isinstance(offsets, dict) else {}
        start = float(offsets.get("from") or 0) / 1000
        end = float(offsets.get("to") or 0) / 1000
        subtitles.append(Subtitle(id=index, start=start, end=end, text=text))
    return subtitles


def detected_language(data: dict) -> str | None:
    result = data.get("result")
    if isinstance(result, dict):
        value = str(result.get("language") or "").strip()
        if value:
            return value
    return None
