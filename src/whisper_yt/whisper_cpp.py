"""whisper.cpp 整合：尋找執行檔、下載模型、呼叫 whisper-cli 並解析輸出。"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx

from .models import Subtitle

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"
VAD_MODEL_NAME = "ggml-silero-v6.2.0.bin"
VAD_MODEL_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/{name}"
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
ERROR_TAIL_LINES = 40
PROGRESS_PATTERN = re.compile(r"progress\s*=\s*(\d+)%")

Log = Callable[[str], None]
Progress = Callable[[str, int], None]


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


def parse_progress(line: str) -> int | None:
    """解析 whisper-cli 的 `progress = 45%` 輸出。"""
    match = PROGRESS_PATTERN.search(line)
    return int(match.group(1)) if match else None


def _download(
    url: str,
    path: Path,
    log: Log | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"下載模型 {path.name}（首次使用時較久）...")
    temporary = path.with_suffix(".part")
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(30, read=600),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            reported = -1
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(DOWNLOAD_CHUNK):
                    handle.write(chunk)
                    done += len(chunk)
                    if on_progress and total:
                        percent = min(done * 100 // total, 100)
                        if percent != reported:
                            reported = percent
                            on_progress(percent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
    if on_progress:
        on_progress(100)
    if log:
        log(f"模型已就緒：{path}")
    return path


def ensure_model(
    model: str,
    model_dir: Path,
    log: Log | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path:
    path = model_file(model, model_dir)
    if path.is_file():
        return path
    return _download(MODEL_URL.format(name=path.name), path, log, on_progress)


def ensure_vad_model(
    model_dir: Path,
    log: Log | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> Path | None:
    """取得 Silero VAD 模型；失敗時回傳 None（改以一般模式轉錄）。"""
    path = model_dir / VAD_MODEL_NAME
    if path.is_file():
        return path
    try:
        return _download(VAD_MODEL_URL.format(name=path.name), path, log, on_progress)
    except (httpx.HTTPError, OSError) as error:
        if log:
            log(f"無法取得 VAD 模型（{error}），將以一般模式轉錄。")
        return None


def transcribe(
    audio: Path,
    model: str,
    language: str | None,
    model_dir: Path,
    log: Log | None = None,
    on_progress: Progress | None = None,
) -> tuple[list[Subtitle], dict[str, object]]:
    binary = binary_path()
    if binary is None:
        raise RuntimeError(
            "找不到 whisper.cpp 執行檔，請先執行 `scripts/build_whisper_cpp.sh`。"
        )
    kind = backend(binary)
    download_progress = (
        (lambda percent: on_progress("download", percent)) if on_progress else None
    )
    model_path = ensure_model(model, model_dir, log, download_progress)
    vad_path = ensure_vad_model(model_dir, log, download_progress)
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
            "-pp",
        ]
        if vad_path is not None:
            command += ["--vad", "-vm", str(vad_path)]
        if kind == "vulkan":
            index = pick_vulkan_device()
            if index is not None:
                command += ["-dev", str(index)]
        elif kind == "cpu":
            command += ["-t", str(physical_cores())]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        tail: deque[str] = deque(maxlen=ERROR_TAIL_LINES)
        try:
            for line in process.stdout or ():
                tail.append(line.rstrip())
                if on_progress is not None:
                    percent = parse_progress(line)
                    if percent is not None:
                        on_progress("transcribe", percent)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        if process.returncode != 0:
            details = "\n".join(tail).strip()
            raise RuntimeError(
                f"whisper.cpp 轉錄失敗（exit {process.returncode}）：{details[-1500:]}"
            )
        json_path = output_base.with_suffix(".json")
        if not json_path.is_file():
            details = "\n".join(tail).strip()
            raise RuntimeError(f"whisper.cpp 沒有產生轉錄輸出：{details[-1500:]}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
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
