#!/usr/bin/env bash
# 建置 whisper.cpp（只建 whisper-cli），供 whisper-yt 使用。
#
# 用法：scripts/build_whisper_cpp.sh [auto|vulkan|cuda|hip|cpu]
# 可用環境變數：WHISPER_CPP_DIR（原始碼目錄）、JOBS（平行建置數）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="v1.9.4"
SRC="${WHISPER_CPP_DIR:-$ROOT/vendor/whisper.cpp}"
BUILD="$SRC/build"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
BACKEND="${1:-auto}"

log() { printf '[whisper-yt] %s\n' "$*"; }
die() { printf '[whisper-yt] 錯誤：%s\n' "$*" >&2; exit 1; }

for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || die "找不到 $tool，請先安裝。"
done

detect_backend() {
  if command -v glslc >/dev/null 2>&1 && command -v vulkaninfo >/dev/null 2>&1; then
    echo vulkan
  elif command -v nvcc >/dev/null 2>&1; then
    echo cuda
  elif command -v hipcc >/dev/null 2>&1; then
    echo hip
  else
    echo cpu
  fi
}

if [ "$BACKEND" = auto ]; then
  BACKEND="$(detect_backend)"
  if [ "$BACKEND" = cpu ]; then
    log "沒有偵測到 GPU 工具鏈（Vulkan／CUDA／ROCm），將建置 CPU 版本。"
  fi
fi

case "$BACKEND" in
  vulkan|cuda|hip|cpu) ;;
  *) die "不支援的 backend：$BACKEND（可用：auto、vulkan、cuda、hip、cpu）" ;;
esac

if [ ! -d "$SRC/.git" ]; then
  log "取得 whisper.cpp $VERSION ..."
  git clone --depth 1 --branch "$VERSION" https://github.com/ggml-org/whisper.cpp.git "$SRC"
fi

CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF)
case "$BACKEND" in
  vulkan)
    CMAKE_FLAGS+=(-DGGML_VULKAN=ON)
    command -v glslc >/dev/null 2>&1 || die "Vulkan 建置需要 glslc（shaderc 套件）。"
    ;;
  cuda) CMAKE_FLAGS+=(-DGGML_CUDA=ON) ;;
  hip) CMAKE_FLAGS+=(-DGGML_HIP=ON) ;;
esac

log "設定後端：$BACKEND"
cmake -S "$SRC" -B "$BUILD" "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD" --target whisper-cli -j "$JOBS"
printf '%s\n' "$BACKEND" >"$BUILD/whisper-yt-backend.txt"

log "完成：$BUILD/bin/whisper-cli（後端：$BACKEND）"
log "之後直接執行 uv run whisper-yt 即可；模型會在首次轉錄時自動下載到 output/.models。"
