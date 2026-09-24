#!/usr/bin/env bash
# whisper-yt 入口腳本：自動準備環境（依賴與 whisper.cpp），再執行轉錄。
#
# 用法：./run.sh <影片或 YouTube 網址> [whisper-yt 的參數...]
# 例如：./run.sh video.mp4
#       ./run.sh "https://www.youtube.com/watch?v=..." --no-burn
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ $# -eq 0 ]; then
  echo "用法：./run.sh <影片或 YouTube 網址> [whisper-yt 的參數...]" >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || {
  echo "[run.sh] 找不到 uv，請先安裝：https://docs.astral.sh/uv/" >&2
  exit 1
}

if [ ! -d .venv ]; then
  echo "[run.sh] 首次執行：安裝 Python 依賴..."
  uv sync
fi

if [ ! -x vendor/whisper.cpp/build/bin/whisper-cli ]; then
  echo "[run.sh] 找不到 whisper.cpp，開始自動建置（首次需要幾分鐘）..."
  if ! ./scripts/build_whisper_cpp.sh; then
    echo "[run.sh] whisper.cpp 建置失敗，改用 PyTorch 路徑繼續。" >&2
  fi
fi

exec uv run --no-sync whisper-yt "$@"
