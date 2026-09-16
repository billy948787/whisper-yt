# whisper-yt

丟 YouTube 網址或本機影片進去，用 Whisper 轉出字幕、透過 OpenCode Go 或 ChatGPT Codex 翻成繁中（台灣），最後拿到字幕檔和燒好字幕的 MP4。

## 安裝

需要 `uv`、`ffmpeg`，以及 OpenCode Go 的 API key：

```bash
uv sync
export OPENCODE_GO_API_KEY="你的金鑰"
```

不想用 API key 的話，也可以用 ChatGPT 訂閱的 Codex 額度，只要 `codex login` 過一次就行（見下面）。

GPU 要自己裝對 PyTorch wheel（依作業系統、驅動和硬體參考[官方安裝頁](https://pytorch.org/get-started/locally/)）。裝好之後不用改設定，程式會在 CUDA／ROCm、Apple MPS、CPU 之間自動挑一個。

## 用法

```bash
# YouTube
uv run whisper-yt "https://www.youtube.com/watch?v=..."

# 本機影片
uv run whisper-yt "/path/to/video.mkv"

# 換小模型，只產字幕不燒影片
uv run whisper-yt video.mp4 --model medium --no-burn
```

翻譯預設走 OpenCode Go 的 `glm-5.3-flash`。互動式執行且沒指定 `--provider`、`--translation-model`、`--variant` 時，會先問你要用 OpenCode Go 還是 ChatGPT Codex，再列出該服務的模型和 variant（推理強度）讓你挑。非互動環境或想跳過選單就直接指定：

```bash
uv run whisper-yt video.mp4 --translation-model deepseek-v4.1-flash --variant max
```

端點用 `--api-url` 換。裝置、來源語言、字型、輸出位置這些參數都看 `--help`。

### 用 Codex 翻譯

不想用 API key、想用 ChatGPT 訂閱額度的話：

```bash
codex login   # 只需要做一次
uv run whisper-yt video.mp4 --provider codex
```

whisper-yt 直接讀 `~/.codex/auth.json`，token 過期會自己更新。不指定 `--translation-model` 時，互動模式會列出 `~/.codex/models_cache.json` 裡的模型和 variant；非互動環境則沿用 `~/.codex/config.toml` 的 `model`。

## 輸出

跑完 `output/` 底下會有：

- `<片名>.original.srt` — 原文
- `<片名>.zh-TW.srt`、`<片名>.zh-TW.ass` — 繁中字幕
- `<片名>.zh-TW.mp4` — 燒好字幕的影片（加 `--no-burn` 就不會產生；燒錄走 libx264）

`output/.<hash>.json` 是轉錄加翻譯的快取，中斷後重跑會從上次的地方接下去，不會重打 API。換 Whisper 模型或來源語言會用另一份快取，換翻譯服務或模型則保留轉錄、只重翻一次。如果同一個路徑換了影片內容，把對應的 json 刪掉再跑。`.models/` 放 Whisper 模型檔。

OpenCode Go 的 key 只會從 `OPENCODE_GO_API_KEY` 讀，不會寫進專案或快取。
