# whisper-yt

丟 YouTube 網址或本機影片進去，用 Whisper 轉出字幕、透過 OpenCode Go 或 ChatGPT Codex 翻成繁中（台灣），最後拿到字幕檔和燒好字幕的 MP4。

## 安裝

需要 `uv`、`ffmpeg`，以及 OpenCode Go 的 API key：

```bash
cp .env.example .env
# 編輯 .env，填入你的金鑰
```

第一次執行 `./run.sh` 時會自動 `uv sync` 並建置 whisper.cpp；想手動先準備也可以：

```bash
uv sync
./scripts/build_whisper_cpp.sh
```

程式會讀取執行時所在目錄的 `.env`；也可以照原本方式使用 `export OPENCODE_GO_API_KEY="你的金鑰"`。若兩邊都有設定，以既有環境變數為準。

不想用 API key 的話，也可以用 ChatGPT 訂閱的 Codex 額度，只要 `codex login` 過一次就行（見下面）。

### Whisper 引擎（建議建置）

轉錄預設優先使用 **whisper.cpp**（比 PyTorch 快，GPU 上約快 1.5～2 倍）。建置一次即可：

```bash
./scripts/build_whisper_cpp.sh
```

腳本會依序自動偵測 **Vulkan → CUDA → ROCm/HIP → CPU** 並建置對應版本：

- Vulkan 是跨廠牌選項（AMD／NVIDIA／Intel 都能用），在 AMD 顯卡上通常最快。需要 `vulkaninfo`（`vulkan-tools`）與 `glslc`（`shaderc`）。
- 沒有 GPU 工具鏈時會建 CPU 版，仍然可用（約為 PyTorch CPU 的 2 倍快）。

沒有建置 whisper.cpp 也能正常使用：程式會自動退回 PyTorch 路徑，並在 CUDA／ROCm、Apple MPS、CPU 之間自動挑選。想手動指定引擎用 `--engine whisper-cpp` 或 `--engine pytorch`。

PyTorch 路徑要自己裝對 wheel（依作業系統、驅動和硬體參考[官方安裝頁](https://pytorch.org/get-started/locally/)）；只在沒建置 whisper.cpp 或想用 `--device` 指定裝置時才會用到。

> whisper.cpp 的模型（`ggml-*.bin`）和 PyTorch 模型一樣放在 `output/.models/`，首次轉錄時自動下載（下載與轉錄都會顯示進度條）。

## 用法

最快的方式是用 `run.sh`：第一次執行會自動安裝依賴並建置 whisper.cpp，之後直接轉錄。

```bash
./run.sh video.mp4
./run.sh "https://www.youtube.com/watch?v=..." --no-burn
```

也可以照原本方式直接呼叫 CLI（自行確保依賴與 whisper.cpp 已備妥）：

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

端點用 `--api-url` 換。引擎、裝置、來源語言、字型、輸出位置這些參數都看 `--help`。

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

`output/.<hash>.json` 是轉錄加翻譯的快取。翻譯會同時處理最多 4 批字幕，每批完成就顯示進度並寫入快取；中斷後重跑會接續未翻譯的字幕。換 Whisper 模型或來源語言會用另一份快取，換翻譯服務或模型則保留轉錄、只重翻一次。如果同一個路徑換了影片內容，把對應的 json 刪掉再跑。`.models/` 放 Whisper 模型檔。

OpenCode Go 的 key 只會從 `OPENCODE_GO_API_KEY` 讀，不會寫進快取。`.env` 已加入 `.gitignore`，請勿將金鑰提交到版本控制。
