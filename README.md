# whisper-yt

輸入 YouTube 網址或本機影片，以 Whisper 產生逐段字幕，再透過 OpenCode Go 翻譯成繁體中文（台灣），最後輸出字幕檔與已燒錄字幕的 MP4。

## 功能

- YouTube URL 與本機影片輸入
- 預設使用 Whisper `large-v3`
- 自動選擇 CUDA／ROCm、Apple MPS 或 CPU，不綁定特定顯卡品牌
- OpenCode Go 批次翻譯與失敗重試
- 轉錄及翻譯快取，中斷後可續跑
- 輸出原文 SRT、繁中 SRT、繁中 ASS 與燒錄字幕 MP4

## 安裝

需求：`uv`、`ffmpeg`，以及 OpenCode Go API key。

```bash
uv sync
export OPENCODE_GO_API_KEY="你的金鑰"
```

`uv` 會依 `.python-version` 使用 Python 3.12。PyTorch 是否能使用 GPU 取決於作業系統、驅動與安裝的 PyTorch wheel；程式本身不寫死 ROCm 或 CUDA。

若要使用 GPU，請依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 安裝符合作業系統與硬體的 PyTorch。程式會自動偵測可用裝置，不需要修改程式碼。

## 使用方式

YouTube：

```bash
uv run whisper-yt "https://www.youtube.com/watch?v=..."
```

本機影片：

```bash
uv run whisper-yt "/path/to/video.mkv"
```

指定較小模型或只產生字幕：

```bash
uv run whisper-yt video.mp4 --model medium --no-burn
```

指定裝置、來源語言或輸出位置：

```bash
uv run whisper-yt video.mp4 --device auto --language en --output-dir output
```

完整參數：

```bash
uv run whisper-yt --help
```

預設翻譯模型為 `glm-5.3-flash`，使用官方端點 `https://opencode.ai/zen/go/v1/chat/completions`。可透過 `--translation-model` 與 `--api-url` 調整。

## 輸出

`output/` 會包含：

- `<片名>.original.srt`
- `<片名>.zh-TW.srt`
- `<片名>.zh-TW.ass`
- `<片名>.zh-TW.mp4`

隱藏的 JSON 檔是續跑快取，`.models/` 是 Whisper 模型快取。模型或來源語言參數不同時會使用不同快取；若來源檔內容已更換但路徑相同，請刪除對應 JSON 快取後重跑。

## 注意事項

- YouTube 下載需遵守影片授權、著作權與 YouTube 服務條款。
- OpenCode Go 金鑰只從 `OPENCODE_GO_API_KEY` 讀取，不會寫入專案或快取。
- 字幕燒錄使用相容性較高的 `libx264`，不依賴特定品牌的硬體編碼器。
