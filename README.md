# epub-to-audiobook

將 EPUB 電子書轉換為 M4B 有聲書的工具腳本集，支援章節標記、字幕生成與封面嵌入。

## 腳本說明

### `epub_to_m4b_edgetts.py` — Edge-TTS 雲端語音（推薦）
使用 Microsoft Edge 免費雲端神經語音，無需 GPU，支援台灣中文口音。

```bash
python epub_to_m4b_edgetts.py \
  --epub "/path/to/book.epub" \
  --output-dir "/path/to/output_dir" \
  --voice zh-TW-YunJheNeural \
  --max-chars 2000
```

**可用台灣口音語音：**
- `zh-TW-YunJheNeural`（男聲，預設）
- `zh-TW-HsiaoChenNeural`（女聲，溫柔）
- `zh-TW-HsiaoYuNeural`（女聲，親切）

**主要功能：**
- 自動提取 EPUB 封面並嵌入 M4B
- 生成 WebVTT / SRT 字幕檔
- 支援斷點續傳（自動跳過已完成段落）

---

### `epub_to_m4b.py` — CosyVoice 本地模型
使用本地 CosyVoice 模型（需要 GPU），支援 SFT 和 CosyVoice3 Instruct 模式。

```bash
python epub_to_m4b.py \
  --epub "/path/to/book.epub" \
  --output-dir "/path/to/output_dir" \
  --model-dir pretrained_models/CosyVoice-300M-SFT \
  --mode sft \
  --speaker 中文男 \
  --max-chars 360
```

---

### `package_segments_to_m4b.py` — 段落打包工具
將已生成的音訊段落重新打包為 M4B，可用於預覽或重新封裝。

```bash
python package_segments_to_m4b.py "/path/to/output_dir" \
  --prefix 000_ \
  --title "Book Title" \
  --output "/path/to/output.m4b"
```

## 依賴需求

```bash
pip install ebooklib edge-tts
# 需要 ffmpeg
```
