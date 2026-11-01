# epub-to-audiobook

將 EPUB 電子書轉換為 M4B 有聲書的工具腳本集，支援章節標記、封面嵌入與斷點續傳。

## 腳本說明

### `epub_to_m4b_gemini.py` — Google Gemini 3.8 Flash TTS（旗艦推薦）
使用 Google 官方最新 `google-genai` SDK（Interactions API & Voices API），支援高品質錄音室等級的台灣口音朗讀。

**特點：**
- **模型選擇**：支援 `gemini-3.8-flash-tts`（預設，擬真與演繹張力最佳）與 `gemini-3.8-flash-lite-tts`（高速、高性價比）。
- **台灣男女聲預設**：內建 `--preset female`（女聲，預設）與 `--preset male`（男聲），搭配 `speech_metadata.style` 調校道地台灣國語。
- **Voice Design 專屬聲線**：支援 `--design-voice` 一鍵透過 AI 設計專屬台灣旁白聲線，儲存永久 `voice_id` 並生成繁中試聽檔（`voice_previews/`）。
- **章節範圍篩選**：支援 1-based `--chapters 1` 或 `--chapters 1-3`，可先轉單章試聽效果再跑全書。
- **排版優化**：自動過濾書籍內文註腳號碼（如 `①`、`②`、`[1]`），避免唸出不必要的註解標號；支援巢狀標籤與章節跨檔補齊。
- **斷點續傳與 M4B 封裝**：已合成段落自動跳過；完成後自動打包為含章節標記與封面的 `.m4b` 檔案（不需字幕檔）。

#### 快速開始

1. **安裝依賴**：
   ```bash
   pip install -U "google-genai>=2.30.1" ebooklib beautifulsoup4
   # 需要 ffmpeg (例如: brew install ffmpeg)
   ```

2. **設定 API 金鑰**：
   ```bash
   export GEMINI_TTS_API_KEY="你的Gemini金鑰"
   # 或在專案目錄建立 .env 檔案：GEMINI_TTS_API_KEY=你的金鑰
   ```

3. **步驟一：建立並試聽台灣口音專屬聲線（可選）**：
   ```bash
   # 設計女聲台灣旁白（會存下 voice_id 至 gemini_voices.json，並產出試聽檔）
   python epub_to_m4b_gemini.py --design-voice --preset female

   # 設計男聲台灣旁白
   python epub_to_m4b_gemini.py --design-voice --preset male
   ```
   產出的試聽檔位在 `voice_previews/preview_female_zh_tw.wav` 與 `voice_previews/preview_male_zh_tw.wav`，可直接播放確認滿意度。

4. **步驟二：檢視章節結構（乾跑不消耗 API）**：
   ```bash
   python epub_to_m4b_gemini.py --epub "book.epub" --dry-run
   ```

5. **步驟三：試轉單一章節（例如第 4 章）**：
   ```bash
   python epub_to_m4b_gemini.py \
     --epub "book.epub" \
     --output-dir "output/book_female" \
     --preset female \
     --chapters 4
   ```

6. **步驟四：轉換全書**：
   ```bash
   # 使用 gemini-3.8-flash-tts 全書轉檔
   python epub_to_m4b_gemini.py \
     --epub "book.epub" \
     --output-dir "output/book_female" \
     --preset female

   # 或使用 gemini-3.8-flash-lite-tts 節省額度
   python epub_to_m4b_gemini.py \
     --epub "book.epub" \
     --output-dir "output/book_lite" \
     --model gemini-3.8-flash-lite-tts \
     --preset male
   ```

---

### `epub_to_m4b_edgetts.py` — Edge-TTS 雲端語音
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
