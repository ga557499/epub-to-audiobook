#!/usr/bin/env python3
"""
Convert EPUB books to M4B audiobooks using Google Gemini 3.8 Flash TTS / Gemini 3.8 Flash-Lite TTS.

Features:
- Official Google GenAI SDK (Interactions API & Voices API)
- Models: gemini-3.8-flash-tts (default) and gemini-3.8-flash-lite-tts
- Presets: female (default, warm/gentle) and male (calm/steady) tailored for Taiwan Mandarin
- speech_metadata style integration for natural Taiwanese accent
- --design-voice: Create persistent Taiwan accent narrator voice via Voice Design and test with preview
- 1-based --chapters filtering (e.g. --chapters 1, --chapters 1-3)
- Breakpoint resume: automatically skips already synthesized segments
- Full EPUB chapter extraction with footnote removal and nested container support
- Automatic M4B packaging with chapter marks and embedded cover art (no subtitles)
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urldefrag

import ebooklib
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub

try:
    from google import genai
except ImportError:
    genai = None


PRESET_CONFIGS = {
    "female": {
        "gender": "female",
        "prebuilt_voice": "Sulafat",
        "display_name": "Taiwan Female Narrator",
        "design_prompt": (
            "一位30多歲的台灣專業女性有聲書旁白，使用自然親切的台灣國語口音，"
            "咬字清晰、語調溫和柔美、節奏舒緩穩定，無捲舌與兒化音，非常適合長篇故事與商業有聲書聆聽。"
        ),
        "style_prebuilt": "Natural Taiwan Mandarin accent, gentle, warm, and steady audiobook narration",
        "style_designed": "gentle, warm, and steady audiobook narration",
    },
    "male": {
        "gender": "male",
        "prebuilt_voice": "Schedar",
        "display_name": "Taiwan Male Narrator",
        "design_prompt": (
            "一位30多歲的台灣專業男性有聲書旁白，使用自然親切的台灣國語口音，"
            "咬字清晰、聲音沉穩有磁性、節奏從容知性，無捲舌與兒化音，非常適合長篇知識性書籍與有聲書聆聽。"
        ),
        "style_prebuilt": "Natural Taiwan Mandarin accent, calm, warm, and steady audiobook narration",
        "style_designed": "calm, warm, and steady audiobook narration",
    },
}

DEFAULT_SAMPLE_TEXT = (
    "歡迎收聽有聲書。這是使用 Gemini 3.8 Flash TTS 所生成的台灣口音旁白測試。"
    "希望這段聲音能在接下來的閱讀旅程中，為你帶來清晰、舒適且專注的聆聽體驗。"
)


@dataclass
class Chapter:
    index: int
    title: str
    href: str
    text: str


def load_env_file() -> None:
    """Load key-value pairs from .env in current directory or script directory if present."""
    search_dirs = [Path.cwd(), Path(__file__).resolve().parent]
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass


def resolve_api_key(cli_key: str | None) -> str:
    """Resolve Gemini API key specifically from GEMINI_TTS_API_KEY or CLI."""
    if cli_key:
        return cli_key.strip()
    load_env_file()
    key = os.environ.get("GEMINI_TTS_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "錯誤：未設定 GEMINI_TTS_API_KEY。\n"
            "請設定環境變數：export GEMINI_TTS_API_KEY=\"你的金鑰\"\n"
            "或在專案根目錄建立 .env 檔案並寫入 GEMINI_TTS_API_KEY=你的金鑰\n"
            "亦可透過參數傳入：--api-key \"你的金鑰\""
        )
    return key


def get_genai_client(api_key: str):
    if genai is None:
        raise SystemExit(
            "錯誤：未安裝 google-genai 套件。\n"
            "請執行：pip install -U \"google-genai>=2.30.1\""
        )
    return genai.Client(api_key=api_key)


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_note_references(soup: BeautifulSoup) -> None:
    """Remove footnote and note reference markers (e.g., ①, ②, 1) so narrator does not speak them."""
    for node in soup.find_all(
        lambda tag: (
            tag.name == "a"
            and (
                tag.get("role") in {"doc-noteref", "doc-biblioref"}
                or tag.get("epub:type") in {"noteref", "biblioref"}
                or "noteref" in str(tag.get("class", []))
            )
        )
    ):
        node.decompose()

    for node in soup.find_all(class_=lambda c: c and any("ref" in cls or "footnote" in cls for cls in c.split())):
        if node.name in {"span", "sup", "sub", "a"} and len(node.get_text(strip=True)) <= 4:
            node.decompose()


def get_element_text(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in {"script", "style", "nav"}:
        return ""
    if node.name == "br":
        return "\n"
    return node.get_text(" ", strip=True)


def split_href(href: str) -> tuple[str, str]:
    file_name, anchor = urldefrag(href)
    return unquote(file_name), unquote(anchor)


def toc_links(toc: Iterable) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []

    def visit(entry) -> None:
        if isinstance(entry, epub.Link):
            links.append((entry.title.strip(), entry.href))
            return
        if isinstance(entry, epub.Section):
            if entry.href:
                links.append((entry.title.strip(), entry.href))
            return
        if isinstance(entry, tuple):
            for child in entry:
                if isinstance(child, (list, tuple)):
                    for nested in child:
                        visit(nested)
                else:
                    visit(child)
            return
        if isinstance(entry, list):
            for child in entry:
                visit(child)

    for item in toc:
        visit(item)
    return links


def find_anchor_parent_and_index(container: Tag, anchor: str) -> tuple[Tag, int]:
    """Find the child index under container or nearest common parent matching the anchor."""
    if not anchor:
        return container, 0

    target = container.find(id=anchor) or container.find(attrs={"name": anchor})
    if target is None:
        return container, 0

    cur = target
    while cur.parent is not None and cur.parent != container and cur.parent.name not in {"body", "html"}:
        cur = cur.parent

    parent = cur.parent or container
    children = [c for c in parent.children if isinstance(c, (Tag, NavigableString))]
    try:
        idx = children.index(cur)
        return parent, idx
    except ValueError:
        return container, 0


def spine_document_items(book: epub.EpubBook) -> list:
    items = []
    seen = set()
    for spine_item in book.spine:
        idref = spine_item[0] if isinstance(spine_item, tuple) else spine_item
        item = book.get_item_with_id(idref)
        if item and item.get_type() == ebooklib.ITEM_DOCUMENT and item.get_name() not in seen:
            items.append(item)
            seen.add(item.get_name())
    if items:
        return items
    return [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]


def extract_chapters(
    epub_path: Path,
    min_chars: int = 40,
    keep_noterefs: bool = False,
    split_subsections: bool = False,
) -> list[Chapter]:
    book = epub.read_epub(str(epub_path))
    documents = spine_document_items(book)
    doc_order = {item.get_name(): idx for idx, item in enumerate(documents)}
    doc_by_name = {item.get_name(): item for item in documents}

    soups: dict[str, BeautifulSoup] = {}
    for item in documents:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        if not keep_noterefs:
            remove_note_references(soup)
        soups[item.get_name()] = soup

    raw_links = toc_links(book.toc)
    links = []
    for title, href in raw_links:
        file_name, anchor = split_href(href)
        if file_name in doc_by_name:
            links.append((title or file_name, href, file_name, anchor))

    if not links:
        chapters = []
        for idx, item in enumerate(documents):
            soup = soups[item.get_name()]
            text = clean_text(soup.get_text("\n", strip=True))
            if len(text) >= min_chars:
                chapters.append(Chapter(len(chapters), item.get_name(), item.get_name(), text))
        return chapters

    def sort_key(link: tuple[str, str, str, str]) -> tuple[int, int]:
        _, _, file_name, anchor = link
        soup = soups[file_name]
        body = soup.body or soup
        _, idx = find_anchor_parent_and_index(body, anchor)
        return doc_order.get(file_name, 10**9), idx

    links.sort(key=sort_key)

    # Filter out redundant subheadings if split_subsections is False and they are in the same file
    filtered_links: list[tuple[str, str, str, str]] = []
    seen_files_without_split: set[str] = set()

    for title, href, file_name, anchor in links:
        t_strip = title.strip()
        if t_strip in {"目錄", "目录", "Table of Contents", "Contents"}:
            continue
        if not split_subsections:
            if file_name in seen_files_without_split:
                continue
            seen_files_without_split.add(file_name)
        filtered_links.append((t_strip, href, file_name, anchor))

    chapters: list[Chapter] = []

    for idx, (title, href, file_name, anchor) in enumerate(filtered_links):
        soup = soups[file_name]
        body = soup.body or soup
        parent, start_idx = find_anchor_parent_and_index(body, anchor)
        children = [c for c in parent.children if isinstance(c, (Tag, NavigableString))]

        end_idx = len(children)
        if idx + 1 < len(filtered_links) and filtered_links[idx + 1][2] == file_name:
            next_parent, next_idx = find_anchor_parent_and_index(body, filtered_links[idx + 1][3])
            if next_parent == parent and next_idx > start_idx:
                end_idx = next_idx

        parts = [get_element_text(child) for child in children[start_idx:end_idx]]
        text = clean_text("\n".join(part for part in parts if part.strip()))

        # If next TOC chapter skips some spine files, append their content to this chapter
        current_doc_idx = doc_order.get(file_name, 0)
        next_doc_idx = (
            doc_order.get(filtered_links[idx + 1][2], current_doc_idx)
            if idx + 1 < len(filtered_links)
            else len(documents)
        )

        for skipped_idx in range(current_doc_idx + 1, next_doc_idx):
            skipped_doc = documents[skipped_idx]
            skipped_soup = soups[skipped_doc.get_name()]
            skipped_text = clean_text(skipped_soup.get_text("\n", strip=True))
            if len(skipped_text) >= 20:
                text = f"{text}\n\n{skipped_text}".strip()

        if len(text) < min_chars:
            continue
        chapters.append(Chapter(len(chapters), title, href, text))

    return chapters


def parse_chapters_arg(spec: str, total_chapters: int) -> list[int]:
    """Parse 1-based chapter specifications (e.g. '1', '1-3', '1,4,6'). Returns 0-based indices."""
    if not spec:
        return list(range(total_chapters))

    indices = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            sub = p.split("-", 1)
            try:
                start = int(sub[0])
                end = int(sub[1])
                for num in range(start, end + 1):
                    if 1 <= num <= total_chapters:
                        indices.add(num - 1)
            except ValueError:
                raise ValueError(f"無效的章節範圍：{p}")
        else:
            try:
                num = int(p)
                if 1 <= num <= total_chapters:
                    indices.add(num - 1)
                else:
                    raise ValueError(f"章節編號 {num} 超出範圍 (全書共 {total_chapters} 章)")
            except ValueError as e:
                raise ValueError(f"無效的章節指定：{p} ({e})")

    if not indices:
        raise ValueError(f"指定的章節參數 '{spec}' 沒有匹配到任何章節 (共 {total_chapters} 章)")
    return sorted(indices)


def split_text(text: str, max_chars: int) -> list[str]:
    raw_parts = re.split(r"([。！？!?；;])", text)
    sentences: list[str] = []
    for idx in range(0, len(raw_parts), 2):
        sentence = raw_parts[idx]
        if idx + 1 < len(raw_parts):
            sentence += raw_parts[idx + 1]
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start : start + max_chars].strip())
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current}{sentence}" if current else sentence
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if frames > 0 and rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def load_voice_store(store_path: Path) -> dict:
    if store_path.is_file():
        try:
            return json.loads(store_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_voice_store(store_path: Path, data: dict) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def design_voice(
    client,
    preset: str,
    model: str,
    voice_store_path: Path,
    custom_prompt: str | None = None,
    sample_text: str | None = None,
) -> str:
    config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS["female"])
    prompt_text = custom_prompt or config["design_prompt"]
    preview_dir = Path("voice_previews")
    preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Voice Design] 正在為 [{preset}] 建立專屬台灣口音聲線...")
    print(f"  模型：{model}")
    print(f"  描述：{prompt_text}")

    voice_res = client.voices.create(
        store=True,
        voice={
            "model": model,
            "type": "prompted",
            "display_name": f"{config['display_name']} ({preset})",
            "gender": config["gender"],
            "language_code": "zh-TW",
            "prompted": {
                "input": prompt_text,
            },
        },
    )

    voice_id = voice_res.id
    print(f"  成功建立聲線 ID: {voice_id}")

    # Save API generated sample preview if present
    if getattr(voice_res, "sample_audio", None) and voice_res.sample_audio.data:
        raw_preview = preview_dir / f"api_preview_{preset}.wav"
        raw_preview.write_bytes(base64.b64decode(voice_res.sample_audio.data))
        print(f"  API 原始預覽檔：{raw_preview}")

    # Generate a dedicated Taiwan Mandarin sample audio
    test_text = sample_text or DEFAULT_SAMPLE_TEXT
    print("  正在生成繁體中文有聲書試聽檔...")
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": test_text,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": config["style_designed"],
                            }
                        ],
                    }
                ],
            }
        ],
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [
                {"voice": voice_id},
            ],
        },
    )

    if getattr(interaction, "output_audio", None) and interaction.output_audio.data:
        sample_path = preview_dir / f"preview_{preset}_zh_tw.wav"
        sample_path.write_bytes(base64.b64decode(interaction.output_audio.data))
        dur = wav_duration(sample_path)
        print(f"  試聽檔已儲存：{sample_path} (時長：{dur:.1f} 秒)")
        print(f"  你可以直接播放此試聽檔確認音色！")

    # Update store
    store = load_voice_store(voice_store_path)
    store[preset] = voice_id
    store[f"{preset}_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_voice_store(voice_store_path, store)
    print(f"  已將 voice_id 記錄至：{voice_store_path}\n")

    return voice_id


def resolve_voice_and_style(
    args,
    voice_store_path: Path,
) -> tuple[str, str, bool]:
    preset_cfg = PRESET_CONFIGS.get(args.preset, PRESET_CONFIGS["female"])
    is_designed = False

    # 1. Direct explicit voice passed by user
    if args.voice:
        resolved_voice = args.voice
        is_designed = resolved_voice.startswith("voice_")
    elif not args.use_prebuilt:
        # 2. Check local voice store
        store = load_voice_store(voice_store_path)
        if args.preset in store and store[args.preset]:
            resolved_voice = store[args.preset]
            is_designed = True
        else:
            resolved_voice = preset_cfg["prebuilt_voice"]
    else:
        resolved_voice = preset_cfg["prebuilt_voice"]

    # 3. Resolve style
    if args.style is not None:
        resolved_style = args.style
    else:
        resolved_style = preset_cfg["style_designed"] if is_designed else preset_cfg["style_prebuilt"]

    return resolved_voice, resolved_style, is_designed


def synthesize_segment(
    client,
    model: str,
    text: str,
    voice: str,
    style: str,
    audio_path: Path,
    retries: int = 3,
    delay: float = 0.5,
) -> float:
    existing_dur = wav_duration(audio_path)
    if existing_dur > 0:
        return existing_dur

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = audio_path.with_suffix(".tmp.wav")
    last_err = None

    content_item: dict = {
        "type": "text",
        "text": text,
    }
    if style:
        content_item["annotations"] = [
            {
                "type": "speech_metadata",
                "style": style,
            }
        ]

    for attempt in range(1, retries + 2):
        try:
            active_client = client() if callable(client) else client
            interaction = active_client.interactions.create(
                model=model,
                store=False,
                input=[
                    {
                        "type": "user_input",
                        "content": [content_item],
                    }
                ],
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": voice},
                    ],
                },
                timeout=120.0,
            )

            if not getattr(interaction, "output_audio", None) or not interaction.output_audio.data:
                raise RuntimeError("Gemini API 未回傳音訊資料")

            audio_bytes = base64.b64decode(interaction.output_audio.data)
            tmp_path.write_bytes(audio_bytes)

            dur = wav_duration(tmp_path)
            if dur <= 0:
                raise RuntimeError("生成的音訊檔無效或長度為 0")

            tmp_path.replace(audio_path)
            if delay > 0:
                time.sleep(delay)
            return dur

        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            if attempt <= retries:
                wait_s = min(30.0, (2 ** attempt) + (attempt * 0.5))
                err_msg = str(exc)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    print(f"    [429 限速] 等待 {wait_s:.1f} 秒後重試段落...", flush=True)
                else:
                    print(f"    [重試 {attempt}/{retries}] 合成失敗 ({exc})，{wait_s:.1f} 秒後重試...", flush=True)
                if callable(client):
                    try:
                        client(refresh=True)
                    except TypeError:
                        pass
                time.sleep(wait_s)

    raise RuntimeError(f"段落合成失敗 ({retries + 1} 次重試後): {last_err}") from last_err


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    common_mac_paths = [
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
    ]
    for p in common_mac_paths:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).is_file():
                return exe
        except Exception:
            pass
    return None


def extract_cover_image(book: epub.EpubBook, output_dir: Path) -> Path | None:
    # 1. Try cover items declared in metadata
    meta_cover = book.get_metadata("OPF", "cover")
    if meta_cover:
        cover_id = meta_cover[0][1].get("content")
        item = book.get_item_with_id(cover_id)
        if item and "image" in item.media_type:
            suffix = Path(item.get_name()).suffix or ".jpg"
            out = output_dir / f"cover{suffix}"
            out.write_bytes(item.get_content())
            return out

    # 2. Check manifest properties cover-image
    for item in book.get_items():
        props = getattr(item, "properties", []) or []
        if "cover-image" in props and "image" in item.media_type:
            suffix = Path(item.get_name()).suffix or ".jpg"
            out = output_dir / f"cover{suffix}"
            out.write_bytes(item.get_content())
            return out

    # 3. Check filename containing cover
    for item in book.get_items():
        name = item.get_name().lower()
        if "cover" in name and "image" in getattr(item, "media_type", ""):
            suffix = Path(item.get_name()).suffix or ".jpg"
            out = output_dir / f"cover{suffix}"
            out.write_bytes(item.get_content())
            return out

    return None


def escape_metadata(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def package_m4b(
    output_dir: Path,
    book_title: str,
    chapter_metadata: list[dict],
    segment_paths: list[Path],
    bitrate: str,
    final_m4b_path: Path,
    cover_path: Path | None = None,
) -> Path:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        print("\n[注意] 找不到 ffmpeg 工具，略過 M4B 打包步驟。")
        print("  所有 WAV 段落已妥善保存在：", output_dir / "segments")
        print("  若需打包成 M4B，請先安裝 ffmpeg (例如：pip install imageio-ffmpeg 或 brew install ffmpeg) 後再次執行。")
        return final_m4b_path

    filelist = output_dir / f"{final_m4b_path.stem}_filelist.txt"
    metadata = output_dir / f"{final_m4b_path.stem}_ffmetadata.txt"
    raw_audio = output_dir / f"{final_m4b_path.stem}_raw.m4a"

    with filelist.open("w", encoding="utf-8") as handle:
        for path in segment_paths:
            handle.write(f"file '{quote_concat_path(path)}'\n")

    with metadata.open("w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        handle.write(f"title={escape_metadata(book_title)}\n")
        handle.write("artist=Google Gemini TTS\n")
        handle.write("encoder=Gemini 3.8 Flash TTS + FFmpeg\n\n")
        for ch in chapter_metadata:
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={ch['start_ms']}\n")
            handle.write(f"END={ch['end_ms']}\n")
            handle.write(f"title={escape_metadata(ch['title'])}\n\n")

    print("\n[FFmpeg] 正在將 WAV 段落編碼為 AAC...")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(filelist),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-movflags",
            "+faststart",
            str(raw_audio),
        ],
        check=True,
    )

    print("[FFmpeg] 正在注入章節標記與封面封裝成 M4B...")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_audio),
        "-i",
        str(metadata),
    ]

    if cover_path and cover_path.is_file():
        cmd.extend(
            [
                "-i",
                str(cover_path),
                "-map",
                "0:a",
                "-map",
                "2:v",
                "-c:v",
                "copy",
                "-disposition:v:0",
                "attached_pic",
            ]
        )
    else:
        cmd.extend(["-map", "0:a"])

    cmd.extend(
        [
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "copy",
            str(final_m4b_path),
        ]
    )

    subprocess.run(cmd, check=True)
    raw_audio.unlink(missing_ok=True)
    return final_m4b_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 Google Gemini 3.8 Flash TTS / Gemini 3.8 Flash-Lite TTS 將繁體中文 EPUB 轉為 M4B 有聲書。"
    )
    parser.add_argument("--epub", type=Path, help="EPUB 電子書檔案路徑。")
    parser.add_argument("--output-dir", type=Path, default=Path("output_audiobook"), help="輸出資料夾 (預設: output_audiobook)。")
    parser.add_argument("--output", type=Path, help="自訂輸出 .m4b 檔案完整路徑。")
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash-tts",
        choices=("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"),
        help="使用的 Gemini TTS 模型 (預設: gemini-3.8-flash-tts)。",
    )
    parser.add_argument(
        "--preset",
        choices=("female", "male"),
        default="female",
        help="聲音預設組 (預設: female 女聲，可選 male 男聲)。",
    )
    parser.add_argument("--voice", help="手動指定 voice_... ID 或預建聲線 (覆寫 --preset)。")
    parser.add_argument("--use-prebuilt", action="store_true", help="強制使用預建聲線 (忽略已儲存的自訂 voice_id)。")
    parser.add_argument("--style", help="自訂 speech_metadata.style 表演風格文字 (傳入空字串 '' 則不帶 style)。")
    parser.add_argument("--design-voice", action="store_true", help="為當前 preset 建立專屬台灣口音 Voice Design 聲線並試聽。")
    parser.add_argument("--voice-prompt", help="自訂 Voice Design 人設描述 (搭配 --design-voice)。")
    parser.add_argument("--sample-text", help="自訂試聽文字 (搭配 --design-voice)。")
    parser.add_argument("--voice-store", type=Path, default=Path("gemini_voices.json"), help="存放自訂 voice_id 的 JSON 檔案路徑。")
    parser.add_argument("--chapters", help="指定要轉換的 1-based 章節範圍 (例如: 1 或 1-3 或 1,4,6)。建議先 --dry-run 檢視。")
    parser.add_argument("--max-chars", type=int, default=600, help="每段合成文字上限字數 (預設: 600 字)。")
    parser.add_argument("--min-chars", type=int, default=40, help="視為有效章節的最少字數 (預設: 40 字)。")
    parser.add_argument("--bitrate", default="64k", help="輸出 M4B 音訊位元率 (預設: 64k)。")
    parser.add_argument("--retries", type=int, default=3, help="段落合成重試次數 (預設: 3)。")
    parser.add_argument("--delay", type=float, default=0.5, help="段落請求間隔秒數 (預設: 0.5s)。")
    parser.add_argument("--api-key", help="Gemini API Key (若未指定則讀取 GEMINI_TTS_API_KEY)。")
    parser.add_argument("--keep-noterefs", action="store_true", help="保留內文註腳號碼 (預設自動剔除 ①、②、[1])。")
    parser.add_argument("--split-subsections", action="store_true", help="將同一檔內的次小節拆分為獨立章節 (預設合併為大章)。")
    parser.add_argument("--overwrite", action="store_true", help="強制重新合成已存在的段落。")
    parser.add_argument("--dry-run", action="store_true", help="僅解析章節並列出資訊，不呼叫 API 也不生成音訊。")
    parser.add_argument("--no-package", action="store_true", help="僅生成 WAV 音訊段落，不執行 M4B 封裝。")

    args = parser.parse_args()

    # Lazily initialize API client so dry-run or repackaging already-synthesized WAVs works without API key
    _client = None

    def ensure_client(refresh: bool = False):
        nonlocal _client
        if _client is None or refresh:
            api_key = resolve_api_key(args.api_key)
            _client = get_genai_client(api_key)
        return _client

    if args.design_voice:
        if args.dry_run:
            print("[Dry Run] --design-voice 乾跑模式：不會呼叫 API 建立自訂聲線。")
        else:
            design_voice(
                client=ensure_client(),
                preset=args.preset,
                model=args.model,
                voice_store_path=args.voice_store,
                custom_prompt=args.voice_prompt,
                sample_text=args.sample_text,
            )
        if not args.epub:
            print("未指定 --epub，程式結束。")
            return 0

    if not args.epub:
        parser.error("請提供 --epub 檔案路徑，或加上 --design-voice 進行聲線設計。")

    if not args.epub.is_file():
        raise SystemExit(f"找不到指定的 EPUB 檔案：{args.epub}")

    # 2. Resolve Voice and Style
    voice, style, is_designed = resolve_voice_and_style(args, args.voice_store)

    # 3. Extract and filter chapters
    print(f"\n正在讀取電子書：{args.epub}")
    all_chapters = extract_chapters(
        args.epub,
        min_chars=args.min_chars,
        keep_noterefs=args.keep_noterefs,
        split_subsections=args.split_subsections,
    )

    book = epub.read_epub(str(args.epub))
    title_metadata = book.get_metadata("DC", "title")
    book_title = title_metadata[0][0].strip() if title_metadata else args.epub.stem

    selected_indices = parse_chapters_arg(args.chapters, len(all_chapters))
    selected_chapters = [all_chapters[i] for i in selected_indices]

    total_chars = sum(len(ch.text) for ch in selected_chapters)
    total_segments = sum(len(split_text(ch.text, args.max_chars)) for ch in selected_chapters)

    print(f"書名：{book_title}")
    print(f"全書章節數：{len(all_chapters)} 章")
    print(f"選取轉換章節：{len(selected_chapters)} 章 (總字數：{total_chars:,} 字，約切成 {total_segments} 段)")
    print(f"模型：{args.model}")
    print(f"預設組：{args.preset}")
    print(f"聲線：{voice} ({'自訂 Voice Design' if is_designed else '預建聲線'})")
    print(f"風格標註 (speech_metadata.style)：{style or '(無)'}\n")

    print("章節清單：")
    for original_idx in selected_indices:
        ch = all_chapters[original_idx]
        chunks = split_text(ch.text, args.max_chars)
        print(f"  [{original_idx + 1:02d}] {ch.title} — {len(ch.text):,} 字 ({len(chunks)} 段)")

    if args.dry_run:
        print("\n[Dry Run] 乾跑模式已完成，不呼叫 API 也不建立檔案。")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = args.output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_data = {
        "book_title": book_title,
        "epub": str(args.epub.resolve()),
        "model": args.model,
        "preset": args.preset,
        "voice": voice,
        "style": style,
        "max_chars": args.max_chars,
        "chapters_filter": args.chapters,
        "selected_chapters": [
            {
                "original_index": original_idx + 1,
                "title": all_chapters[original_idx].title,
                "chars": len(all_chapters[original_idx].text),
            }
            for original_idx in selected_indices
        ],
    }

    # Consistency warning
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prev.get("voice") != voice or prev.get("model") != args.model:
                print(
                    f"\n[警告] 此輸出資料夾前次使用的聲線為 '{prev.get('voice')}' ({prev.get('model')})，"
                    f"本次為 '{voice}' ({args.model})。\n"
                    f"  若要混用不同聲線請注意斷點續傳可能沿用前次檔案；"
                    f"若要全數更新請加上 --overwrite 參數，或指定新的 --output-dir。\n"
                )
        except Exception:
            pass

    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Synthesize segments
    chapter_metadata: list[dict] = []
    segment_paths: list[Path] = []
    total_seconds = 0.0

    print(f"\n開始執行語音合成 (輸出目錄: {args.output_dir})...\n")

    for ch_pos, original_idx in enumerate(selected_indices, start=1):
        ch = all_chapters[original_idx]
        chunks = split_text(ch.text, args.max_chars)
        chapter_start = total_seconds

        print(
            f"=== 正在處理章節 [{original_idx + 1}/{len(all_chapters)}] ({ch_pos}/{len(selected_chapters)}): "
            f"{ch.title} ({len(chunks)} 段) ==="
        )

        for seg_idx, chunk in enumerate(chunks, start=1):
            seg_filename = f"{original_idx:03d}_{seg_idx - 1:04d}.wav"
            seg_path = segments_dir / seg_filename

            if args.overwrite and seg_path.exists():
                seg_path.unlink(missing_ok=True)

            dur = synthesize_segment(
                client=ensure_client,
                model=args.model,
                text=chunk,
                voice=voice,
                style=style,
                audio_path=seg_path,
                retries=args.retries,
                delay=args.delay,
            )
            segment_paths.append(seg_path)
            total_seconds += dur

            if seg_idx == 1 or seg_idx % 5 == 0 or seg_idx == len(chunks):
                print(
                    f"  段落 {seg_idx:02d}/{len(chunks):02d} 完成，"
                    f"此章音訊累計 {total_seconds - chapter_start:.1f}s，全書累計 {total_seconds:.1f}s",
                    flush=True,
                )

        chapter_metadata.append(
            {
                "title": ch.title,
                "start_ms": int(round(chapter_start * 1000)),
                "end_ms": int(round(total_seconds * 1000)),
            }
        )

    (args.output_dir / "chapters.json").write_text(
        json.dumps(chapter_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n所有指定章節合成完畢！總音訊長度：{total_seconds / 60.0:.1f} 分鐘 ({total_seconds:.1f} 秒)")

    if args.no_package:
        print("[完成] --no-package 已設定，略過 M4B 封裝。")
        return 0

    # 5. Packaging
    cover_path = extract_cover_image(book, args.output_dir)
    if cover_path:
        print(f"已提取封面圖片：{cover_path}")

    if args.output:
        final_m4b = args.output.resolve()
        if final_m4b.suffix.lower() != ".m4b":
            final_m4b = final_m4b.with_suffix(".m4b")
    else:
        if args.chapters and len(selected_indices) < len(all_chapters):
            ch_suffix = f"_ch{selected_indices[0] + 1:02d}"
            if len(selected_indices) > 1:
                ch_suffix += f"-{selected_indices[-1] + 1:02d}"
            final_m4b = args.output_dir / f"{book_title}{ch_suffix}.m4b"
        else:
            final_m4b = args.output_dir / f"{book_title}.m4b"

    package_m4b(
        output_dir=args.output_dir,
        book_title=book_title,
        chapter_metadata=chapter_metadata,
        segment_paths=segment_paths,
        bitrate=args.bitrate,
        final_m4b_path=final_m4b,
        cover_path=cover_path,
    )

    if final_m4b.is_file():
        size_mb = final_m4b.stat().st_size / (1024 * 1024)
        print(f"\n[成功建立 M4B 有聲書]：{final_m4b} ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
