#!/usr/bin/env python3
import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urldefrag

import ebooklib
import torch
import torchaudio
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub

sys.path.append("third_party/Matcha-TTS")

import cosyvoice.cli.cosyvoice as cosyvoice_module
from cosyvoice.cli.cosyvoice import AutoModel


DEFAULT_INSTRUCT = "一個帶有磁性的成熟男性中文聲音，語速適中，口齒清晰，適合長篇有聲書朗讀。"
DEFAULT_CV3_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


@dataclass
class Chapter:
    index: int
    title: str
    href: str
    text: str


def quiet_third_party_logs() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("modelscope").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cosyvoice_module.tqdm = lambda iterable, *args, **kwargs: iterable


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_element_text(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in {"script", "style"}:
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


def child_has_anchor(child: Tag | NavigableString, anchor: str) -> bool:
    if not anchor or not isinstance(child, Tag):
        return False
    if child.get("id") == anchor or child.get("name") == anchor:
        return True
    return child.find(id=anchor) is not None or child.find(attrs={"name": anchor}) is not None


def anchor_child_index(children: list[Tag | NavigableString], anchor: str) -> int:
    if not anchor:
        return 0
    for idx, child in enumerate(children):
        if child_has_anchor(child, anchor):
            return idx
    return 0


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


def extract_chapters(epub_path: Path, min_chars: int) -> list[Chapter]:
    book = epub.read_epub(str(epub_path))
    documents = spine_document_items(book)
    doc_order = {item.get_name(): idx for idx, item in enumerate(documents)}
    doc_by_name = {item.get_name(): item for item in documents}
    bodies: dict[str, list[Tag | NavigableString]] = {}

    for item in documents:
        soup = BeautifulSoup(item.get_content(), "html.parser")
        body = soup.body or soup
        bodies[item.get_name()] = list(body.children)

    raw_links = toc_links(book.toc)
    links = []
    for title, href in raw_links:
        file_name, anchor = split_href(href)
        if file_name in doc_by_name:
            links.append((title or file_name, href, file_name, anchor))

    if not links:
        chapters = []
        for idx, item in enumerate(documents):
            text = clean_text(BeautifulSoup(item.get_content(), "html.parser").get_text("\n", strip=True))
            if len(text) >= min_chars:
                chapters.append(Chapter(idx, item.get_name(), item.get_name(), text))
        return chapters

    def sort_key(link: tuple[str, str, str, str]) -> tuple[int, int]:
        _, _, file_name, anchor = link
        return doc_order.get(file_name, 10**9), anchor_child_index(bodies[file_name], anchor)

    links.sort(key=sort_key)
    chapters: list[Chapter] = []

    for idx, (title, href, file_name, anchor) in enumerate(links):
        if title.strip() in {"目錄", "目录", "Table of Contents"}:
            continue

        children = bodies[file_name]
        start = anchor_child_index(children, anchor)
        end = len(children)
        if idx + 1 < len(links) and links[idx + 1][2] == file_name:
            end = anchor_child_index(children, links[idx + 1][3])

        parts = [get_element_text(child) for child in children[start:end]]
        text = clean_text("\n".join(part for part in parts if part.strip()))
        if len(text) < min_chars:
            continue
        chapters.append(Chapter(len(chapters), title.strip(), href, text))

    return chapters


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
                chunks.append(sentence[start:start + max_chars].strip())
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
    info = torchaudio.info(str(path))
    return info.num_frames / float(info.sample_rate)


def cv3_prompt_text(text: str) -> str:
    if "<|endofprompt|>" in text:
        return text
    return f"You are a helpful assistant.<|endofprompt|>{text}"


def cv3_instruct_text(text: str) -> str:
    if "<|endofprompt|>" in text:
        return text
    return f"You are a helpful assistant. {text}<|endofprompt|>"


def synthesize_segment(
    model,
    text: str,
    path: Path,
    speaker: str,
    instruct: str,
    prompt_text: str,
    prompt_wav: Path,
    speed: float,
    retries: int,
    mode: str,
) -> float:
    if path.exists() and path.stat().st_size > 44:
        return wav_duration(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.wav")
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        try:
            parts = []
            with torch.inference_mode():
                if mode == "sft":
                    outputs = model.inference_sft(text, speaker, stream=False, speed=speed)
                elif mode == "cv3-zero-shot":
                    outputs = model.inference_zero_shot(
                        text,
                        cv3_prompt_text(prompt_text),
                        str(prompt_wav),
                        stream=False,
                        speed=speed,
                    )
                elif mode == "cv3-instruct":
                    outputs = model.inference_instruct2(
                        text,
                        cv3_instruct_text(instruct),
                        str(prompt_wav),
                        stream=False,
                        speed=speed,
                    )
                else:
                    outputs = model.inference_instruct(text, speaker, instruct, stream=False, speed=speed)
                for output in outputs:
                    parts.append(output["tts_speech"].detach().cpu())
            if not parts:
                raise RuntimeError("CosyVoice returned no audio")
            wave = torch.cat(parts, dim=1)
            torchaudio.save(str(tmp_path), wave, model.sample_rate)
            tmp_path.replace(path)
            return wave.shape[1] / float(model.sample_rate)
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt <= retries:
                time.sleep(2)

    raise RuntimeError(f"failed to synthesize segment after {retries + 1} attempts: {last_error}") from last_error


def quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def escape_metadata(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def write_concat_list(paths: list[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{quote_concat_path(path)}'\n")


def write_ffmetadata(book_title: str, chapters: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        handle.write(f"title={escape_metadata(book_title)}\n")
        handle.write("artist=CosyVoice\n")
        handle.write("encoder=CosyVoice local TTS + FFmpeg\n\n")
        for chapter in chapters:
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={chapter['start_ms']}\n")
            handle.write(f"END={chapter['end_ms']}\n")
            handle.write(f"title={escape_metadata(chapter['title'])}\n\n")


def package_m4b(output_dir: Path, book_title: str, chapter_metadata: list[dict], segment_paths: list[Path], bitrate: str) -> Path:
    filelist = output_dir / "filelist.txt"
    metadata = output_dir / "ffmpeg_metadata.txt"
    raw_audio = output_dir / "raw_audio.m4a"
    final_m4b = output_dir / f"{book_title}.m4b"

    write_concat_list(segment_paths, filelist)
    write_ffmetadata(book_title, chapter_metadata, metadata)

    ffmpeg = Path(sys.executable).with_name("ffmpeg")
    if not ffmpeg.exists():
        ffmpeg = Path(shutil.which("ffmpeg") or "ffmpeg")

    subprocess.run(
        [
            str(ffmpeg),
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
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_audio),
            "-i",
            str(metadata),
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-c:a",
            "copy",
            str(final_m4b),
        ],
        check=True,
    )
    return final_m4b


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an EPUB into a chaptered M4B audiobook with local CosyVoice.")
    parser.add_argument("--epub", type=Path, default=Path("/home/cicadmin/CozyVoidce/富爸爸窮爸爸.epub"))
    parser.add_argument("--output-dir", type=Path, default=Path("richdad_audiobook"))
    parser.add_argument("--model-dir", default="pretrained_models/CosyVoice-300M-Instruct")
    parser.add_argument("--mode", choices=("instruct", "sft", "cv3-zero-shot", "cv3-instruct"), default="instruct")
    parser.add_argument("--speaker", default="中文男")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--prompt-text", default=DEFAULT_CV3_PROMPT_TEXT)
    parser.add_argument("--prompt-wav", type=Path, default=Path(__file__).resolve().parent / "asset" / "zero_shot_prompt.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-chars", type=int, default=360)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--bitrate", default="64k")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        quiet_third_party_logs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = args.output_dir / "segments"
    chapters = extract_chapters(args.epub, args.min_chars)
    book = epub.read_epub(str(args.epub))
    title_metadata = book.get_metadata("DC", "title")
    book_title = title_metadata[0][0].strip() if title_metadata else args.epub.stem

    print(f"Book: {book_title}")
    print(f"Chapters: {len(chapters)}")
    for chapter in chapters:
        chunks = split_text(chapter.text, args.max_chars)
        print(f"{chapter.index:02d}. {chapter.title} - {len(chapter.text)} chars, {len(chunks)} segments")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "book_title": book_title,
                "epub": str(args.epub),
                "model_dir": args.model_dir,
                "mode": args.mode,
                "speaker": args.speaker,
                "instruct": args.instruct,
                "prompt_text": args.prompt_text,
                "prompt_wav": str(args.prompt_wav),
                "speed": args.speed,
                "max_chars": args.max_chars,
                "chapters": [
                    {"index": chapter.index, "title": chapter.title, "href": chapter.href, "chars": len(chapter.text)}
                    for chapter in chapters
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        return 0

    torch.set_default_dtype(torch.float32)
    if args.mode.startswith("cv3-") and not args.prompt_wav.exists():
        raise FileNotFoundError(f"prompt wav not found: {args.prompt_wav}")

    model = AutoModel(model_dir=args.model_dir, fp16=False)
    if args.mode in {"instruct", "sft"}:
        speakers = model.list_available_spks()
        if args.speaker not in speakers:
            raise ValueError(f"speaker {args.speaker!r} not found. Available: {speakers}")

    print(f"Sample rate: {model.sample_rate}")
    print(f"Mode: {args.mode}")
    if args.mode.startswith("cv3-"):
        print(f"Prompt wav: {args.prompt_wav}")
    else:
        print(f"Speaker: {args.speaker}")

    chapter_metadata: list[dict] = []
    segment_paths: list[Path] = []
    total_seconds = 0.0

    for chapter in chapters:
        chunks = split_text(chapter.text, args.max_chars)
        chapter_start = total_seconds
        print(f"Starting chapter {chapter.index + 1}/{len(chapters)}: {chapter.title} ({len(chunks)} segments)")

        for segment_idx, chunk in enumerate(chunks):
            segment_path = segments_dir / f"{chapter.index:03d}_{segment_idx:04d}.wav"
            duration = synthesize_segment(
                model=model,
                text=chunk,
                path=segment_path,
                speaker=args.speaker,
                instruct=args.instruct,
                prompt_text=args.prompt_text,
                prompt_wav=args.prompt_wav,
                speed=args.speed,
                retries=args.retries,
                mode=args.mode,
            )
            segment_paths.append(segment_path)
            total_seconds += duration
            if segment_idx == 0 or (segment_idx + 1) % 10 == 0 or segment_idx + 1 == len(chunks):
                print(
                    f"  segment {segment_idx + 1}/{len(chunks)} "
                    f"done, chapter audio {total_seconds - chapter_start:.1f}s, total {total_seconds:.1f}s",
                    flush=True,
                )

        chapter_metadata.append(
            {
                "title": chapter.title,
                "start_ms": int(round(chapter_start * 1000)),
                "end_ms": int(round(total_seconds * 1000)),
            }
        )

    (args.output_dir / "chapters.json").write_text(
        json.dumps(chapter_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.no_package:
        return 0

    final_m4b = package_m4b(args.output_dir, book_title, chapter_metadata, segment_paths, args.bitrate)
    print(f"Created: {final_m4b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
