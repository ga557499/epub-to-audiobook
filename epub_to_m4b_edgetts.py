#!/usr/bin/env python3
import argparse
import asyncio
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
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub
import edge_tts

@dataclass
class Chapter:
    index: int
    title: str
    href: str
    text: str


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


def find_tool(name: str) -> str:
    env_tool = Path("/home/cicadmin/cosyvoice_work/env/bin") / name
    if env_tool.exists():
        return str(env_tool)
    found = shutil.which(name)
    if found:
        return found
    return name


def audio_duration(path: Path) -> float:
    ffprobe = find_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return float(result.stdout.strip())


async def async_iter_with_timeout(aiterable, timeout: float = 45.0):
    it = aiterable.__aiter__()
    while True:
        try:
            val = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            break
        yield val


async def synthesize_segment(
    text: str,
    audio_path: Path,
    cues_path: Path,
    voice: str,
    rate: str,
    volume: str,
    retries: int,
    sub_type: str = "SentenceBoundary"
) -> tuple[float, list[dict]]:
    if audio_path.exists() and cues_path.exists() and audio_path.stat().st_size > 0:
        try:
            duration = audio_duration(audio_path)
            cues = json.loads(cues_path.read_text(encoding="utf-8"))
            return duration, cues
        except Exception:
            pass # Re-synthesize if duration probe or JSON load fails

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_audio_path = audio_path.with_suffix(".tmp.mp3")
    tmp_cues_path = cues_path.with_suffix(".tmp.json")
    last_error = None

    for attempt in range(1, retries + 2):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
            submaker = edge_tts.SubMaker()
            
            with open(str(tmp_audio_path), "wb") as audio_file:
                async for chunk in async_iter_with_timeout(communicate.stream(), timeout=45.0):
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == sub_type:
                        submaker.feed(chunk)
            
            duration = audio_duration(tmp_audio_path)
            
            cues = []
            for cue in submaker.cues:
                cues.append({
                    "start_ms": int(cue.start.total_seconds() * 1000),
                    "end_ms": int(cue.end.total_seconds() * 1000),
                    "text": cue.content
                })
            
            tmp_cues_path.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
            
            tmp_audio_path.replace(audio_path)
            tmp_cues_path.replace(cues_path)
            return duration, cues
        except Exception as exc:
            last_error = exc
            if tmp_audio_path.exists():
                tmp_audio_path.unlink()
            if tmp_cues_path.exists():
                tmp_cues_path.unlink()
            if attempt <= retries:
                err_msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                print(f"    [Warning] Segment synthesis attempt {attempt} failed ({err_msg}). Retrying in 3s...", flush=True)
                await asyncio.sleep(3)

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
        handle.write("artist=EdgeTTS\n")
        handle.write("encoder=EdgeTTS + FFmpeg\n\n")
        for chapter in chapters:
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={chapter['start_ms']}\n")
            handle.write(f"END={chapter['end_ms']}\n")
            handle.write(f"title={escape_metadata(chapter['title'])}\n\n")


def extract_cover_image(book: epub.EpubBook, output_dir: Path) -> Path | None:
    for item in book.get_items():
        name = item.get_name().lower()
        if "cover" in name and "image" in item.media_type:
            suffix = Path(item.get_name()).suffix or ".jpg"
            cover_path = output_dir / f"cover{suffix}"
            cover_path.write_bytes(item.get_content())
            return cover_path
    for item in book.get_items():
        if "image" in item.media_type:
            suffix = Path(item.get_name()).suffix or ".jpg"
            cover_path = output_dir / f"cover{suffix}"
            cover_path.write_bytes(item.get_content())
            return cover_path
    return None


def package_m4b(
    output_dir: Path,
    book_title: str,
    chapter_metadata: list[dict],
    segment_paths: list[Path],
    bitrate: str,
    cover_path: Path | None = None,
) -> Path:
    filelist = output_dir / "filelist.txt"
    metadata = output_dir / "ffmpeg_metadata.txt"
    raw_audio = output_dir / "raw_audio.m4a"
    final_m4b = output_dir / f"{book_title}.m4b"

    write_concat_list(segment_paths, filelist)
    write_ffmetadata(book_title, chapter_metadata, metadata)

    ffmpeg = find_tool("ffmpeg")

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
    if cover_path and cover_path.exists():
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
        cmd.extend(
            [
                "-map",
                "0:a",
            ]
        )

    cmd.extend(
        [
            "-map_metadata",
            "1",
            "-c:a",
            "copy",
            str(final_m4b),
        ]
    )

    subprocess.run(cmd, check=True)
    raw_audio.unlink(missing_ok=True)
    return final_m4b


def format_timestamp(ms: int, srt_format: bool = True) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sep = "," if srt_format else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{milliseconds:03d}"


def generate_srt(cues: list[dict]) -> str:
    lines = []
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        start_ts = format_timestamp(cue["start_ms"], srt_format=True)
        end_ts = format_timestamp(cue["end_ms"], srt_format=True)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines)


def generate_vtt(cues: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for idx, cue in enumerate(cues, start=1):
        lines.append(str(idx))
        start_ts = format_timestamp(cue["start_ms"], srt_format=False)
        end_ts = format_timestamp(cue["end_ms"], srt_format=False)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines)


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Convert an EPUB into a chaptered M4B audiobook using Edge-TTS with subtitle generation.")
    parser.add_argument("--epub", type=Path, required=True, help="Path to input EPUB file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save segments and final M4B.")
    parser.add_argument("--voice", default="zh-TW-YunJheNeural", help="Edge-TTS voice to use (default: zh-TW-YunJheNeural).")
    parser.add_argument("--rate", default="+0%", help="Speed rate change (default: +0%).")
    parser.add_argument("--volume", default="+0%", help="Volume change (default: +0%).")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max characters per synthesis segment (default: 2000).")
    parser.add_argument("--min-chars", type=int, default=40, help="Min characters to qualify as a chapter (default: 40).")
    parser.add_argument("--bitrate", default="64k", help="Output audio bitrate (default: 64k).")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries per segment synthesis.")
    parser.add_argument("--sub-boundary", choices=("SentenceBoundary", "WordBoundary"), default="SentenceBoundary", help="Subtitle boundary type (default: SentenceBoundary).")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run to inspect chapters and segment counts.")
    parser.add_argument("--no-package", action="store_true", help="Skip packaging segments into final M4B.")
    args = parser.parse_args()

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
                "voice": args.voice,
                "rate": args.rate,
                "volume": args.volume,
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

    print(f"Voice: {args.voice}")
    print(f"Rate: {args.rate}")
    print(f"Volume: {args.volume}")
    print(f"Subtitle Boundary: {args.sub_boundary}")

    chapter_metadata: list[dict] = []
    segment_paths: list[Path] = []
    master_cues: list[dict] = []
    total_seconds = 0.0

    for chapter in chapters:
        chunks = split_text(chapter.text, args.max_chars)
        chapter_start = total_seconds
        print(f"Starting chapter {chapter.index + 1}/{len(chapters)}: {chapter.title} ({len(chunks)} segments)")

        for segment_idx, chunk in enumerate(chunks):
            segment_path = segments_dir / f"{chapter.index:03d}_{segment_idx:04d}.mp3"
            cues_path = segments_dir / f"{chapter.index:03d}_{segment_idx:04d}.json"
            
            duration, cues = await synthesize_segment(
                text=chunk,
                audio_path=segment_path,
                cues_path=cues_path,
                voice=args.voice,
                rate=args.rate,
                volume=args.volume,
                retries=args.retries,
                sub_type=args.sub_boundary,
            )
            segment_paths.append(segment_path)
            
            # Shift cue timings by the accumulated offset
            offset_ms = int(round(total_seconds * 1000))
            for cue in cues:
                master_cues.append({
                    "start_ms": cue["start_ms"] + offset_ms,
                    "end_ms": cue["end_ms"] + offset_ms,
                    "text": cue["text"]
                })
                
            total_seconds += duration
            if segment_idx == 0 or (segment_idx + 1) % 5 == 0 or segment_idx + 1 == len(chunks):
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

    cover_path = extract_cover_image(book, args.output_dir)
    if cover_path:
        print(f"Cover image: {cover_path}")

    final_m4b = package_m4b(args.output_dir, book_title, chapter_metadata, segment_paths, args.bitrate, cover_path=cover_path)
    print(f"Created: {final_m4b}")

    # Generate master subtitles
    srt_path = args.output_dir / f"{book_title}.srt"
    vtt_path = args.output_dir / f"{book_title}.vtt"
    srt_path.write_text(generate_srt(master_cues), encoding="utf-8")
    vtt_path.write_text(generate_vtt(master_cues), encoding="utf-8")
    print(f"Created subtitles:")
    print(f"  SRT: {srt_path}")
    print(f"  VTT: {vtt_path}")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
