#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path


DEFAULT_ENV_BIN = Path("/home/cicadmin/cosyvoice_work/env/bin")


def find_tool(name: str) -> str:
    env_tool = DEFAULT_ENV_BIN / name
    if env_tool.exists():
        return str(env_tool)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"Cannot find {name}. Install FFmpeg or use the CosyVoice environment.")


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


def wav_duration(path: Path, ffprobe: str) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except wave.Error:
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


def write_filelist(paths: list[Path], filelist: Path) -> None:
    with filelist.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{quote_concat_path(path)}'\n")


def write_metadata(title: str, chapter_title: str, duration_ms: int, metadata: Path) -> None:
    with metadata.open("w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        handle.write(f"title={escape_metadata(title)}\n")
        handle.write("artist=CosyVoice\n")
        handle.write("encoder=CosyVoice local TTS + FFmpeg\n\n")
        handle.write("[CHAPTER]\n")
        handle.write("TIMEBASE=1/1000\n")
        handle.write("START=0\n")
        handle.write(f"END={duration_ms}\n")
        handle.write(f"title={escape_metadata(chapter_title)}\n")


def default_output_name(book_dir: Path, prefix: str, chapter_title: str) -> str:
    book_name = book_dir.name
    if prefix:
        return f"{book_name}_{chapter_title}.m4b"
    return f"{book_name}.m4b"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package CosyVoice WAV segments into an M4B audiobook preview."
    )
    parser.add_argument(
        "book_dir",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Directory containing segments/. Default: current directory.",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        help="Directory containing WAV files. Default: BOOK_DIR/segments.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only include WAV files whose filename starts with this prefix, e.g. 000_.",
    )
    parser.add_argument(
        "--glob",
        default="*.wav",
        help="WAV filename glob inside segments-dir. Ignored when --prefix is set.",
    )
    parser.add_argument("--title", help="M4B book title metadata. Default: BOOK_DIR name.")
    parser.add_argument("--chapter-title", default="第一章", help="Chapter title metadata.")
    parser.add_argument("--output", type=Path, help="Output .m4b path. Default: BOOK_DIR/<name>.m4b.")
    parser.add_argument("--bitrate", default="64k", help="AAC bitrate. Default: 64k.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep intermediate raw_audio.m4a.")
    parser.add_argument("--dry-run", action="store_true", help="List selected WAV files without encoding.")
    args = parser.parse_args()

    book_dir = args.book_dir.expanduser().resolve()
    segments_dir = (args.segments_dir.expanduser().resolve() if args.segments_dir else book_dir / "segments")
    pattern = f"{args.prefix}*.wav" if args.prefix else args.glob
    segments = sorted(segments_dir.glob(pattern))
    if not segments:
        raise SystemExit(f"No WAV files found: {segments_dir}/{pattern}")

    title = args.title or book_dir.name
    output = args.output.expanduser() if args.output else book_dir / default_output_name(book_dir, args.prefix, args.chapter_title)
    if output.suffix.lower() != ".m4b":
        output = output.with_suffix(".m4b")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")

    durations = [wav_duration(path, ffprobe) for path in segments]
    duration = sum(durations)
    duration_ms = int(round(duration * 1000))

    print(f"Book directory: {book_dir}")
    print(f"Segments: {len(segments)} files from {segments_dir}/{pattern}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Output: {output}")

    if args.dry_run:
        for path, seconds in zip(segments, durations):
            print(f"{path.name}\t{seconds:.3f}s")
        return 0

    filelist = output.parent / f"{output.stem}_filelist.txt"
    metadata = output.parent / f"{output.stem}_metadata.txt"
    raw_audio = output.parent / f"{output.stem}_raw_audio.m4a"

    write_filelist(segments, filelist)
    write_metadata(title, args.chapter_title, duration_ms, metadata)

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
            args.bitrate,
            "-movflags",
            "+faststart",
            str(raw_audio),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
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
            str(output),
        ],
        check=True,
    )

    if not args.keep_raw:
        raw_audio.unlink(missing_ok=True)

    summary = {
        "output": str(output),
        "segments": len(segments),
        "duration_seconds": duration,
        "title": title,
        "chapter_title": args.chapter_title,
        "filelist": str(filelist),
        "metadata": str(metadata),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
