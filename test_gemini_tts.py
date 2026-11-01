#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from bs4 import BeautifulSoup

from epub_to_m4b_gemini import (
    clean_text,
    extract_chapters,
    load_voice_store,
    parse_chapters_arg,
    remove_note_references,
    resolve_voice_and_style,
    save_voice_store,
    split_text,
    wav_duration,
)


class TestEpubToM4bGemini(unittest.TestCase):
    def test_clean_text(self):
        raw = " 這是一段\u3000帶有全形空白   與\n\n\n\n多餘換行的文字。 "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "這是一段 帶有全形空白 與\n\n多餘換行的文字。")

    def test_remove_note_references(self):
        html = (
            "<p>測試語音合成"
            "<a role='doc-noteref' href='#fn1'>①</a>"
            "是否能正確剔除標號"
            "<span class='ref'>[2]</span>"
            "以維持朗讀自然。</p>"
        )
        soup = BeautifulSoup(html, "html.parser")
        remove_note_references(soup)
        text = soup.get_text()
        self.assertNotIn("①", text)
        self.assertNotIn("[2]", text)
        self.assertIn("測試語音合成", text)
        self.assertIn("以維持朗讀自然", text)

    def test_parse_chapters_arg(self):
        total = 10
        self.assertEqual(parse_chapters_arg("1", total), [0])
        self.assertEqual(parse_chapters_arg("1-3", total), [0, 1, 2])
        self.assertEqual(parse_chapters_arg("1,4,7-9", total), [0, 3, 6, 7, 8])

        with self.assertRaises(ValueError):
            parse_chapters_arg("15", total)

        with self.assertRaises(ValueError):
            parse_chapters_arg("abc", total)

    def test_split_text(self):
        text = "這是第一句。這是第二句！這是第三句？這是一段很長很長的句子；"
        chunks = split_text(text, max_chars=15)
        self.assertTrue(len(chunks) >= 3)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 15)
            self.assertTrue(len(chunk) > 0)
        rejoined = "".join(chunks)
        self.assertEqual(rejoined, text)

    def test_extract_chapters_on_real_epub(self):
        epub_path = Path("ebooks/財富階梯-資料科學家為你打造適用人生各階段的致富策略.epub")
        if not epub_path.is_file():
            self.skipTest("Sample epub not found")

        chapters = extract_chapters(epub_path, min_chars=40)
        self.assertEqual(len(chapters), 22)
        # Check specific chapters exist
        titles = [c.title for c in chapters]
        self.assertTrue(any("第1章" in t for t in titles))
        self.assertTrue(any("第13章" in t for t in titles))
        self.assertTrue(any("第2部" in t for t in titles))
        self.assertTrue(any("各界推薦" in t for t in titles))

        # Check chapter 4 text length
        ch4 = [c for c in chapters if "第1章" in c.title][0]
        self.assertGreater(len(ch4.text), 5000)
        self.assertNotIn("①", ch4.text)

    def test_voice_store_and_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "gemini_voices.json"
            save_voice_store(store_path, {"female": "voice_test_123"})
            store = load_voice_store(store_path)
            self.assertEqual(store.get("female"), "voice_test_123")

            class DummyArgs:
                preset = "female"
                voice = None
                use_prebuilt = False
                style = None

            voice, style, is_designed = resolve_voice_and_style(DummyArgs(), store_path)
            self.assertEqual(voice, "voice_test_123")
            self.assertTrue(is_designed)
            self.assertIn("gentle", style)

            # Test forced prebuilt
            DummyArgs.use_prebuilt = True
            voice, style, is_designed = resolve_voice_and_style(DummyArgs(), store_path)
            self.assertEqual(voice, "Sulafat")
            self.assertFalse(is_designed)
            self.assertIn("Natural Taiwan Mandarin accent", style)

    def test_wav_duration_invalid(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(b"not a wav")
            f.flush()
            dur = wav_duration(Path(f.name))
            self.assertEqual(dur, 0.0)


if __name__ == "__main__":
    unittest.main()
