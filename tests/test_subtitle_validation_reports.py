import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "src/video_subtitle/commands"


class SubtitleValidationReportTests(unittest.TestCase):
    def test_timing_validator_writes_a_report_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            srt = root / "short.srt"
            report = root / "timing.json"
            srt.write_text("1\n00:00:00,000 --> 00:00:00,100\n测试\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "validate_srt.py"), str(srt), "--report", str(report)],
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertEqual(payload["block_count"], 1)
            self.assertTrue(payload["errors"])

    def test_style_validator_writes_a_report_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            srt = root / "punctuation.srt"
            report = root / "style.json"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n测试，字幕\n\n"
                "2\n00:00:01,100 --> 00:00:02,100\n增长20％\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "validate_subtitle_style.py"), str(srt), "--report", str(report)],
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["valid"])
            self.assertEqual(payload["block_count"], 2)
            self.assertEqual(len(payload["errors"]), 2)

    def test_style_validator_allows_semantic_symbols_without_rewriting_the_srt(self):
        texts = [
            "玛丽·居里",
            "术语ready-made示例",
            "英文don't stop示例",
            "英文don’t stop示例",
            "数值3.5",
            "增长20%",
            "比例1/1",
            "时间12:30",
            "日期2026-07",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            srt = root / "allowed.srt"
            report = root / "style.json"
            blocks = []
            for index, text in enumerate(texts, start=1):
                blocks.append(
                    f"{index}\n00:00:{index:02d},000 --> 00:00:{index:02d},900\n{text}"
                )
            original = "\n\n".join(blocks) + "\n"
            srt.write_text(original, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "validate_subtitle_style.py"), str(srt), "--report", str(report)],
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["errors"], [])
            self.assertEqual(srt.read_text(encoding="utf-8"), original)

    def test_token_mapping_preserves_allowed_symbols_in_the_generated_srt(self):
        texts = [
            "玛丽·居里", "ready-made", "don't stop", "don’t stop", "数值3.5",
            "增长20%", "变化3.5%", "比例1/1", "时间12:30", "日期2026-07",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cues = root / "cues.txt"
            tokens = root / "tokens.json"
            output = root / "output.srt"
            cues.write_text("\n".join(texts) + "\n", encoding="utf-8")
            tokens.write_text(json.dumps({"items": [
                {"text": text, "start": float(index), "end": float(index) + 0.8}
                for index, text in enumerate(texts)
            ]}, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "tokens_to_srt.py"), str(cues), str(output),
                 "--tokens", f"{tokens}:0"],
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            rendered = output.read_text(encoding="utf-8")
            for text in texts:
                self.assertIn(text, rendered)

    def test_token_mapping_reports_invalid_token_times_and_affected_cues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cues = root / "cues.txt"
            tokens = root / "tokens.json"
            output = root / "output.srt"
            report = root / "mapping.json"
            cues.write_text("塞尚的苹果\n", encoding="utf-8")
            tokens.write_text(json.dumps({"items": [
                {"text": "塞尚", "start": 1.0, "end": 1.4},
                {"text": "的", "start": 1.4, "end": 1.4},
                {"text": "苹果", "start": 1.4, "end": 2.0},
            ]}, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "tokens_to_srt.py"), str(cues), str(output),
                 "--tokens", f"{tokens}:0", "--report", str(report)],
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["invalid_time_item_count"], 1)
            self.assertEqual(payload["invalid_time_items"][0]["token_index"], 2)
            self.assertEqual(payload["cues_with_invalid_time_items"], [{
                "cue_number": 1,
                "text": "塞尚的苹果",
                "invalid_token_indices": [2],
            }])


if __name__ == "__main__":
    unittest.main()
