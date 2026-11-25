import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "src/video_subtitle/commands"
sys.path.insert(0, str(SCRIPTS))

from alignment_utils import (  # noqa: E402
    aligner_clean,
    plan_alignment_chunks,
    restore_allowed_symbols,
    validate_alignment_inputs,
)


class SubtitleAlignmentUtilsTests(unittest.TestCase):
    def test_restores_every_allowed_symbol_removed_by_qwen(self):
        expected = "玛丽·居里ready-madedon'tdon’t3.520%1/112:302026-07"
        items = [
            {"text": character, "start": index / 10, "end": (index + 1) / 10}
            for index, character in enumerate(aligner_clean(expected))
        ]

        restored, symbols = restore_allowed_symbols(items, expected)

        self.assertEqual("".join(str(item["text"]) for item in restored), expected)
        self.assertEqual(symbols, {"·": 1, "-": 2, "’": 1, ".": 1, "%": 1, "/": 1, ":": 1})

    def test_long_audio_chunks_share_audio_text_and_cue_boundaries(self):
        cues = [
            "天地玄黄", "宇宙洪荒", "日月盈昃", "辰宿列张",
            "寒来暑往", "秋收冬藏", "闰余成岁", "律吕调阳",
            "云腾致雨", "露结为霜", "金生丽水", "玉出昆冈",
        ]
        segments = [
            {"start_ms": index * 30_000, "end_ms": (index + 1) * 30_000, "text": cue}
            for index, cue in enumerate(cues)
        ]

        chunks = plan_alignment_chunks(
            segments=segments,
            transcript="".join(cues),
            cues=cues,
            target_chunk_ms=120_000,
            max_chunk_ms=150_000,
        )

        self.assertEqual(
            [(chunk["start_ms"], chunk["end_ms"], chunk["cue_start"], chunk["cue_end"])
             for chunk in chunks],
            [(0, 120_000, 0, 4), (120_000, 240_000, 4, 8), (240_000, 360_000, 8, 12)],
        )
        self.assertEqual("".join(str(chunk["text"]) for chunk in chunks), "".join(cues))

    def test_long_audio_keeps_boundaries_in_the_correct_repetition(self):
        fixture = Path(__file__).parents[1] / "examples/window-mint.cues.txt"
        cues = fixture.read_text(encoding="utf-8").splitlines() * 7
        recognized = [cue.replace("它", "他") if index // 11 not in (0, 3) else cue
                      for index, cue in enumerate(cues)]
        segment_ends = [(10, 22_029), (17, 39_961), (26, 61_135), (33, 77_854),
                        (43, 100_479), (50, 118_411), (51, 120_670), (61, 144_561),
                        (70, 165_735), (77, 183_050)]
        segments = []
        first_cue = start_ms = 0
        for last_cue, end_ms in segment_ends:
            segments.append({"start_ms": start_ms, "end_ms": end_ms,
                             "text": "".join(recognized[first_cue:last_cue])})
            first_cue, start_ms = last_cue, end_ms

        chunks = plan_alignment_chunks(segments=segments, transcript="".join(cues), cues=cues)

        self.assertEqual([(chunk["end_ms"], chunk["cue_end"]) for chunk in chunks],
                         [(120_670, 51), (183_050, 77)])

    def test_preflight_rejects_transcript_and_cues_mismatch_before_alignment(self):
        with self.assertRaisesRegex(ValueError, "transcript and cues differ"):
            validate_alignment_inputs(
                transcript="增长20%",
                cues=["增长百分之20"],
                duration_ms=10_000,
                asr_timed=None,
                max_chunk_ms=150_000,
            )

    def test_preflight_requires_asr_timing_for_long_audio(self):
        with self.assertRaisesRegex(ValueError, "asr-timed JSON is required"):
            validate_alignment_inputs(
                transcript="这是完整口播",
                cues=["这是完整口播"],
                duration_ms=633_641,
                asr_timed=None,
                max_chunk_ms=150_000,
            )


if __name__ == "__main__":
    unittest.main()
