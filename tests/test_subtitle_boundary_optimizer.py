import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src/video_subtitle/commands/align_srt_to_audio.py"
SPEC = importlib.util.spec_from_file_location("subtitle_boundary_optimizer", SCRIPT)
assert SPEC and SPEC.loader
optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer)


class SubtitleBoundaryOptimizerTests(unittest.TestCase):
    def test_score_prefers_a_meaningful_pause_over_a_slightly_closer_short_gap(self):
        anchor = 1_000
        short_gap = optimizer.score_boundary_candidate(anchor, 980, 1_100)
        meaningful_pause = optimizer.score_boundary_candidate(anchor, 900, 1_260)

        self.assertLess(meaningful_pause, short_gap)

    def test_pause_aware_timing_leaves_more_space_for_a_longer_silence(self):
        short_end, short_start = optimizer.boundary_timing(1_000, 1_120)
        long_end, long_start = optimizer.boundary_timing(1_000, 1_500)

        self.assertGreater(long_start - long_end, short_start - short_end)

    def test_alignment_skips_silence_that_crosses_an_aligned_token_boundary(self):
        cues = [
            {"idx": 1, "start": 0, "end": 1_000, "text": ["前句"]},
            {"idx": 2, "start": 1_200, "end": 2_000, "text": ["后句"]},
        ]

        adjusted, diagnostics = optimizer.align_with_diagnostics(
            cues, [[900, 1_300]], duration_ms=2_000
        )

        self.assertEqual(adjusted, cues)
        self.assertEqual(diagnostics["skipped_token_boundary_candidates"], 1)

    def test_silence_between_cues_cannot_move_subtitles_across_a_long_pause(self):
        cues = [
            {"idx": 1, "start": 64_960, "end": 66_000, "text": ["前句"]},
            {"idx": 2, "start": 69_600, "end": 71_440, "text": ["后句"]},
        ]

        adjusted, diagnostics = optimizer.align_with_diagnostics(
            cues, [[67_686, 68_010]], duration_ms=71_440,
        )

        self.assertEqual(adjusted[0]["end"], 66_250)
        self.assertEqual(adjusted[1]["start"], 69_350)
        self.assertEqual(diagnostics["max_start_shift_ms"], 250)
