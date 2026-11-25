import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import call, patch

from video_subtitle.pipeline import COMMANDS, SubtitlePipeline, plan_audio_chunks


class SubtitlePipelineTests(unittest.TestCase):
    @staticmethod
    def successful_runner(commands=None):
        def run(command):
            if commands is not None:
                commands.append(command)
            script = Path(command[1]).name
            if script == "preflight_alignment.py":
                Path(command[command.index("--report") + 1]).write_text(json.dumps({
                    "valid": True,
                    "duration_ms": 1_000,
                    "cue_count": 1,
                    "chunking_required": False,
                    "errors": [],
                }), encoding="utf-8")
            elif script == "qwen_forced_align.py":
                Path(command[4]).write_text(json.dumps({
                    "diagnostics": {"text_matches_ignoring_whitespace": True, "invalid_time_items": []},
                    "items": [{"text": "测试", "start": 0.0, "end": 1.0}],
                }), encoding="utf-8")
            elif script == "tokens_to_srt.py":
                Path(command[3]).write_text("1\n00:00:00,000 --> 00:00:01,000\n测试\n", encoding="utf-8")
                Path(command[command.index("--report") + 1]).write_text(json.dumps({
                    "valid": True,
                    "cue_count": 1,
                    "token_count": 1,
                    "invalid_time_item_count": 0,
                    "invalid_time_items": [],
                    "cues_with_invalid_time_items": [],
                }), encoding="utf-8")
            elif script == "align_srt_to_audio.py":
                Path(command[4]).write_text("1\n00:00:00,000 --> 00:00:01,000\n测试\n", encoding="utf-8")
                Path(command[command.index("--report") + 1]).write_text(json.dumps({
                    "cue_count": 1,
                    "silence_count": 0,
                    "adjusted_cue_count": 0,
                    "skipped_token_boundary_candidates": 0,
                    "max_start_shift_ms": 0,
                    "max_end_shift_ms": 0,
                }), encoding="utf-8")
            elif script in {"validate_srt.py", "validate_subtitle_style.py"}:
                Path(command[command.index("--report") + 1]).write_text(
                    json.dumps({"valid": True, "errors": [], "warnings": []}), encoding="utf-8"
                )

        return run

    def test_align_deletes_the_asr_draft_only_after_successful_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = root / "asr-draft.txt"
            draft.write_text("草稿", encoding="utf-8")
            pipeline = SubtitlePipeline(Path("/qwen"), runner=self.successful_runner())

            pipeline.align(
                audio=root / "audio.wav", transcript=root / "transcript.txt", cues=root / "cues.txt",
                output=root / "final.srt", asr_python=Path("/qwen/.venv-asr/bin/python"), delete_asr_draft=draft,
            )

            self.assertFalse(draft.exists())
            report = json.loads((root / "final.validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["published"])

    def test_chunk_cleanup_preserves_preexisting_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preserved = root / "draft-chunks"
            preserved.mkdir()
            (preserved / "keep.txt").write_text("旧文件", encoding="utf-8")
            output = root / "draft.txt"
            pipeline = SubtitlePipeline(
                runner=lambda command: Path(command[-1]).touch(),
                transcribe_many=lambda audios, _python: ["测试录音"] * len(audios),
            )
            with patch("video_subtitle.pipeline._audio_duration_ms", return_value=241_000), \
                    patch("video_subtitle.pipeline._detect_silences", return_value=[]):
                pipeline.transcribe_audio(root / "audio.wav", output)
            self.assertTrue(output.exists())
            self.assertTrue((preserved / "keep.txt").is_file())
            self.assertEqual(list(root.glob("draft-chunks-*")), [])

    def test_align_rejects_deleting_a_source_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio.wav"
            audio.write_bytes(b"source")
            pipeline = SubtitlePipeline(runner=self.successful_runner())
            with self.assertRaisesRegex(ValueError, "separate ASR draft"):
                pipeline.align(audio=audio, transcript=root / "transcript.txt", cues=root / "cues.txt",
                               output=root / "final.srt", delete_asr_draft=audio)
            self.assertEqual(audio.read_bytes(), b"source")

    def test_chunk_planner_prefers_nearby_natural_silences(self):
        chunks = plan_audio_chunks(
            duration_ms=358_000,
            silences=[(117_000, 117_500), (238_000, 238_400)],
            target_ms=120_000,
        )

        self.assertEqual(chunks, [(0, 117_500), (117_500, 238_400), (238_400, 358_000)])

    def test_chunk_planner_uses_the_target_when_no_safe_silence_exists(self):
        chunks = plan_audio_chunks(duration_ms=250_000, silences=[], target_ms=120_000)

        self.assertEqual(chunks, [(0, 120_000), (120_000, 240_000), (240_000, 250_000)])

    def test_chunk_planner_ignores_a_silence_end_at_the_end_of_the_recording(self):
        chunks = plan_audio_chunks(124_000, [(123_000, 124_000)], 120_000)

        self.assertEqual(chunks, [(0, 120_000), (120_000, 124_000)])

    def test_transcribe_writes_an_audio_first_draft_with_the_configured_mlx_python(self):
        received = {}

        def transcribe(audio, mlx_python):
            received.update(audio=audio, mlx_python=mlx_python)
            return "这是实际口播"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "full-track-asr.txt"
            pipeline = SubtitlePipeline(Path("/qwen"), transcribe=transcribe)

            with patch("video_subtitle.pipeline._audio_duration_ms", return_value=3_500):
                result = pipeline.transcribe_audio(root / "full-track.wav", output, mlx_python=Path("/mlx/bin/python"), chunk_seconds=0)

            self.assertEqual(result, output.resolve())
            self.assertEqual(output.read_text(encoding="utf-8"), "这是实际口播\n")
            self.assertEqual(received, {"audio": (root / "full-track.wav").resolve(), "mlx_python": Path("/mlx/bin/python")})

    def test_transcribe_writes_asr_timing_for_review_without_using_it_as_final_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "full-track-asr.txt"
            pipeline = SubtitlePipeline(
                Path("/qwen"),
                transcribe=lambda _audio, _python: {
                    "text": "这是实际口播",
                    "segments": [{"text": "这是实际口播", "start": 1.25, "end": 3.5}],
                },
            )

            with patch("video_subtitle.pipeline._audio_duration_ms", return_value=3_500):
                pipeline.transcribe_audio(root / "full-track.wav", output, mlx_python=Path("/mlx/bin/python"), chunk_seconds=0)

            timed = root / "full-track-asr-timed.json"
            payload = json.loads(timed.read_text(encoding="utf-8"))
            self.assertEqual(payload["timing_kind"], "asr_review_approximate")
            self.assertEqual(payload["text"], "这是实际口播")
            self.assertEqual(payload["segments"], [{
                "index": 1,
                "start_ms": 1250,
                "end_ms": 3500,
                "text": "这是实际口播",
            }])

    def test_transcribe_keeps_chunk_manifest_text_when_intermediates_are_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "draft.txt"
            chunk_paths = [root / "chunk-001.wav", root / "chunk-002.wav"]
            pipeline = SubtitlePipeline(
                Path("/qwen"),
                transcribe_many=lambda _audios, _python: [
                    {"text": "第一段", "segments": []},
                    {"text": "第二段", "segments": []},
                ],
            )

            with patch("video_subtitle.pipeline._audio_duration_ms", return_value=240_000), \
                    patch("video_subtitle.pipeline._detect_silences", return_value=[]), \
                    patch("video_subtitle.pipeline._render_chunks", return_value=chunk_paths):
                pipeline.transcribe_audio(
                    root / "audio.wav", output, chunk_seconds=120, keep_intermediates=True,
                )

            manifest = json.loads((root / "draft-chunks.json").read_text(encoding="utf-8"))
            self.assertEqual([chunk["text"] for chunk in manifest["chunks"]], ["第一段", "第二段"])

    def test_align_preserves_the_virtual_environment_python_symlink(self):
        commands = []
        project_root = Path("/qwen")
        asr_python = project_root / ".venv-asr/bin/python"
        pipeline = SubtitlePipeline(project_root, runner=self.successful_runner(commands))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline.align(
                audio=root / "full-track.wav", transcript=root / "transcript.txt", cues=root / "cues.txt",
                output=root / "full-track.srt", asr_python=asr_python,
            )

        qwen_command = next(command for command in commands if Path(command[1]).name == "qwen_forced_align.py")
        self.assertEqual(qwen_command[0], str(asr_python))

    def test_align_backs_up_an_existing_srt_before_replacing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "full-track.srt"
            output.write_text("人工校对版", encoding="utf-8")
            pipeline = SubtitlePipeline(Path("/qwen"), runner=self.successful_runner())

            pipeline.align(
                audio=root / "full-track.wav", transcript=root / "transcript.txt", cues=root / "cues.txt",
                output=output, asr_python=Path("/qwen/.venv-asr/bin/python"),
            )

            self.assertEqual((root / "full-track.srt.bak").read_text(encoding="utf-8"), "人工校对版")

    def test_align_runs_qwen_mapping_adjustment_and_validation_in_order(self):
        commands = []
        pipeline = SubtitlePipeline(Path("/qwen"), runner=self.successful_runner(commands))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            audio = (root / "full-track.wav").resolve()
            transcript = (root / "transcript.txt").resolve()
            cues = (root / "cues.txt").resolve()
            output = (root / "full-track.srt").resolve()
            result = pipeline.align(
                audio=audio, transcript=transcript, cues=cues, output=output,
                asr_python=Path("/qwen/.venv-asr/bin/python"),
            )

            tokens = root / "full-track-tokens.json"
            initial = root / "full-track.initial.srt"
            pending = root / "full-track.pending.srt"
            self.assertEqual(result.tokens_path, tokens)
            self.assertEqual(result.initial_srt_path, initial)
            self.assertEqual(result.output_path, output)
            self.assertEqual(result.validation_report_path, root / "full-track.validation.json")
            self.assertEqual(commands, [
                [sys.executable, str(COMMANDS / "preflight_alignment.py"), str(audio), str(transcript), str(cues), "--max-chunk-seconds", "150", "--report", str(root / ".full-track.preflight.json")],
                ["/qwen/.venv-asr/bin/python", str(COMMANDS / "qwen_forced_align.py"), str(audio), str(transcript), str(tokens), "--cues", str(cues), "--max-chunk-seconds", "150"],
                [sys.executable, str(COMMANDS / "tokens_to_srt.py"), str(cues), str(initial), "--tokens", f"{tokens}:0", "--report", str(root / ".full-track.mapping.json")],
                [sys.executable, str(COMMANDS / "align_srt_to_audio.py"), str(initial), str(audio), str(pending), "--report", str(root / ".full-track.adjustment.json")],
                [sys.executable, str(COMMANDS / "validate_srt.py"), str(pending), str(audio), "--report", str(root / ".full-track.timing-validation.json")],
                [sys.executable, str(COMMANDS / "validate_subtitle_style.py"), str(pending), "--report", str(root / ".full-track.style-validation.json")],
            ])

    def test_align_passes_asr_timing_to_preflight_and_qwen(self):
        commands = []
        pipeline = SubtitlePipeline(Path("/qwen"), runner=self.successful_runner(commands))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            timed = root / "full-track-asr-timed.json"
            timed.write_text("{}", encoding="utf-8")
            pipeline.align(
                audio=root / "full-track.wav",
                transcript=root / "full-track-transcript.txt",
                cues=root / "full-track-cues.txt",
                output=root / "full-track.srt",
                asr_python=Path("/qwen/.venv-asr/bin/python"),
                asr_timed=timed,
            )

        self.assertIn("--asr-timed", commands[0])
        self.assertIn(str(timed), commands[0])
        self.assertIn("--asr-timed", commands[1])
        self.assertIn(str(timed), commands[1])

    def test_failed_validation_keeps_the_existing_srt_and_writes_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "final.srt"
            output.write_text("已验证旧字幕", encoding="utf-8")
            draft = root / "asr-draft.txt"
            draft.write_text("草稿", encoding="utf-8")
            successful = self.successful_runner()

            def fail_timing_validation(command):
                if Path(command[1]).name == "validate_srt.py":
                    Path(command[command.index("--report") + 1]).write_text(
                        json.dumps({"valid": False, "errors": ["overlap"], "warnings": []}),
                        encoding="utf-8",
                    )
                    raise subprocess.CalledProcessError(1, command)
                successful(command)

            pipeline = SubtitlePipeline(Path("/qwen"), runner=fail_timing_validation)

            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.align(
                    audio=root / "audio.wav", transcript=root / "transcript.txt", cues=root / "cues.txt",
                    output=output, asr_python=Path("/qwen/.venv-asr/bin/python"), delete_asr_draft=draft,
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "已验证旧字幕")
            self.assertTrue(draft.exists())
            self.assertTrue((root / "final.pending.srt").exists())
            report = json.loads((root / "final.validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["published"])
            self.assertEqual(report["stages"][-1]["name"], "timing_validation")
            self.assertEqual(report["stages"][-1]["details"]["errors"], ["overlap"])

    def test_failed_qwen_alignment_includes_its_diagnostics_in_the_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            successful = self.successful_runner()

            def fail_alignment(command):
                if Path(command[1]).name != "qwen_forced_align.py":
                    successful(command)
                    return
                tokens = Path(command[4])
                tokens.write_text(json.dumps({
                    "diagnostics": {
                        "text_matches_ignoring_whitespace": False,
                        "invalid_time_items": [],
                    },
                    "items": [{"text": "错字", "start": 0.0, "end": 1.0}],
                }), encoding="utf-8")
                raise subprocess.CalledProcessError(1, command)

            pipeline = SubtitlePipeline(Path("/qwen"), runner=fail_alignment)

            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.align(
                    audio=root / "audio.wav", transcript=root / "transcript.txt", cues=root / "cues.txt",
                    output=root / "final.srt", asr_python=Path("/qwen/.venv-asr/bin/python"),
                )

            report = json.loads((root / "final.validation.json").read_text(encoding="utf-8"))
            stage = next(stage for stage in report["stages"] if stage["name"] == "qwen_forced_alignment")
            self.assertEqual(stage["status"], "failed")
            self.assertFalse(stage["details"]["diagnostics"]["text_matches_ignoring_whitespace"])
