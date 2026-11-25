"""Reusable local subtitle pipeline backed by Qwen Forced Aligner."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable, Mapping, Sequence, TextIO


CommandRunner = Callable[[list[str]], None]
TranscriptionPayload = str | Mapping[str, object]
AudioTranscriber = Callable[[Path, Path], TranscriptionPayload]
AudioBatchTranscriber = Callable[[Sequence[Path], Path], list[TranscriptionPayload]]
COMMANDS = Path(__file__).with_name("commands")


@dataclass(frozen=True)
class SubtitleResult:
    tokens_path: Path
    initial_srt_path: Path
    output_path: Path
    validation_report_path: Path


class SubtitlePipeline:
    def __init__(self, runtime_root: Path | None = None, runner: CommandRunner | None = None,
                 transcribe: AudioTranscriber | None = None, transcribe_many: AudioBatchTranscriber | None = None) -> None:
        self.runtime_root = runtime_root or Path(sys.prefix).parent
        self.runner = runner or _run
        self.transcribe = transcribe or _transcribe_mlx
        self.transcribe_many = transcribe_many or _transcribe_mlx_many

    def transcribe_audio(self, audio: Path, output: Path, mlx_python: Path | None = None,
                         chunk_seconds: int = 120, keep_intermediates: bool = False) -> Path:
        audio, output = audio.resolve(), output.resolve()
        if chunk_seconds < 0:
            raise ValueError("chunk_seconds must be non-negative")
        timed_output = output.with_name(f"{output.stem}-timed.json")
        if audio in {output, timed_output.resolve()}:
            raise ValueError("transcription output must differ from the source audio")
        python = mlx_python or self.runtime_root / ".venv-mlx/bin/python"
        duration_ms = _audio_duration_ms(audio)
        silences = _detect_silences(audio) if chunk_seconds and duration_ms > chunk_seconds * 1_000 else []
        chunks = plan_audio_chunks(duration_ms, silences, chunk_seconds * 1_000) if chunk_seconds and duration_ms > chunk_seconds * 1_000 else [(0, duration_ms)]
        output.parent.mkdir(parents=True, exist_ok=True)
        chunk_paths = [audio]
        if len(chunks) > 1:
            directory = Path(mkdtemp(prefix=f"{output.stem}-chunks-", dir=output.parent))
            try:
                chunk_paths = _render_chunks(audio, directory, chunks, self.runner)
                results = self.transcribe_many(chunk_paths, python)
            finally:
                if not keep_intermediates:
                    shutil.rmtree(directory)
        else:
            results = [self.transcribe(audio, python)]
        if len(results) != len(chunks):
            raise ValueError("MLX Qwen ASR returned an empty or incomplete transcription")
        transcriptions = [_normalise_transcription(result, start_ms, end_ms)
                          for result, (start_ms, end_ms) in zip(results, chunks)]
        if any(not item[0] for item in transcriptions):
            raise ValueError("MLX Qwen ASR returned an empty or incomplete transcription")
        text = "\n\n".join(item[0] for item in transcriptions)
        _backup_existing(output)
        output.write_text(f"{text}\n", encoding="utf-8")
        _backup_existing(timed_output)
        timed_segments = [segment for _text, segments in transcriptions for segment in segments]
        timed_output.write_text(json.dumps({
            "audio": str(audio),
            "transcript": str(output),
            "timing_kind": "asr_review_approximate",
            "note": "Review-location timestamps from MLX Qwen ASR; do not use as final subtitle timing.",
            "text": text,
            "segments": [dict(segment, index=index) for index, segment in enumerate(timed_segments, start=1)],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if len(chunks) > 1 and keep_intermediates:
            manifest = output.with_name(f"{output.stem}-chunks.json")
            _backup_existing(manifest)
            manifest.write_text(json.dumps({"audio": str(audio), "chunk_seconds": chunk_seconds, "chunks": [
                {"index": index + 1, "start_ms": start, "end_ms": end, "audio": str(path), "text": chunk_text.strip()}
                for index, ((start, end), path, (chunk_text, _segments))
                in enumerate(zip(chunks, chunk_paths, transcriptions))
            ]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    def align(self, *, audio: Path, transcript: Path, cues: Path, output: Path,
              asr_python: Path | None = None, delete_asr_draft: Path | None = None,
              asr_timed: Path | None = None, max_align_chunk_seconds: int = 150) -> SubtitleResult:
        audio, transcript, cues, output = (path.resolve() for path in (audio, transcript, cues, output))
        asr_timed = asr_timed.resolve() if asr_timed is not None else None
        if max_align_chunk_seconds <= 0 or max_align_chunk_seconds > 180:
            raise ValueError("max_align_chunk_seconds must be between 1 and 180")
        if output.suffix != ".srt":
            raise ValueError("output must end in .srt")
        stem = output.with_suffix("")
        tokens = stem.with_name(f"{stem.name}-tokens.json")
        initial = stem.with_name(f"{stem.name}.initial.srt")
        pending = stem.with_name(f"{stem.name}.pending.srt")
        validation_report = stem.with_name(f"{stem.name}.validation.json")
        adjustment_report = stem.with_name(f".{stem.name}.adjustment.json")
        timing_report = stem.with_name(f".{stem.name}.timing-validation.json")
        style_report = stem.with_name(f".{stem.name}.style-validation.json")
        preflight_report = stem.with_name(f".{stem.name}.preflight.json")
        mapping_report = stem.with_name(f".{stem.name}.mapping.json")
        aligner_python = asr_python or self.runtime_root / ".venv-asr/bin/python"
        stage_reports = [preflight_report, mapping_report, adjustment_report, timing_report, style_report]
        protected = {audio, transcript, cues, asr_timed}
        generated = {path.resolve() for path in (output, tokens, initial, pending, validation_report, *stage_reports)}
        if protected & generated:
            raise ValueError("subtitle outputs must differ from the input files")
        draft = delete_asr_draft.absolute() if delete_asr_draft else None
        if draft is not None and draft.resolve() in protected | generated:
            raise ValueError("--delete-asr-draft must name a separate ASR draft")
        output.parent.mkdir(parents=True, exist_ok=True)
        pending.unlink(missing_ok=True)
        for path in stage_reports:
            path.unlink(missing_ok=True)
        report: dict[str, object] = {
            "status": "running",
            "published": False,
            "audio": str(audio),
            "transcript": str(transcript),
            "cues": str(cues),
            "output": str(output),
            "pending_output": str(pending),
            "tokens": str(tokens),
            "initial_srt": str(initial),
            "asr_timed": str(asr_timed) if asr_timed else None,
            "max_align_chunk_seconds": max_align_chunk_seconds,
            "stages": [],
        }
        stages = report["stages"]
        assert isinstance(stages, list)

        def run_stage(name: str, command: list[str], details_path: Path | None = None,
                      details_factory: Callable[[], Mapping[str, object]] | None = None) -> None:
            stage: dict[str, object] = {"name": name, "status": "running"}
            stages.append(stage)
            try:
                self.runner(command)
                if details_path is not None and details_path.exists():
                    stage["details"] = json.loads(details_path.read_text(encoding="utf-8"))
                elif details_factory is not None:
                    stage["details"] = dict(details_factory())
            except Exception:
                stage["status"] = "failed"
                if details_path is not None and details_path.exists():
                    stage["details"] = json.loads(details_path.read_text(encoding="utf-8"))
                elif details_factory is not None:
                    stage["details"] = dict(details_factory())
                raise
            stage["status"] = "passed"

        def token_details() -> Mapping[str, object]:
            if not tokens.exists():
                return {}
            payload = json.loads(tokens.read_text(encoding="utf-8"))
            items = payload.get("items", [])
            diagnostics = payload.get("diagnostics", {})
            return {
                "token_count": len(items) if isinstance(items, list) else 0,
                "diagnostics": diagnostics if isinstance(diagnostics, Mapping) else {},
            }

        try:
            preflight_command = [
                sys.executable, str(COMMANDS / "preflight_alignment.py"), str(audio), str(transcript), str(cues),
                "--max-chunk-seconds", str(max_align_chunk_seconds),
            ]
            alignment_command = [
                str(aligner_python), str(COMMANDS / "qwen_forced_align.py"), str(audio), str(transcript), str(tokens),
                "--cues", str(cues), "--max-chunk-seconds", str(max_align_chunk_seconds),
            ]
            if asr_timed is not None:
                preflight_command.extend(["--asr-timed", str(asr_timed)])
                alignment_command.extend(["--asr-timed", str(asr_timed)])
            preflight_command.extend(["--report", str(preflight_report)])
            run_stage("alignment_preflight", preflight_command, details_path=preflight_report)
            run_stage(
                "qwen_forced_alignment",
                alignment_command,
                details_factory=token_details,
            )
            run_stage(
                "cue_token_mapping",
                [sys.executable, str(COMMANDS / "tokens_to_srt.py"), str(cues), str(initial),
                 "--tokens", f"{tokens}:0", "--report", str(mapping_report)],
                details_path=mapping_report,
            )
            run_stage(
                "silence_adjustment",
                [sys.executable, str(COMMANDS / "align_srt_to_audio.py"), str(initial), str(audio), str(pending),
                 "--report", str(adjustment_report)],
                details_path=adjustment_report,
            )
            run_stage(
                "timing_validation",
                [sys.executable, str(COMMANDS / "validate_srt.py"), str(pending), str(audio),
                 "--report", str(timing_report)],
                details_path=timing_report,
            )
            run_stage(
                "style_validation",
                [sys.executable, str(COMMANDS / "validate_subtitle_style.py"), str(pending),
                 "--report", str(style_report)],
                details_path=style_report,
            )
            if not pending.exists():
                raise ValueError(f"validated pending SRT was not produced: {pending}")
            _backup_existing(output)
            pending.replace(output)
            report["published"] = True
            report["status"] = "passed"
            if draft is not None:
                draft.unlink(missing_ok=True)
        except Exception as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            _write_json(validation_report, report)
            raise
        else:
            _write_json(validation_report, report)
        finally:
            for path in stage_reports:
                path.unlink(missing_ok=True)

        return SubtitleResult(tokens, initial, output, validation_report)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _transcribe_mlx(audio: Path, mlx_python: Path) -> TranscriptionPayload:
    return _transcribe_mlx_many([audio], mlx_python)[0]


def _transcribe_mlx_many(audios: Sequence[Path], mlx_python: Path) -> list[TranscriptionPayload]:
    command = [str(mlx_python), str(COMMANDS / "mlx_asr_worker.py")]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", bufsize=1) as worker:
        assert worker.stdin is not None and worker.stdout is not None
        try:
            if _read_worker_message(worker.stdout, "startup") != {"ready": True}:
                raise RuntimeError("MLX ASR worker returned an invalid startup response")
            results = []
            for audio in audios:
                worker.stdin.write(json.dumps({"audio": str(audio)}, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                results.append(_read_worker_message(worker.stdout, "transcription"))
            return results
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()


def _read_worker_message(stream: TextIO, phase: str) -> dict[str, object]:
    diagnostics: deque[str] = deque(maxlen=5)
    for line in stream:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if line.strip():
                diagnostics.append(line.strip())
            continue
        if isinstance(payload, dict):
            if "error" in payload:
                raise RuntimeError(f"MLX ASR failed during {phase}: {payload['error']}")
            if "ready" in payload or "text" in payload:
                return payload
    detail = f": {' | '.join(diagnostics)}" if diagnostics else ""
    raise RuntimeError(f"MLX ASR worker stopped during {phase}{detail}")


def _normalise_transcription(result: TranscriptionPayload, offset_ms: int, chunk_end_ms: int) -> tuple[str, list[dict[str, object]]]:
    if isinstance(result, str):
        text = result.strip()
        raw_segments: object = []
    else:
        text = str(result.get("text", "")).strip()
        raw_segments = result.get("segments", [])

    segments: list[dict[str, object]] = []
    if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes)):
        for raw in raw_segments:
            if not isinstance(raw, Mapping):
                continue
            segment_text = str(raw.get("text", "")).strip()
            try:
                start_ms = offset_ms + round(float(raw["start"]) * 1_000)
                end_ms = offset_ms + round(float(raw["end"]) * 1_000)
            except (KeyError, TypeError, ValueError):
                continue
            if segment_text and end_ms > start_ms:
                segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": segment_text})
    if not segments and text:
        segments.append({"start_ms": offset_ms, "end_ms": max(offset_ms + 1, chunk_end_ms), "text": text})
    return text, segments


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.bak")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak.{suffix}")
        suffix += 1
    shutil.copy2(path, backup)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def plan_audio_chunks(duration_ms: int, silences: Sequence[tuple[int, int]], target_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0 or target_ms <= 0:
        raise ValueError("duration_ms and target_ms must be positive")
    chunks: list[tuple[int, int]] = []
    start = 0
    while duration_ms - start > target_ms:
        desired = start + target_ms
        candidates = [end for _silence_start, end in silences
                      if start + target_ms // 2 <= end <= start + int(target_ms * 1.25) and end < duration_ms]
        end = min(candidates, key=lambda value: abs(value - desired)) if candidates else desired
        chunks.append((start, end))
        start = end
    chunks.append((start, duration_ms))
    return chunks


def _audio_duration_ms(audio: Path) -> int:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
                            check=True, capture_output=True, text=True)
    return round(float(result.stdout.strip()) * 1_000)


def _detect_silences(audio: Path) -> list[tuple[int, int]]:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio),
                             "-af", "silencedetect=noise=-40dB:d=0.25", "-f", "null", "-"],
                            check=True, capture_output=True, text=True)
    start: float | None = None
    silences: list[tuple[int, int]] = []
    for line in result.stderr.splitlines():
        if match := re.search(r"silence_start: ([0-9.]+)", line):
            start = float(match.group(1))
        elif match := re.search(r"silence_end: ([0-9.]+)", line):
            if start is not None:
                silences.append((round(start * 1_000), round(float(match.group(1)) * 1_000)))
                start = None
    return silences


def _render_chunks(audio: Path, directory: Path, chunks: Sequence[tuple[int, int]], runner: CommandRunner) -> list[Path]:
    paths = []
    for index, (start, end) in enumerate(chunks, start=1):
        path = directory / f"chunk-{index:03d}.wav"
        runner(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start / 1000:.3f}", "-t", f"{(end - start) / 1000:.3f}",
                "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", str(path)])
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build validated Chinese SRT with local Qwen Forced Aligner")
    subcommands = parser.add_subparsers(dest="command", required=True)
    align = subcommands.add_parser("align", help="Align a proofread, audio-verified transcript, then create and validate an SRT")
    align.add_argument("audio", type=Path)
    align.add_argument("transcript", type=Path, help="Spoken text without Markdown or annotations")
    align.add_argument("cues", type=Path, help="One natural subtitle cue per line; must match the transcript")
    align.add_argument("output", type=Path)
    align.add_argument("--asr-python", type=Path, help="Override the Qwen ASR/aligner Python environment")
    align.add_argument("--asr-timed", type=Path,
                       help="ASR review timing JSON used only to plan safe long-audio alignment chunks")
    align.add_argument("--max-align-chunk-seconds", type=int, default=150,
                       help="Maximum Qwen forced-alignment chunk duration; must not exceed 180 seconds")
    align.add_argument("--delete-asr-draft", type=Path,
                       help="Delete this ASR draft only after successful alignment and both validations")
    transcribe = subcommands.add_parser("transcribe", help="Create an audio-first draft for Codex review; source manuscript is optional")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("output", type=Path, help="Draft transcript to review before alignment")
    transcribe.add_argument("--mlx-python", type=Path, help="Override the MLX Qwen ASR Python environment")
    transcribe.add_argument("--chunk-seconds", type=int, default=120,
                            help="Target duration for silence-aware long-audio chunks; 0 disables splitting")
    transcribe.add_argument("--keep-intermediates", action="store_true",
                            help="Keep extracted chunk WAVs and the chunk manifest after success")
    args = parser.parse_args(argv)
    try:
        pipeline = SubtitlePipeline()
        if args.command == "transcribe":
            print(pipeline.transcribe_audio(args.audio, args.output, mlx_python=args.mlx_python,
                                            chunk_seconds=args.chunk_seconds,
                                            keep_intermediates=args.keep_intermediates))
            return 0
        asr_timed = args.asr_timed
        if asr_timed is None and args.delete_asr_draft is not None:
            inferred = args.delete_asr_draft.with_name(f"{args.delete_asr_draft.stem}-timed.json")
            if inferred.exists():
                asr_timed = inferred
        result = pipeline.align(audio=args.audio, transcript=args.transcript, cues=args.cues,
                                output=args.output, asr_python=args.asr_python,
                                delete_asr_draft=args.delete_asr_draft, asr_timed=asr_timed,
                                max_align_chunk_seconds=args.max_align_chunk_seconds)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
