#!/usr/bin/env python3
"""Validate subtitle text and long-audio inputs before loading Qwen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from alignment_utils import validate_alignment_inputs


def audio_duration_ms(audio: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        check=True, capture_output=True, text=True,
    )
    return round(float(result.stdout.strip()) * 1000)


def write_report(path: Path | None, payload: dict[str, object]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("cues", type=Path)
    parser.add_argument("--asr-timed", type=Path)
    parser.add_argument("--max-chunk-seconds", type=int, default=150)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        timed = json.loads(args.asr_timed.read_text(encoding="utf-8")) if args.asr_timed else None
        payload = validate_alignment_inputs(
            transcript=args.transcript.read_text(encoding="utf-8"),
            cues=args.cues.read_text(encoding="utf-8").splitlines(),
            duration_ms=audio_duration_ms(args.audio),
            asr_timed=timed,
            max_chunk_ms=args.max_chunk_seconds * 1000,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        payload = {"valid": False, "errors": [str(error)]}
        write_report(args.report, payload)
        print(str(error), file=sys.stderr)
        return 1
    write_report(args.report, payload)
    print(
        f"preflight valid=true, cues={payload['cue_count']}, "
        f"chunks={payload['chunk_count']}, duration_ms={payload['duration_ms']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
