#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


TIME_RE = re.compile(r"^\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d$")


def to_ms(value: str) -> int:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def audio_duration_ms(path: Path) -> Optional[int]:
    if not path:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return round(float(result.stdout.strip()) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    srt_path = args.input_srt
    audio_ms = audio_duration_ms(args.audio) if args.audio else None
    blocks = re.split(r"\n\s*\n", srt_path.read_text(encoding="utf-8").strip())

    ok = True
    errors = []
    warnings = []
    prev_end = -1
    min_ms = 10**12
    max_ms = -1
    final_ms = -1

    for expected, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            message = f"bad block {expected}: missing text or time"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
            continue
        if lines[0] != str(expected):
            message = f"bad block {expected}: got index {lines[0]!r}"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
        if not TIME_RE.match(lines[1]):
            message = f"bad block {expected}: bad time {lines[1]!r}"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
            continue

        start_s, end_s = lines[1].split(" --> ")
        start = to_ms(start_s)
        end = to_ms(end_s)
        duration = end - start
        min_ms = min(min_ms, duration)
        max_ms = max(max_ms, duration)
        final_ms = end

        if end <= start:
            message = f"bad timing block {expected}: end <= start"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
        if start < prev_end:
            message = f"bad timing block {expected}: overlap or time reversal"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
        if duration < 250:
            message = f"short block {expected}: {duration}ms"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False
        elif duration < 600:
            message = f"review short timing block {expected}: {duration}ms"
            warnings.append(message)
            print(message)
        elif duration > 4000:
            message = f"review long timing block {expected}: {duration}ms"
            warnings.append(message)
            print(message)
        if audio_ms is not None and end > audio_ms + 5:
            message = f"block {expected} exceeds audio duration"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False

        prev_end = end

    report = {
        "valid": ok,
        "block_count": len(blocks),
        "min_duration_ms": min_ms,
        "max_duration_ms": max_ms,
        "final_ms": final_ms,
        "audio_ms": audio_ms,
        "errors": errors,
        "warnings": warnings,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"blocks={len(blocks)}, valid={str(ok).lower()}, "
        f"min_ms={min_ms}, max_ms={max_ms}, final_ms={final_ms}"
        + (f", audio_ms={audio_ms}" if audio_ms is not None else "")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
