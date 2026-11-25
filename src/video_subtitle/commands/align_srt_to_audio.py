#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def to_ms(value: str) -> int:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def fmt(ms: float) -> str:
    total = max(0, round(ms))
    seconds, millis = divmod(total, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt(path: Path) -> list[dict]:
    cues = []
    text = path.read_text(encoding="utf-8").strip()
    for block in re.split(r"\n\s*\n", text):
        lines = block.splitlines()
        start_s, end_s = lines[1].split(" --> ")
        cues.append(
            {
                "idx": int(lines[0]),
                "start": to_ms(start_s),
                "end": to_ms(end_s),
                "text": lines[2:],
            }
        )
    return cues


def detect_silences(audio_path: Path) -> tuple[list[list[float]], float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=noise=-40dB:d=0.10",
            "-f",
            "null",
            "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    log = result.stderr
    silences = []
    current = None
    for line in log.splitlines():
        match = re.search(r"silence_start: ([0-9.]+)", line)
        if match:
            current = float(match.group(1))
            continue
        match = re.search(r"silence_end: ([0-9.]+)", line)
        if match and current is not None:
            silences.append([current * 1000, float(match.group(1)) * 1000])
            current = None

    merged = []
    for start, end in silences:
        if merged and start - merged[-1][1] <= 80:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    duration = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
    ).strip()
    return merged, float(duration) * 1000


def score_boundary_candidate(anchor_ms: float, silence_start: float, silence_end: float) -> float:
    """Prefer a nearby, meaningful pause without letting it override token timing."""
    pause_ms = silence_end - silence_start
    midpoint_distance = abs((silence_start + silence_end) / 2 - anchor_ms)
    pause_reward = min(pause_ms, 500) * 0.25
    return midpoint_distance - pause_reward


def boundary_timing(silence_start: float, silence_end: float) -> tuple[float, float]:
    """Place each cue edge inside a silence, leaving more breathing room for long pauses."""
    pause_ms = silence_end - silence_start
    tail = min(150, max(80, pause_ms * 0.5))
    lead = min(90, max(40, pause_ms * 0.25))
    end = min(silence_start + tail, silence_end - 20)
    start = max(silence_end - lead, end + 20)
    return end, start


MAX_SILENCE_EXTENSION_MS = 250


def align_with_diagnostics(
    cues: list[dict], silences: list[list[float]], duration_ms: float
) -> tuple[list[dict], dict]:
    previous_end = -1
    for cue in cues:
        if cue["end"] <= cue["start"]:
            raise ValueError(f"invalid cue {cue['idx']}: end <= start")
        if cue["start"] < previous_end:
            raise ValueError(f"invalid cue {cue['idx']}: overlap or time reversal")
        previous_end = cue["end"]

    adjusted = [dict(cue) for cue in cues]
    lead = 60
    used = set()
    skipped_token_boundary_candidates = 0

    if (
        silences
        and silences[0][0] <= 40
        and 0 <= adjusted[0]["start"] - silences[0][1] <= 1000
    ):
        adjusted[0]["start"] = max(0, silences[0][1] - lead)

    for idx in range(len(adjusted) - 1):
        boundary = (cues[idx]["end"] + cues[idx + 1]["start"]) / 2
        candidates = []
        for silence_idx, (start, end) in enumerate(silences):
            if silence_idx in used or end - start < 110:
                continue
            if end >= boundary - 250 and start <= boundary + 1000:
                score = score_boundary_candidate(boundary, start, end)
                new_end, new_start = boundary_timing(start, end)
                if new_end < cues[idx]["end"] or new_start > cues[idx + 1]["start"]:
                    skipped_token_boundary_candidates += 1
                    continue
                new_end = min(new_end, cues[idx]["end"] + MAX_SILENCE_EXTENSION_MS)
                new_start = max(new_start, cues[idx + 1]["start"] - MAX_SILENCE_EXTENSION_MS)
                candidates.append((score, silence_idx, new_end, new_start))

        if not candidates:
            continue

        score, silence_idx, new_end, new_start = min(candidates)
        if score > 750:
            continue

        if new_end - adjusted[idx]["start"] < 250:
            continue
        if adjusted[idx + 1]["end"] - new_start < 250:
            continue

        used.add(silence_idx)
        adjusted[idx]["end"] = new_end
        adjusted[idx + 1]["start"] = new_start

    if 40 < duration_ms - adjusted[-1]["end"] < 500:
        adjusted[-1]["end"] = duration_ms

    for idx in range(len(adjusted) - 1):
        if adjusted[idx]["end"] >= adjusted[idx + 1]["start"]:
            middle = (adjusted[idx]["end"] + adjusted[idx + 1]["start"]) / 2
            adjusted[idx]["end"] = middle - 10
            adjusted[idx + 1]["start"] = middle + 10

    adjusted_boundaries = 0
    max_start_shift_ms = 0.0
    max_end_shift_ms = 0.0
    for original, result in zip(cues, adjusted):
        start_shift = abs(result["start"] - original["start"])
        end_shift = abs(result["end"] - original["end"])
        if start_shift or end_shift:
            adjusted_boundaries += 1
        max_start_shift_ms = max(max_start_shift_ms, start_shift)
        max_end_shift_ms = max(max_end_shift_ms, end_shift)

    return adjusted, {
        "cue_count": len(adjusted),
        "silence_count": len(silences),
        "adjusted_cue_count": adjusted_boundaries,
        "skipped_token_boundary_candidates": skipped_token_boundary_candidates,
        "max_start_shift_ms": round(max_start_shift_ms, 3),
        "max_end_shift_ms": round(max_end_shift_ms, 3),
    }


def align(cues: list[dict], silences: list[list[float]], duration_ms: float) -> list[dict]:
    adjusted, _diagnostics = align_with_diagnostics(cues, silences, duration_ms)
    return adjusted


def write_srt(cues: list[dict], path: Path) -> None:
    parts = []
    for idx, cue in enumerate(cues, start=1):
        parts.append(str(idx))
        parts.append(f"{fmt(cue['start'])} --> {fmt(cue['end'])}")
        parts.extend(cue["text"])
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cues = parse_srt(args.input_srt)
    silences, duration_ms = detect_silences(args.audio)
    adjusted, diagnostics = align_with_diagnostics(cues, silences, duration_ms)
    write_srt(adjusted, args.output)
    if args.report:
        args.report.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"aligned cues={len(adjusted)}, silences={len(silences)}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
