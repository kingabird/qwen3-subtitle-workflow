#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
work=$(mktemp -d "${TMPDIR:-/tmp}/qwen-subtitle-smoke.XXXXXXXX")
cli="$PWD/.venv/bin/video-subtitle"
python="$PWD/.venv/bin/python"
sample="$PWD/examples/window-mint"
printf '测试结果目录 %s\n' "$work"

ffmpeg -nostdin -v error -i "$sample.mp3" -map 0:a:0 -ac 1 -ar 16000 "$work/short.wav"
ffmpeg -nostdin -v error -stream_loop 6 -i "$work/short.wav" -c:a pcm_s16le "$work/long.wav"
"$python" - "$work" "$sample" <<'PY'
from pathlib import Path
import sys

work, sample = map(Path, sys.argv[1:])
for suffix in ("transcript", "cues"):
    contents = sample.with_suffix(f".{suffix}.txt").read_text(encoding="utf-8")
    (work / f"short.{suffix}.txt").write_text(contents, encoding="utf-8")
    (work / f"long.{suffix}.txt").write_text(contents * 7, encoding="utf-8")
PY

cd -- "$work"
for fixture in short long; do
  "$cli" transcribe "$fixture.wav" "$fixture-asr.txt"
  "$cli" align "$fixture.wav" "$fixture.transcript.txt" "$fixture.cues.txt" \
    "$fixture.srt" --asr-timed "$fixture-asr-timed.json"
done

"$python" - <<'PY'
import json
from pathlib import Path

for fixture, expected_cues in (("short", 11), ("long", 77)):
    report = json.loads(Path(f"{fixture}.validation.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage["details"] for stage in report["stages"]}
    timing = stages["timing_validation"]
    diagnostics = stages["qwen_forced_alignment"]["diagnostics"]
    if not (report["published"] and timing["valid"] and timing["block_count"] == expected_cues
            and diagnostics["text_matches_ignoring_whitespace"]):
        raise SystemExit(f"{fixture}: the SRT or its validation report is incomplete")
    if fixture == "long" and not (timing["audio_ms"] > 150_000 and diagnostics["chunk_count"] > 1):
        raise SystemExit("long: fixtures did not exercise long-audio alignment")
    print(f"{fixture}: {expected_cues} cues, {timing['audio_ms']} ms, {diagnostics['chunk_count']} alignment chunks")
PY
