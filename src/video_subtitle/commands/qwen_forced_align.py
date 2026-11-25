#!/usr/bin/env python3
"""Export Qwen3 Forced Aligner token timestamps as JSON."""

from __future__ import annotations

import argparse
import os
import json
import sys
from collections import Counter
from pathlib import Path

from alignment_utils import MAX_QWEN_ALIGNMENT_MS, compact, plan_alignment_chunks, restore_allowed_symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cues", type=Path)
    parser.add_argument("--asr-timed", type=Path)
    parser.add_argument("--max-chunk-seconds", type=int, default=150)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--model", default=os.environ.get("VIDEO_SUBTITLE_ALIGN_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B"))
    parser.add_argument("--device", default="mps:0")
    args = parser.parse_args()

    text = compact(args.transcript.read_text(encoding="utf-8"))
    if not text:
        raise SystemExit("transcript is empty")

    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ForcedAligner

    audio, sample_rate = sf.read(str(args.audio), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    duration_ms = round(len(audio) / sample_rate * 1000)
    max_chunk_ms = args.max_chunk_seconds * 1000
    if max_chunk_ms <= 0 or max_chunk_ms > MAX_QWEN_ALIGNMENT_MS:
        raise SystemExit("--max-chunk-seconds must be between 1 and 180")

    if duration_ms > max_chunk_ms:
        if args.cues is None or args.asr_timed is None:
            raise SystemExit("long-audio alignment requires --cues and --asr-timed")
        cues = [line.strip() for line in args.cues.read_text(encoding="utf-8").splitlines() if line.strip()]
        timed = json.loads(args.asr_timed.read_text(encoding="utf-8"))
        chunks = plan_alignment_chunks(
            segments=timed.get("segments", []), transcript=text, cues=cues,
            target_chunk_ms=min(120_000, max_chunk_ms), max_chunk_ms=max_chunk_ms,
        )
    else:
        chunks = [{"index": 1, "start_ms": 0, "end_ms": duration_ms,
                   "cue_start": 0, "cue_end": None, "text": text}]

    dtype = torch.bfloat16 if args.device.startswith("mps") else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(
        args.model, device_map=args.device, dtype=dtype,
    )
    items: list[dict[str, object]] = []
    restored_symbols: Counter[str] = Counter()
    chunk_diagnostics = []
    for chunk in chunks:
        start_ms, end_ms = int(chunk["start_ms"]), int(chunk["end_ms"])
        start_sample = round(start_ms * sample_rate / 1000)
        end_sample = round(end_ms * sample_rate / 1000)
        chunk_text = str(chunk["text"])
        result = aligner.align(
            audio=(audio[start_sample:end_sample], sample_rate),
            text=chunk_text,
            language=args.language,
        )[0]
        raw_items = [
            {"text": item.text, "start": item.start_time, "end": item.end_time}
            for item in result.items
        ]
        restored, symbols = restore_allowed_symbols(raw_items, chunk_text)
        restored_symbols.update(symbols)
        offset = start_ms / 1000
        first_index = len(items) + 1
        for item in restored:
            item["start"] = round(float(item["start"]) + offset, 3)
            item["end"] = round(float(item["end"]) + offset, 3)
            items.append(item)
        invalid_count = sum(float(item["end"]) <= float(item["start"]) for item in restored)
        chunk_diagnostics.append({
            "index": chunk["index"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "cue_start": None if chunk["cue_start"] is None else int(chunk["cue_start"]) + 1,
            "cue_end": chunk["cue_end"],
            "token_start": first_index,
            "token_end": len(items),
            "token_count": len(restored),
            "invalid_time_item_count": invalid_count,
        })

    if not items:
        raise SystemExit("Qwen returned no aligned tokens")

    compact_aligned = compact("".join(str(item["text"]) for item in items))
    text_matches = text == compact_aligned
    invalid_times = [
        index for index, item in enumerate(items, start=1)
        if float(item["end"]) <= float(item["start"])
    ]
    payload = {
        "audio": str(args.audio.resolve()),
        "transcript": str(args.transcript.resolve()),
        "language": args.language,
        "model": args.model,
        "diagnostics": {
            "text_matches_ignoring_whitespace": text_matches,
            "invalid_time_items": invalid_times,
            "invalid_time_item_count": len(invalid_times),
            "chunking_required": len(chunks) > 1,
            "chunk_count": len(chunks),
            "chunks": chunk_diagnostics,
            "restored_allowed_symbols": dict(restored_symbols),
        },
        "items": items,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"aligned tokens={len(items)}, chunks={len(chunks)}, text_matches={str(text_matches).lower()}, "
        f"invalid_times={len(invalid_times)}, output={args.output}"
    )
    if invalid_times:
        print(
            "warning: some token timestamps have zero or negative duration; "
            "validate cue-level timings after segmentation",
            file=sys.stderr,
        )
    if not text_matches:
        print("aligned text does not match the transcript", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
