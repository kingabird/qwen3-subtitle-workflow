#!/usr/bin/env python3
"""Build an SRT by exactly matching subtitle cues to Qwen token JSON."""

import argparse
import json
import re
from pathlib import Path


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def fmt(milliseconds: float) -> str:
    total = max(0, round(milliseconds))
    seconds, millis = divmod(total, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_token_source(value: str) -> tuple[Path, float]:
    try:
        path, offset = value.rsplit(":", 1)
        return Path(path), float(offset)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use --tokens /path/to/tokens.json:offset_seconds"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cues", type=Path, help="one subtitle cue per non-empty line")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--tokens",
        action="append",
        required=True,
        type=parse_token_source,
        help="Qwen token JSON and its position in the original audio",
    )
    parser.add_argument("--report", type=Path, help="write cue/token mapping diagnostics as JSON")
    args = parser.parse_args()

    cues = [line.strip() for line in args.cues.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cues:
        raise SystemExit("no subtitle cues")

    items = []
    for path, offset in args.tokens:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            value = compact(str(item["text"]))
            if value:
                start = float(item["start"]) * 1000 + offset * 1000
                end = float(item["end"]) * 1000 + offset * 1000
                items.append(
                    {
                        "token_index": len(items) + 1,
                        "text": value,
                        "start": start,
                        "end": end,
                        "invalid_time": end <= start,
                    }
                )

    cursor = 0
    rendered = []
    cues_with_invalid_time_items = []
    previous_end = -1.0
    for cue_number, cue in enumerate(cues, start=1):
        target = compact(cue)
        matched = ""
        first = None
        last = None
        cue_invalid_token_indices = []
        while len(matched) < len(target):
            if cursor >= len(items):
                raise SystemExit(f"cue {cue_number} has no remaining token: {cue}")
            item = items[cursor]
            candidate = matched + item["text"]
            if not target.startswith(candidate):
                raise SystemExit(
                    f"cue {cue_number} does not match token {cursor + 1}: "
                    f"expected prefix {target[len(matched):]!r}, got {item['text']!r}"
                )
            if first is None:
                first = item
            last = item
            if item["invalid_time"]:
                cue_invalid_token_indices.append(item["token_index"])
            matched = candidate
            cursor += 1

        if matched != target:
            raise SystemExit(f"cue {cue_number} cuts through a token: {cue}")
        assert first is not None and last is not None
        if last["end"] <= first["start"]:
            raise SystemExit(f"cue {cue_number} has non-positive duration: {cue}")
        if first["start"] < previous_end:
            raise SystemExit(f"cue {cue_number} overlaps the previous cue: {cue}")
        previous_end = last["end"]
        if cue_invalid_token_indices:
            cues_with_invalid_time_items.append({
                "cue_number": cue_number,
                "text": cue,
                "invalid_token_indices": cue_invalid_token_indices,
            })
        rendered.extend(
            [
                str(cue_number),
                f"{fmt(first['start'])} --> {fmt(last['end'])}",
                cue,
                "",
            ]
        )

    if cursor != len(items):
        remainder = "".join(item["text"] for item in items[cursor:cursor + 20])
        raise SystemExit(f"unmatched trailing tokens beginning with: {remainder!r}")

    args.output.write_text("\n".join(rendered), encoding="utf-8")
    invalid_time_items = [
        {
            "token_index": item["token_index"],
            "text": item["text"],
            "start_ms": item["start"],
            "end_ms": item["end"],
        }
        for item in items if item["invalid_time"]
    ]
    if args.report:
        args.report.write_text(json.dumps({
            "valid": True,
            "cue_count": len(cues),
            "token_count": len(items),
            "invalid_time_item_count": len(invalid_time_items),
            "invalid_time_items": invalid_time_items,
            "cues_with_invalid_time_items": cues_with_invalid_time_items,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"mapped cues={len(cues)}, tokens={len(items)}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
