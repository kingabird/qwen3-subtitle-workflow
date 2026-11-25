#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


TIME_RE = re.compile(r"^\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d$")
PUNCT_RE = re.compile(
    r"[\u3000-\u303F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\uFFE0-\uFFEE"
    r"，。！？；：、“”‘’（）《》〈〉【】『』「」·…—,.!?;:\"'()\[\]{}<>/\\|~`@#$%^&*_+=\-]"
)
AWKWARD_ENDINGS = (
    "把她",
    "把他",
    "把它",
    "把自己",
    "把自己的",
    "被人们形容",
    "被形容",
    "被称为",
    "形容",
    "不只",
)
LATIN_RE = re.compile(r"[A-Za-z]")
LOWER_LATIN_RE = re.compile(r"[a-z]")


def visible_text(value: str) -> str:
    return re.sub(r"\s+", "", remove_allowed_internal_symbols(value))


def remove_allowed_internal_symbols(value: str) -> str:
    value = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z])·(?=[\u4e00-\u9fffA-Za-z])", "", value)
    value = re.sub(r"(?<=[A-Za-z])[-'’](?=[A-Za-z])", "", value)
    value = re.sub(r"(?<=\d)[./:-](?=\d)", "", value)
    value = re.sub(r"(?<=\d)%", "", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    path = args.input_srt
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    ok = True
    errors = []
    warnings = []
    min_chars = 10**9
    max_chars = -1

    cue_texts = []
    for fallback_index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3 or not TIME_RE.match(lines[1]):
            continue

        cue_id = lines[0] if lines[0].isdigit() else str(fallback_index)
        text_lines = lines[2:]
        cue_texts.append((cue_id, "".join(text_lines).strip()))
        for text_line in text_lines:
            compact = visible_text(text_line)
            char_count = len(compact)
            min_chars = min(min_chars, char_count)
            max_chars = max(max_chars, char_count)

            disallowed_text = remove_allowed_internal_symbols(text_line)
            if PUNCT_RE.search(disallowed_text):
                message = f"punctuation block {cue_id}: {text_line}"
                errors.append(message)
                print(message, file=sys.stderr)
                ok = False
            if char_count < 2 or char_count > 18:
                message = f"length block {cue_id}: {char_count} chars: {text_line}"
                errors.append(message)
                print(message, file=sys.stderr)
                ok = False
            elif char_count >= 17:
                message = f"review long block {cue_id}: {char_count} chars: {text_line}"
                warnings.append(message)
                print(message)
            elif char_count == 2:
                message = f"review short block {cue_id}: {text_line}"
                warnings.append(message)
                print(message)
            if compact.endswith(AWKWARD_ENDINGS):
                message = f"awkward break block {cue_id}: {text_line}"
                errors.append(message)
                print(message, file=sys.stderr)
                ok = False

    for (left_id, left), (right_id, right) in zip(cue_texts, cue_texts[1:]):
        left_compact = re.sub(r"\s+", "", left)
        right_compact = re.sub(r"\s+", "", right)
        if (
            left_compact
            and right_compact
            and LATIN_RE.fullmatch(left_compact[-1])
            and LOWER_LATIN_RE.fullmatch(right_compact[0])
        ):
            message = f"split latin term blocks {left_id}/{right_id}: {left} / {right}"
            errors.append(message)
            print(message, file=sys.stderr)
            ok = False

    if not blocks or min_chars == 10**9:
        min_chars = 0
        max_chars = 0

    report = {
        "valid": ok,
        "block_count": len(blocks),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "errors": errors,
        "warnings": warnings,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"blocks={len(blocks)}, style_valid={str(ok).lower()}, "
        f"min_chars={min_chars}, max_chars={max_chars}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
