"""Shared text, symbol, and long-audio planning helpers for Qwen alignment."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

from rapidfuzz.distance import Levenshtein


MAX_QWEN_ALIGNMENT_MS = 180_000


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def aligner_clean(value: str) -> str:
    return "".join(
        character for character in compact(value)
        if character == "'" or unicodedata.category(character).startswith(("L", "N"))
    )


def _is_letter(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("L")


def _is_digit(character: str) -> bool:
    return bool(character) and character.isdigit()


def _allowed_symbol(value: str, index: int) -> bool:
    character = value[index]
    left = value[index - 1] if index else ""
    right = value[index + 1] if index + 1 < len(value) else ""
    if character == "·":
        return _is_letter(left) and _is_letter(right)
    if character in {"'", "’"}:
        return _is_letter(left) and _is_letter(right)
    if character == "-":
        return (_is_letter(left) and _is_letter(right)) or (_is_digit(left) and _is_digit(right))
    if character in {".", "/", ":"}:
        return _is_digit(left) and _is_digit(right)
    if character == "%":
        return _is_digit(left)
    return False


def validate_subtitle_text(value: str, label: str) -> None:
    compact_value = compact(value)
    for index, character in enumerate(compact_value):
        if unicodedata.category(character).startswith(("L", "N")):
            continue
        if not _allowed_symbol(compact_value, index):
            raise ValueError(f"disallowed symbol in {label}: {character!r}")


def visible_length(value: str) -> int:
    compact_value = compact(value)
    return sum(
        1 for index, character in enumerate(compact_value)
        if not _allowed_symbol(compact_value, index)
    )


def validate_alignment_inputs(*, transcript: str, cues: Sequence[str], duration_ms: int,
                              asr_timed: Mapping[str, object] | None, max_chunk_ms: int) -> dict[str, object]:
    transcript_compact = compact(transcript)
    cue_values = [compact(cue) for cue in cues if compact(cue)]
    if not transcript_compact:
        raise ValueError("transcript is empty")
    if not cue_values:
        raise ValueError("cues are empty")
    if transcript_compact != "".join(cue_values):
        raise ValueError("transcript and cues differ after removing whitespace")
    if max_chunk_ms <= 0 or max_chunk_ms > MAX_QWEN_ALIGNMENT_MS:
        raise ValueError(f"max alignment chunk must be between 1 and {MAX_QWEN_ALIGNMENT_MS // 1000} seconds")
    validate_subtitle_text(transcript_compact, "transcript")
    for index, cue in enumerate(cue_values, 1):
        validate_subtitle_text(cue, f"cue {index}")
        length = visible_length(cue)
        if length < 2 or length > 18:
            raise ValueError(f"cue {index} must contain 2-18 visible characters, got {length}")

    chunking_required = duration_ms > max_chunk_ms
    chunks: list[dict[str, object]] = []
    if chunking_required:
        if asr_timed is None:
            raise ValueError("asr-timed JSON is required for long-audio alignment")
        segments = asr_timed.get("segments")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise ValueError("asr-timed JSON has no valid segments")
        chunks = plan_alignment_chunks(
            segments=segments,
            transcript=transcript_compact,
            cues=cue_values,
            target_chunk_ms=min(120_000, max_chunk_ms),
            max_chunk_ms=max_chunk_ms,
        )
    return {
        "valid": True,
        "duration_ms": duration_ms,
        "cue_count": len(cue_values),
        "chunking_required": chunking_required,
        "chunk_count": len(chunks) if chunks else 1,
        "chunks": [
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ],
        "errors": [],
    }


def _map_boundary(opcodes, source_index: int, target_length: int) -> int:
    for tag, source_start, source_end, target_start, target_end in opcodes:
        if source_start <= source_index <= source_end:
            if tag == "equal":
                return target_start + min(source_index - source_start, target_end - target_start)
            if source_end == source_start:
                return target_start
            ratio = (source_index - source_start) / (source_end - source_start)
            return round(target_start + ratio * (target_end - target_start))
    return target_length


def plan_alignment_chunks(*, segments: Sequence[object], transcript: str, cues: Sequence[str],
                          target_chunk_ms: int = 120_000, max_chunk_ms: int = 150_000) -> list[dict[str, object]]:
    if not segments:
        raise ValueError("cannot plan long-audio alignment without ASR segments")
    normalized_segments: list[dict[str, object]] = []
    source_position = 0
    for raw in segments:
        if not isinstance(raw, Mapping):
            raise ValueError("invalid ASR timing segment")
        try:
            start_ms = int(raw["start_ms"])
            end_ms = int(raw["end_ms"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid ASR timing segment") from error
        text = aligner_clean(str(raw.get("text", "")))
        if end_ms <= start_ms or not text:
            raise ValueError("invalid ASR timing segment")
        source_position += len(text)
        normalized_segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "source_end": source_position,
        })

    transcript_compact = compact(transcript)
    target_clean = aligner_clean(transcript_compact)
    source_clean = "".join(str(segment["text"]) for segment in normalized_segments)
    # Minimum-edit alignment preserves positions through repeated phrases with localized ASR errors.
    opcodes = Levenshtein.opcodes(source_clean, target_clean)
    cue_values = [compact(cue) for cue in cues]
    if "".join(cue_values) != transcript_compact:
        raise ValueError("transcript and cues differ after removing whitespace")

    cue_clean_ends: list[int] = []
    running = 0
    for cue in cue_values:
        running += len(aligner_clean(cue))
        cue_clean_ends.append(running)

    duration_ms = int(normalized_segments[-1]["end_ms"])
    boundaries = [0]
    cue_boundaries = [0]
    start_ms = 0
    start_cue = 0
    while duration_ms - start_ms > max_chunk_ms:
        candidates = []
        minimum = start_ms + min(target_chunk_ms * 3 // 4, max_chunk_ms)
        maximum = min(duration_ms, start_ms + max_chunk_ms)
        for segment in normalized_segments:
            end_ms = int(segment["end_ms"])
            if not minimum <= end_ms <= maximum or end_ms >= duration_ms:
                continue
            mapped = _map_boundary(opcodes, int(segment["source_end"]), len(target_clean))
            choices = [
                (abs(cue_clean_ends[index] - mapped), index + 1)
                for index in range(start_cue, len(cue_values) - 1)
            ]
            if not choices:
                continue
            distance, cue_end = min(choices)
            candidates.append((distance, abs(end_ms - (start_ms + target_chunk_ms)), end_ms, cue_end))
        if not candidates:
            raise ValueError(f"no safe ASR/cue boundary found after {start_ms} ms")
        distance, _duration_delta, end_ms, cue_end = min(candidates)
        if distance > 12:
            raise ValueError(f"ASR/cue boundary differs by {distance} characters near {end_ms} ms")
        if cue_end <= start_cue:
            raise ValueError("alignment chunk cue boundaries are not increasing")
        boundaries.append(end_ms)
        cue_boundaries.append(cue_end)
        start_ms = end_ms
        start_cue = cue_end
    boundaries.append(duration_ms)
    cue_boundaries.append(len(cue_values))

    chunks = []
    for index in range(len(boundaries) - 1):
        cue_start, cue_end = cue_boundaries[index], cue_boundaries[index + 1]
        chunks.append({
            "index": index + 1,
            "start_ms": boundaries[index],
            "end_ms": boundaries[index + 1],
            "cue_start": cue_start,
            "cue_end": cue_end,
            "text": "".join(cue_values[cue_start:cue_end]),
        })
    if "".join(str(chunk["text"]) for chunk in chunks) != transcript_compact:
        raise ValueError("planned chunks do not reproduce the transcript")
    return chunks


def restore_allowed_symbols(items: Sequence[Mapping[str, object]], expected: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    expected_compact = compact(expected)
    validate_subtitle_text(expected_compact, "aligned transcript")
    aligned_clean = aligner_clean("".join(str(item.get("text", "")) for item in items))
    if aligned_clean != aligner_clean(expected_compact):
        raise ValueError("Qwen aligned text differs from the transcript beyond allowed symbols")

    restored = [dict(item) for item in items]
    cursor = 0
    for item in restored:
        needed = len(aligner_clean(str(item.get("text", ""))))
        start = cursor
        consumed = 0
        while cursor < len(expected_compact) and consumed < needed:
            if aligner_clean(expected_compact[cursor]):
                consumed += 1
            cursor += 1
        while cursor < len(expected_compact) and not aligner_clean(expected_compact[cursor]):
            cursor += 1
        item["text"] = expected_compact[start:cursor]
    if cursor != len(expected_compact):
        raise ValueError("failed to restore allowed transcript symbols")

    symbols = Counter(
        character for index, character in enumerate(expected_compact)
        if _allowed_symbol(expected_compact, index) and not aligner_clean(character)
    )
    return restored, dict(symbols)
