"""Serve MLX Qwen3 ASR requests using the model environment's Python."""

import json
import os
import sys


DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-6bit"


def transcribe(model, audio: str) -> dict:
    result = model.generate(audio, language="Chinese", chunk_duration=20.0)
    text = str(result.text).strip()
    if not text:
        raise ValueError("MLX-Audio returned empty transcription")
    segments = []
    for segment in getattr(result, "segments", []) or []:
        if not isinstance(segment, dict):
            continue
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        segment_text = str(segment.get("text", "")).strip()
        if segment_text and end > start:
            segments.append({"text": segment_text, "start": start, "end": end})
    return {"text": text, "segments": segments}


def main() -> None:
    from mlx_audio.stt import load

    model = load(os.environ.get("VIDEO_SUBTITLE_ASR_MODEL", DEFAULT_MODEL))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            response = transcribe(model, json.loads(line)["audio"])
        except Exception as error:
            response = {"error": str(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
