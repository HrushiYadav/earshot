#!/usr/bin/env python3
"""Build a ~40 s demo WAV for the earshot live-loop video by
concatenating selected clips in order. Skips music clips (copyrighted
songs). Inserts a short silence between segments so the dashboard's
reaction to each clip has time to settle.

Order: silence, calm_speech, clap, knock, stop, alarm, loud.

Usage:
    uv run python scripts/make_demo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO_ROOT / "clips"
OUT_DIR = REPO_ROOT / "private" / "demo"
SR = 16000  # all clips in this repo are 16 kHz mono

# (clip_id, label shown in the burned-in caption)
SEGMENTS = [
    ("silence_1", "silence"),
    ("calm_1",    "talking"),
    ("clap_1",    "clap"),
    ("knock_1",   "knock"),
    ("stop_1",    "stop"),
    ("alarm_1",   "alarm"),
    ("loud_1",    "loud"),
]
# Silence between segments — long enough for the dashboard's EMA +
# hysteresis to settle but short enough that the demo fits ~40 s.
GAP_SECONDS = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--gap-seconds", type=float, default=GAP_SECONDS,
                        help=f"silence between segments (default {GAP_SECONDS:.1f}s)")
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "demo.wav",
                        help=f"output wav (default {OUT_DIR / 'demo.wav'})")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    gap = np.zeros(int(args.gap_seconds * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    start_times: list[tuple[float, str, str]] = []  # (start_s, clip_id, label)
    cursor = 0.0

    print(f"Building {args.out} from {len(SEGMENTS)} clips at {SR} Hz\n")
    for clip_id, label in SEGMENTS:
        path = CLIPS_DIR / f"{clip_id}.wav"
        if not path.exists():
            print(f"  WARN: {path} missing, skipping", file=sys.stderr)
            continue
        audio, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            print(f"  WARN: {path} is {file_sr} Hz, expected {SR}; skipping",
                  file=sys.stderr)
            continue
        print(f"  [{cursor:6.2f}s]  {clip_id:12s}  '{label}'  "
              f"({len(audio) / SR:.2f}s)")
        parts.append(audio)
        start_times.append((cursor, clip_id, label))
        cursor += len(audio) / SR
        parts.append(gap)
        cursor += args.gap_seconds

    final = np.concatenate(parts)
    sf.write(str(args.out), final, SR)

    total_s = len(final) / SR
    print(f"\nwrote {args.out}  ({total_s:.2f}s, {len(final)} samples @ {SR} Hz)")
    print("\nsegment start times (for muxing to the video):")
    for t, clip_id, label in start_times:
        print(f"  {t:6.2f}s  {clip_id:12s}  '{label}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())