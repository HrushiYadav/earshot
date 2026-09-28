#!/usr/bin/env python3
"""Stage 1 — record labelled test clips.

Usage:
    uv run scripts/record_clips.py --label music --seconds 5 --count 2

Each clip is saved to clips/<label>_<seconds>[_NN].wav at 16 kHz mono float32.
A 3-2-1 countdown runs before every clip so you have time to set up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO_ROOT / "clips"
DEFAULT_SR = 16_000


def _countdown(seconds_left: int, label: str, idx: int, total: int) -> None:
    """Print a single-line countdown, then a RECORDING marker on the next line."""
    pad = lambda s: f"{s:<8}"  # noqa: E731
    sys.stdout.write(
        f"\r{pad(f'{idx}/{total}')} {label:<20s} recording in {seconds_left}s...   "
    )
    sys.stdout.flush()


def _record_one(seconds: float, sr: int) -> np.ndarray:
    """Record `seconds` from the default mic, return mono float32."""
    frames = int(round(seconds * sr))
    audio = sd.rec(
        frames,
        samplerate=sr,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # block until the buffer is full
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32, copy=False)


def _out_path(label: str, seconds: float, idx: int, total: int) -> Path:
    """Build the output filename. Multiple clips with the same label get a
    zero-padded suffix so they don't overwrite each other."""
    base = f"{label}_{int(round(seconds))}"
    suffix = f"_{idx:02d}" if total > 1 else ""
    return CLIPS_DIR / f"{base}{suffix}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--label", required=True,
                        help="short label for the clip (e.g. 'music', 'silence', 'speech')")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="length of each clip in seconds (default: 5)")
    parser.add_argument("--count", type=int, default=1,
                        help="how many clips to record back-to-back (default: 1)")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR,
                        help=f"sample rate (default: {DEFAULT_SR})")
    parser.add_argument("--no-countdown", action="store_true",
                        help="skip the 3-2-1 countdown (not recommended)")
    args = parser.parse_args()

    if args.seconds <= 0:
        print("--seconds must be > 0", file=sys.stderr)
        return 2
    if args.count <= 0:
        print("--count must be > 0", file=sys.stderr)
        return 2

    try:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"could not create {CLIPS_DIR}: {e}", file=sys.stderr)
        return 1

    try:
        sd.query_devices(kind="input")  # sanity check — fail fast if no input
    except Exception as e:  # noqa: BLE001
        print(f"no usable input device: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "On macOS, allow Terminal in System Settings → Privacy & Security "
            "→ Microphone, then re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"recording {args.count} × {args.seconds}s clip(s) at {args.sr} Hz mono "
          f"float32 → {CLIPS_DIR}/\n", flush=True)

    for i in range(1, args.count + 1):
        if not args.no_countdown:
            for n in (3, 2, 1):
                _countdown(n, args.label, i, args.count)
                time.sleep(1.0)
        else:
            _countdown(0, args.label, i, args.count)

        try:
            audio = _record_one(args.seconds, args.sr)
        except Exception as e:  # noqa: BLE001
            print(f"\nrecord failed on clip {i}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        path = _out_path(args.label, args.seconds, i, args.count)
        try:
            sf.write(str(path), audio, args.sr, subtype="FLOAT")
        except Exception as e:  # noqa: BLE001
            print(f"\nwrite failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

        print(
            f"\r{path.name:<28s}  rms={rms:.4f}  peak={peak:.3f}  "
            f"({audio.shape[0] / args.sr:.2f}s)   ✓",
            flush=True,
        )

    print(f"\nrecorded {args.count} clip(s) to {CLIPS_DIR}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
