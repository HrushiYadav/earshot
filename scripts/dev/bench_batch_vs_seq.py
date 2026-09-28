"""Stage 3 quick timing — sequential vs batched on one clip, after warm-up.

For N in [1, 4, 8], score the model questions on a single clip and report:
  - sequential: time for N single-question forward passes
  - batched:    time for 1 prefix forward + ceil(N/bs) suffix forward passes

Prints a markdown-style table to stdout. The first pass is always slow
(MPS kernel compilation), so warm-up is mandatory for honest numbers.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earshot.prompts import STARTER_BOOL_QUESTIONS  # noqa: E402
from earshot.scorer import score_batched, score_sequential  # noqa: E402

CLIPS_DIR = REPO_ROOT / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.csv"

# Use a representative clip — calm_1 has RMS well above silent threshold
# and exercises a speech-style prompt.
CLIP_LABEL = "calm_1"
N_LIST = [1, 4, 8]
WARMUP_N = 3


def load_audio(wav_path: Path) -> np.ndarray:
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 2
    # Use the actual model questions on this clip, in manifest order.
    with MANIFEST_PATH.open() as f:
        manifest = list(csv.DictReader(f))
    label = CLIP_LABEL
    audio = load_audio(CLIPS_DIR / f"{label}.wav")

    model_qs = [q for q in STARTER_BOOL_QUESTIONS if q.get("source", "model") == "model"]
    print(f"clip={label},  {len(model_qs)} model questions available", flush=True)

    # Warm-up: run both sequential and batched a couple of times so the
    # MPS kernels are compiled. Don't time these.
    print(f"\n[warm-up] {WARMUP_N}x sequential + {WARMUP_N}x batched on {label} …", flush=True)
    for _ in range(WARMUP_N):
        score_sequential(audio, model_qs)
        score_batched(audio, model_qs)

    rows = []
    for N in N_LIST:
        sub = model_qs[:N]
        # 5 timed passes each, take the median.
        seq_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            score_sequential(audio, sub)
            seq_times.append(time.perf_counter() - t0)
        bat_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            score_batched(audio, sub)
            bat_times.append(time.perf_counter() - t0)
        seq_med = float(np.median(seq_times))
        bat_med = float(np.median(bat_times))
        speedup = seq_med / bat_med if bat_med > 0 else float("inf")
        rows.append((N, seq_med, bat_med, speedup))

    print(f"\n{'N':>4s}  {'sequential (s)':>16s}  {'batched (s)':>13s}  {'speedup':>10s}")
    print("-" * 50)
    for N, s, b, sp in rows:
        print(f"{N:>4d}  {s:>16.3f}  {b:>13.3f}  {sp:>9.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())