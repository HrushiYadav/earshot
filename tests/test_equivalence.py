"""Stage 3 — equivalence test: score_batched must agree with score_sequential
within 0.02 in P(Yes) for every (clip, model question) pair.

The test skips the signal-derived `silent` question (it's not a model call).
For everything else: |batched - sequential| < 0.02.

Run: `.venv/bin/python tests/test_equivalence.py`
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earshot.prompts import STARTER_BOOL_QUESTIONS  # noqa: E402
from earshot.scorer import score_batched, score_sequential  # noqa: E402

CLIPS_DIR = REPO_ROOT / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.csv"
TOLERANCE = 0.02


def load_audio(wav_path: Path) -> np.ndarray:
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 2
    with MANIFEST_PATH.open() as f:
        manifest = list(csv.DictReader(f))

    # Model questions only — silent is signal-derived.
    model_qs = [q for q in STARTER_BOOL_QUESTIONS if q.get("source", "model") == "model"]

    # Warm-up: one clip, both paths, so MPS kernels are compiled before timing.
    first_clip = manifest[0]["label"]
    audio0 = load_audio(CLIPS_DIR / f"{first_clip}.wav")
    print(f"[warm-up] {first_clip}: sequential + batched once each …", flush=True)
    score_sequential(audio0, model_qs)
    score_batched(audio0, model_qs)

    diffs: list[tuple[str, str, float, float, float]] = []
    # (label, qid, sequential_prob, batched_prob, abs_diff)

    n_clips = len(manifest)
    print(f"\n[equiv] running {n_clips} clip(s) × {len(model_qs)} model question(s); "
          f"tolerance = {TOLERANCE}", flush=True)

    t_seq = 0.0
    t_bat = 0.0
    for row in manifest:
        label = row["label"]
        wav = CLIPS_DIR / f"{label}.wav"
        if not wav.exists():
            print(f"  ! {label}: missing {wav}, skipping", file=sys.stderr)
            continue
        audio = load_audio(wav)
        t0 = time.perf_counter()
        seq = score_sequential(audio, model_qs)
        t_seq += time.perf_counter() - t0

        t0 = time.perf_counter()
        bat = score_batched(audio, model_qs)
        t_bat += time.perf_counter() - t0

        for q in model_qs:
            qid = q["id"]
            s = seq[qid]["prob"]
            b = bat[qid]["prob"]
            d = abs(s - b)
            diffs.append((label, qid, s, b, d))

    max_diff = max(d for *_, d in diffs)
    worst = max(diffs, key=lambda r: r[4])

    print(f"\nmax |batched - sequential| = {max_diff:.6f}")
    print(f"worst case: clip={worst[0]!r} q={worst[1]!r} "
          f"seq={worst[2]:.4f} bat={worst[3]:.4f}")
    print(f"runtime: sequential {t_seq:.1f}s, batched {t_bat:.1f}s "
          f"(speedup x{t_seq / max(t_bat, 1e-6):.2f})")

    fails = [(l, q, s, b, d) for l, q, s, b, d in diffs if d >= TOLERANCE]
    if fails:
        print(f"\nFAIL: {len(fails)} of {len(diffs)} cells exceed {TOLERANCE}:")
        for l, q, s, b, d in fails[:20]:
            print(f"  {l:<22s}{q:<18s}seq={s:.4f}  bat={b:.4f}  d={d:.4f}")
        if len(fails) > 20:
            print(f"  ... and {len(fails) - 20} more")
        return 1
    print(f"\nPASS: all {len(diffs)} cells within tolerance {TOLERANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())