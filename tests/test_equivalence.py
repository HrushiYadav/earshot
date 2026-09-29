"""Stage 3 — equivalence test: score_batched must agree with score_sequential
within the configured tolerance, for every (clip, model question) pair.

The test skips signal-derived questions (currently just `silent`). For
everything else:
  - bool: |P_yes_seq - P_yes_bat| < tolerance
  - choice: L1 distance over the probs dict < tolerance

Modes (after the user's decision on fp16 vs fp32 tolerances):

  default (fp16, tolerance 0.01)
    Default clip set: silence_1, calm_1, music_1, loud_1, typing_1 (5 clips).
    --all flag: run every manifest clip instead of the default 5.
    For Stage 3's regular smoke + Stage 4 production runs.

  --fp32 (tolerance 0.001)
    Loads the model in float32 for an exactness proof. Default clip set:
    silence_1, calm_1, music_1 (3 clips — fp32 is slow; budget a few
    minutes). The model tries MPS first and falls back to CPU if 32-bit
    weights don't fit in RAM.

  --wrong-pos (suffix positions +1)
    Control test — must fail in either mode, proving the test is
    sensitive to position bugs. Requires --fp32 (otherwise fp16 drift
    can mask the deliberate bug); the script errors out if --wrong-pos
    is passed without --fp32.

Usage:
  python tests/test_equivalence.py                # fp16, 5 clips, tolerance 0.01
  python tests/test_equivalence.py --all          # fp16, all clips,  tolerance 0.01
  python tests/test_equivalence.py --fp32         # fp32, 3 clips, tolerance 0.001
  python tests/test_equivalence.py --fp32 --wrong-pos  # MUST FAIL
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earshot.prompts import load_default_questions  # noqa: E402
from earshot.schema import BoolQuestion, ChoiceQuestion  # noqa: E402
from earshot.scorer import (  # noqa: E402
    BoolResult,
    ChoiceResult,
    score_batched,
    score_sequential,
)

CLIPS_DIR = REPO_ROOT / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.csv"

TOLERANCE_FP16 = 0.01
TOLERANCE_FP32 = 0.001

# Default clip sets per mode. Picked to cover silence, speech, and music
# in fp32 (3 clips) and to spread across more categories in fp16 (5).
FP32_DEFAULT_CLIPS = ["silence_1", "calm_1", "music_1"]
FP16_DEFAULT_CLIPS = ["silence_1", "calm_1", "music_1", "loud_1", "typing_1"]


def load_audio(wav_path: Path) -> np.ndarray:
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def prob_diff(a, b, q) -> float:
    """The scalar difference metric for one (sequential, batched) result
    pair. bool: |P_yes_seq - P_yes_bat|. choice: L1 distance over probs."""
    if isinstance(q, ChoiceQuestion):
        keys = set(a.probs) | set(b.probs)
        return sum(abs(a.probs.get(k, 0.0) - b.probs.get(k, 0.0)) for k in keys)
    return abs(a.p - b.p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fp32", action="store_true",
                        help="Load the model in float32 for an exactness proof "
                             "(tighter tolerance, slower).")
    parser.add_argument("--all", action="store_true",
                        help="fp16 only: run every manifest clip instead of "
                             "the default 5-clip subset.")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Run only these clip labels (overrides the "
                             "default set for either mode).")
    parser.add_argument("--wrong-pos", action="store_true",
                        help="Control: offset suffix positions by +1. In "
                             "this mode the test MUST fail. Defaults to fp16; "
                             "if the max diff doesn't exceed the tolerance, "
                             "that's evidence the test isn't sensitive enough "
                             "and the fp32 proof is needed.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dtype = torch.float32 if args.fp32 else torch.float16
    tolerance = TOLERANCE_FP32 if args.fp32 else TOLERANCE_FP16

    if not MANIFEST_PATH.exists():
        print(f"manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 2
    with MANIFEST_PATH.open() as f:
        manifest = list(csv.DictReader(f))

    qs = load_default_questions()
    model_qs = [q for q in qs if q.source == "model"]
    if not model_qs:
        print("no model questions in questions.yaml — nothing to test", file=sys.stderr)
        return 2

    # Choose clip set
    if args.clips:
        labels = [l for l in (row["label"] for row in manifest) if l in args.clips]
    elif args.fp32:
        labels = [l for l in (row["label"] for row in manifest) if l in FP32_DEFAULT_CLIPS]
    elif args.all:
        labels = [row["label"] for row in manifest]
    else:
        labels = [l for l in (row["label"] for row in manifest) if l in FP16_DEFAULT_CLIPS]

    if not labels:
        print("no clips match the filter", file=sys.stderr)
        return 2

    # Warm-up: one clip, both paths, so MPS kernels are compiled.
    first_clip = labels[0]
    audio0 = load_audio(CLIPS_DIR / f"{first_clip}.wav")
    print(f"[warm-up] {first_clip}: sequential + batched once each "
          f"(dtype={'fp32' if args.fp32 else 'fp16'}) …", flush=True)
    score_sequential(audio0, model_qs, dtype=dtype)
    score_batched(audio0, model_qs, suffix_pos_offset=0, dtype=dtype)
    if args.wrong_pos:
        # Also warm the wrong-pos path so its kernel cost is paid here.
        score_batched(audio0, model_qs, suffix_pos_offset=1, dtype=dtype)

    n_clips = len(labels)
    n_bool = sum(1 for q in model_qs if isinstance(q, BoolQuestion))
    n_choice = sum(1 for q in model_qs if isinstance(q, ChoiceQuestion))
    print(f"\n[equiv] running {n_clips} clip(s) × {len(model_qs)} model question(s) "
          f"({n_bool} bool + {n_choice} choice); tolerance = {tolerance}; "
          f"dtype = {'fp32' if args.fp32 else 'fp16'}; wrong_pos = {args.wrong_pos}",
          flush=True)

    diffs = []  # (label, qid, diff, seq_res, bat_res)
    t_seq = 0.0
    t_bat = 0.0
    for label in labels:
        wav = CLIPS_DIR / f"{label}.wav"
        if not wav.exists():
            print(f"  ! {label}: missing {wav}, skipping", file=sys.stderr)
            continue
        audio = load_audio(wav)
        t0 = time.perf_counter()
        seq = score_sequential(audio, model_qs, dtype=dtype)
        t_seq += time.perf_counter() - t0

        t0 = time.perf_counter()
        bat = score_batched(
            audio, model_qs,
            suffix_pos_offset=1 if args.wrong_pos else 0,
            dtype=dtype,
        )
        t_bat += time.perf_counter() - t0

        for q in model_qs:
            qid = q.id
            s_res = seq[qid]
            b_res = bat[qid]
            d = prob_diff(s_res, b_res, q)
            diffs.append((label, qid, d, s_res, b_res))

    max_diff = max(d for *_, d, _, _ in diffs)
    worst = max(diffs, key=lambda r: r[2])

    print(f"\nmax diff = {max_diff:.6f}")
    print(f"worst case: clip={worst[0]!r} q={worst[1]!r} "
          f"seq={_fmt_result(worst[3])} bat={_fmt_result(worst[4])}")
    print(f"runtime: sequential {t_seq:.1f}s, batched {t_bat:.1f}s "
          f"(speedup x{t_seq / max(t_bat, 1e-6):.2f})")

    fails = [(l, q, d, sr, br) for l, q, d, sr, br in diffs if d >= tolerance]

    if args.wrong_pos:
        print(f"\n[wrong-pos] cells over tolerance: {len(fails)}/{len(diffs)}")
        for l, q, d, sr, br in sorted(fails, key=lambda r: -r[2])[:10]:
            print(f"  {l:<22s}{q:<16s}diff={d:.4f}  "
                  f"seq={_fmt_result(sr)} bat={_fmt_result(br)}")
        if fails:
            print(f"\nPASS: --wrong-pos broke equivalence (as expected). "
                  f"Test is sensitive to position bugs.")
            return 0
        print(f"\nFAIL: --wrong-pos did NOT break equivalence. "
              f"Test is not sensitive enough — loosen the bug or tighten the metric.")
        return 1

    if fails:
        print(f"\nFAIL: {len(fails)} of {len(diffs)} cells exceed {tolerance}:")
        for l, q, d, sr, br in fails[:20]:
            print(f"  {l:<22s}{q:<18s}diff={d:.4f}  "
                  f"seq={_fmt_result(sr)} bat={_fmt_result(br)}")
        if len(fails) > 20:
            print(f"  ... and {len(fails) - 20} more")
        return 1
    print(f"\nPASS: all {len(diffs)} cells within tolerance {tolerance}")
    return 0


def _fmt_result(res) -> str:
    if isinstance(res, BoolResult):
        return f"P(Yes)={res.p:.4f}"
    if isinstance(res, ChoiceResult):
        top = res.top
        rest = " ".join(f"{k}={v:.2f}" for k, v in res.probs.items() if k != top)
        return f"top={top}  ({rest})"
    return repr(res)


if __name__ == "__main__":
    raise SystemExit(main())