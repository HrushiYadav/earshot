#!/usr/bin/env python3
"""Stage 2 — evaluate the 8 starter bool questions on every clip in the manifest.

Reads clips/manifest.csv (gitignored) and clips/<label>.wav, runs each of the
8 starter bool questions from earshot.prompts against each clip one clip at a
time (sequentially, no batching — Stage 3 will add the batched variant).

Prints:
  - a table with rows = clips and columns = questions (P(Yes)),
  - per-question accuracy at a threshold (default 0.5) against the manifest,
  - total runtime.

clips/manifest.csv stays gitignored; this script reads it but never writes.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot.prompts import STARTER_BOOL_QUESTIONS
from earshot.scorer import score_sequential


REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO_ROOT / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.csv"
DEFAULT_THRESHOLD = 0.5


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.stdout.write(f"manifest not found at {MANIFEST_PATH}\n")
        sys.stdout.write("Run `uv run scripts/record_session.py --list` first;\n")
        sys.stdout.write("or fill in clips/manifest.csv by hand if you already have clips.\n")
        sys.exit(1)
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def load_audio(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _format_table(clip_results: list[tuple[str, dict[str, float]]],
                  questions: list[dict]) -> str:
    cols = [q["id"] for q in questions]
    lines = []
    header = "{:<22s}".format("clip")
    for c in cols:
        header += "{:>10s}".format(c[:9])
    lines.append(header)
    lines.append("-" * len(header))
    for label, scores in clip_results:
        row = "{:<22s}".format(label)
        for c in cols:
            v = scores.get(c)
            row += "{:>10s}".format("   —   " if v is None else f"{v:.3f}")
        lines.append(row)
    return "\n".join(lines)


def _accuracy(clip_results, manifest_by_label, questions, threshold) -> dict[str, tuple[int, int]]:
    accs: dict[str, tuple[int, int]] = {}
    for q in questions:
        col = q["manifest_col"]
        correct = 0
        total = 0
        for label, scores in clip_results:
            m = manifest_by_label.get(label)
            if m is None:
                continue
            expected = m.get(col, "")
            if expected not in ("Yes", "No"):
                continue
            p_yes = scores.get(q["id"])
            if p_yes is None:
                continue
            predicted = "Yes" if p_yes >= threshold else "No"
            if predicted == expected:
                correct += 1
            total += 1
        accs[q["id"]] = (correct, total)
    return accs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"P(Yes) >= threshold counts as 'Yes' (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Run only these clip labels (default: every label in manifest)")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Run only these question IDs (default: all 8 starters)")
    parser.add_argument("--quiet", action="store_true",
                        help="don't print per-clip progress")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    if not manifest:
        sys.stdout.write("manifest is empty (clips/manifest.csv) — nothing to evaluate.\n")
        return 1

    manifest_by_label = {row["label"]: row for row in manifest}
    labels = [row["label"] for row in manifest]
    if args.clips:
        wanted = set(args.clips)
        labels = [l for l in labels if l in wanted]

    questions = list(STARTER_BOOL_QUESTIONS)
    if args.questions:
        wanted_q = set(args.questions)
        questions = [q for q in questions if q["id"] in wanted_q]
    if not questions:
        sys.stdout.write("no questions selected\n")
        return 1

    n_clips = len(labels)
    n_q = len(questions)
    sys.stdout.write(f"running {n_clips} clip(s) × {n_q} question(s); "
                      f"threshold = {args.threshold}\n\n")
    sys.stdout.flush()

    if n_clips == 0:
        sys.stdout.write("no clips match the filter\n")
        return 1

    clip_results: list[tuple[str, dict[str, float]]] = []
    t_total_start = time.perf_counter()
    for label in labels:
        wav = CLIPS_DIR / f"{label}.wav"
        if not wav.exists():
            sys.stdout.write(f"  ! {label}: missing {wav}, skipping\n")
            continue
        audio = load_audio(wav)
        t0 = time.perf_counter()
        scores = score_sequential(audio, questions)
        dt = time.perf_counter() - t0
        if not args.quiet:
            sys.stdout.write(f"  {label:<22s} {dt:5.1f}s  ({n_q} questions)\n")
            sys.stdout.flush()
        clip_results.append((label, scores))
    t_total = time.perf_counter() - t_total_start

    sys.stdout.write("\n")
    sys.stdout.write(_format_table(clip_results, questions))
    sys.stdout.write("\n\n")

    accs = _accuracy(clip_results, manifest_by_label, questions, args.threshold)
    sys.stdout.write(f"accuracy at threshold >= {args.threshold}\n")
    sys.stdout.write(f"  {'question':<18s} {'correct':>8s} {'total':>6s} {'accuracy':>9s}\n")
    sys.stdout.write("  " + "-" * 44 + "\n")
    n_correct = 0
    n_total = 0
    for q in questions:
        c, t = accs[q["id"]]
        a = (c / t) if t else 0.0
        sys.stdout.write(f"  {q['id']:<18s} {c:>8d} {t:>6d} {a:>9.3f}\n")
        n_correct += c
        n_total += t
    overall = (n_correct / n_total) if n_total else 0.0
    sys.stdout.write("  " + "-" * 44 + "\n")
    sys.stdout.write(f"  {'OVERALL':<18s} {n_correct:>8d} {n_total:>6d} {overall:>9.3f}\n\n")

    if clip_results:
        per_clip = t_total / len(clip_results)
        sys.stdout.write(
            f"runtime: total {t_total:.1f}s, "
            f"per-clip {per_clip:.1f}s, "
            f"per-(clip × Q) {t_total / (len(clip_results) * n_q):.2f}s\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
