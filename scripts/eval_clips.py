#!/usr/bin/env python3
"""Evaluate every model-driven question (originals + variants) against the manifest.

Reads clips/manifest.csv (gitignored) and clips/<label>.wav, runs each of
ALL_BOOL_QUESTIONS from earshot.prompts against each clip sequentially, and
prints:

  - the signal-derived `silent` threshold picked from the manifest data
  - a wide per-clip × per-question P(Yes) table (with source tag)
  - per-question accuracy at the threshold, broken down into Yes-caught
    and No-correct counts
  - an originals-vs-variants side-by-side comparison grouped by manifest_col
  - a list of every remaining miss (clip, question, expected, prob)

Stage 3 will add a fork-and-score mode; this script stays sequential so the
output is comparable across stages.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot.prompts import (
    SIGNAL_SILENT_RMS_THRESHOLD,
    STARTER_BOOL_QUESTIONS,
)
from earshot.scorer import rms, score_sequential


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
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _format_table(clip_results, questions, threshold) -> str:
    """Wide per-clip × per-question P(Yes) table. Cell shows prob + a
    one-letter source tag (M=model, S=signal). Cell suffix shows the
    binary answer in parentheses when it disagrees with P(Yes)>=0.5
    (basically never — kept as a sanity check)."""
    cols = [q["id"] for q in questions]
    header = "{:<22s}".format("clip")
    for c in cols:
        header += "{:>9s}".format(c[:8])
    header += "{:>9s}".format("rms")
    lines = [header, "-" * len(header)]
    for label, scores in clip_results:
        sr = scores.get("__rms__")
        row = "{:<22s}".format(label)
        for c in cols:
            res = scores.get(c)
            if res is None:
                row += "{:>9s}".format("   —    ")
                continue
            p = res["prob"]
            src = "S" if res["source"] == "signal" else "M"
            cell = f"{p:.3f}{src}"
            if (p >= 0.5) != (res["answer"] == "Yes"):
                cell = "!" + cell[1:]
            row += "{:>9s}".format(cell)
        row += "{:>9.4f}".format(sr if sr is not None else 0.0)
        lines.append(row)
    return "\n".join(lines)


def _per_question(clip_results, manifest_by_label, questions, threshold):
    """For each question: (correct, total, yes_caught, yes_total, no_correct, no_total)."""
    stats: dict[str, dict] = {}
    for q in questions:
        col = q["manifest_col"]
        s = {"correct": 0, "total": 0, "yes_correct": 0, "yes_total": 0,
             "no_correct": 0, "no_total": 0}
        for label, scores in clip_results:
            m = manifest_by_label.get(label)
            if m is None:
                continue
            expected = m.get(col, "")
            if expected not in ("Yes", "No"):
                continue
            res = scores.get(q["id"])
            if res is None:
                continue
            predicted = res["answer"]
            ok = predicted == expected
            s["total"] += 1
            s["correct"] += int(ok)
            if expected == "Yes":
                s["yes_total"] += 1
                s["yes_correct"] += int(ok)
            else:
                s["no_total"] += 1
                s["no_correct"] += int(ok)
        stats[q["id"]] = s
    return stats


def _format_accuracy(stats: dict[str, dict], questions) -> str:
    out = []
    out.append(f"accuracy at P(Yes) >= 0.5 — split by expected label")
    out.append(f"  {'question':<18s}{'src':>4s}{'correct':>9s}{'total':>7s}{'acc':>8s}"
               f"  {'yes_caught':>12s}{'no_correct':>12s}")
    out.append("  " + "-" * 70)
    total_c = total_t = total_yc = total_yt = total_nc = total_nt = 0
    for q in questions:
        s = stats[q["id"]]
        acc = (s["correct"] / s["total"]) if s["total"] else 0.0
        yc_yt = f"{s['yes_correct']}/{s['yes_total']}"
        nc_nt = f"{s['no_correct']}/{s['no_total']}"
        src = "M" if q["id"] != "silent" else "S"
        out.append(f"  {q['id']:<18s}{src:>4s}{s['correct']:>9d}{s['total']:>7d}{acc:>8.3f}"
                   f"  {yc_yt:>12s}{nc_nt:>12s}")
        total_c += s["correct"]
        total_t += s["total"]
        total_yc += s["yes_correct"]
        total_yt += s["yes_total"]
        total_nc += s["no_correct"]
        total_nt += s["no_total"]
    out.append("  " + "-" * 70)
    overall = (total_c / total_t) if total_t else 0.0
    out.append(f"  {'OVERALL':<18s}{'':>4s}{total_c:>9d}{total_t:>7d}{overall:>8.3f}"
               f"  {f'{total_yc}/{total_yt}':>12s}{f'{total_nc}/{total_nt}':>12s}")
    return "\n".join(out)


def _format_variant_compare(stats: dict[str, dict]) -> str:
    """Variant compare section is a no-op after the tuning pass folds the
    winners back into STARTER_BOOL_QUESTIONS. Kept as a stub so the eval
    output layout stays stable across tuning re-runs."""
    return "originals vs variants — n/a (tuning pass complete)"


def _format_misses(clip_results, manifest_by_label, questions, threshold) -> str:
    out = ["misses (clip, question, expected, P(Yes), source)"]
    n = 0
    for label, scores in clip_results:
        m = manifest_by_label.get(label)
        if m is None:
            continue
        for q in questions:
            col = q["manifest_col"]
            expected = m.get(col, "")
            if expected not in ("Yes", "No"):
                continue
            res = scores.get(q["id"])
            if res is None:
                continue
            predicted = res["answer"]
            if predicted != expected:
                n += 1
                out.append(f"  {label:<22s}{q['id']:<14s}expected={expected:<4s}"
                           f"got={predicted:<4s}{res['prob']:.3f}  src={res['source']}")
    if n == 0:
        return "misses (clip, question, expected, P(Yes), source)\n  (none)"
    out.insert(1, f"  {n} miss(es)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"P(Yes) >= threshold counts as 'Yes' (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Run only these clip labels (default: every label in manifest)")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Run only these question IDs (default: every starter question)")
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
    sys.stdout.write(
        f"running {n_clips} clip(s) × {n_q} question(s); threshold = {args.threshold}\n"
        f"signal_silent: rms < {SIGNAL_SILENT_RMS_THRESHOLD:.5f} → Yes\n"
        f"  picked from manifest data between the loudest 'silent' clip and "
        f"the quietest 'non-silent' clip\n\n"
    )
    sys.stdout.flush()

    if n_clips == 0:
        sys.stdout.write("no clips match the filter\n")
        return 1

    clip_results = []
    t_total_start = time.perf_counter()
    for label in labels:
        wav = CLIPS_DIR / f"{label}.wav"
        if not wav.exists():
            sys.stdout.write(f"  ! {label}: missing {wav}, skipping\n")
            continue
        audio = load_audio(wav)
        t0 = time.perf_counter()
        scores = score_sequential(audio, questions)
        scores["__rms__"] = rms(audio)
        dt = time.perf_counter() - t0
        if not args.quiet:
            sys.stdout.write(f"  {label:<22s} {dt:5.1f}s  ({n_q} questions, "
                             f"rms={scores['__rms__']:.4f})\n")
            sys.stdout.flush()
        clip_results.append((label, scores))
    t_total = time.perf_counter() - t_total_start

    sys.stdout.write("\n")
    sys.stdout.write(_format_table(clip_results, questions, args.threshold))
    sys.stdout.write("\n\n")

    stats = _per_question(clip_results, manifest_by_label, questions, args.threshold)
    sys.stdout.write(_format_accuracy(stats, questions))
    sys.stdout.write("\n\n")

    sys.stdout.write(_format_variant_compare(stats))
    sys.stdout.write("\n\n")

    sys.stdout.write(_format_misses(clip_results, manifest_by_label, questions, args.threshold))
    sys.stdout.write("\n\n")

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