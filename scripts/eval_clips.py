#!/usr/bin/env python3
"""Evaluate every model-driven question in `questions.yaml` against the manifest.

Reads clips/manifest.csv (gitignored) and clips/<label>.wav, runs each
question from `questions.yaml` sequentially against every clip, and
prints:

  - the signal-derived `silent` threshold picked from the manifest data
  - a wide per-clip × per-question P(Yes) / P(top) table (with source tag)
  - per-question accuracy at the threshold, broken down into Yes-caught
    and No-correct counts
  - the remaining miss list (clip, question, expected, prob)

Stage 4 change: questions are loaded from `questions.yaml` via the
typed schema (`BoolQuestion` / `ChoiceQuestion`) instead of a hard-coded
list. Bool cells show P(Yes); choice cells show P(top) for now (the
eval manifest only carries Yes/No labels).

Stage 3 will add a fork-and-score mode; this script stays sequential so
the output is comparable across stages.
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
    load_default_questions,
)
from earshot.scorer import (
    BoolResult,
    ChoiceResult,
    rms,
    score_sequential,
)
from earshot.schema import BoolQuestion, ChoiceQuestion


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


def _result_p(res) -> float:
    """Cell display value: P(Yes) for bool, P(top) for choice."""
    if isinstance(res, BoolResult):
        return res.p
    if isinstance(res, ChoiceResult):
        return res.probs.get(res.top, 0.0)
    raise TypeError(f"unknown result type: {type(res)}")


def _result_source(res) -> str:
    return getattr(res, "source", "model")


def _format_table(clip_results, questions, threshold) -> str:
    """Wide per-clip × per-question table. Cell shows prob (P(Yes) for bool,
    P(top) for choice) plus a one-letter source tag (M=model, S=signal)."""
    cols = [q.id for q in questions]
    header = "{:<22s}".format("clip")
    for c in cols:
        header += "{:>10s}".format(c[:9])
    header += "{:>10s}".format("rms")
    lines = [header, "-" * len(header)]
    for label, scores in clip_results:
        sr = scores.get("__rms__")
        row = "{:<22s}".format(label)
        for c in cols:
            res = scores.get(c)
            if res is None:
                row += "{:>10s}".format("   —    ")
                continue
            p = _result_p(res)
            src = "S" if _result_source(res) == "signal" else "M"
            row += "{:>10s}".format(f"{p:.3f}{src}")
        row += "{:>10.4f}".format(sr if sr is not None else 0.0)
        lines.append(row)
    return "\n".join(lines)


def _per_question(clip_results, manifest_by_label, questions, threshold):
    """For each question, compute per-question accuracy and split counts.

    Bool: scored as "predicted label == manifest label" (Yes/No). The
    per-option breakdown uses the same Yes/No buckets so the bool row
    shows `yes_caught=11/11 no_correct=10/10`.
    Choice: scored as "predicted top option == manifest value" (e.g. the
            manifest's `main_sound` cell). Per-option counts are
            tracked per option name.
    """
    stats: dict[str, dict] = {}
    for q in questions:
        col = q.manifest_col
        s = {"correct": 0, "total": 0}
        per_option_hits: dict[str, list[int]] = {}
        per_option_totals: dict[str, int] = {}
        for label, scores in clip_results:
            m = manifest_by_label.get(label)
            if m is None:
                continue
            expected = m.get(col or "", "")
            if not expected:
                continue
            res = scores.get(q.id)
            if res is None:
                continue
            predicted = _predicted_from_result(res)
            if predicted is None:
                continue
            ok = predicted == expected
            s["total"] += 1
            s["correct"] += int(ok)
            # Both bool (Yes/No) and choice (option name) get per-bucket
            # counts so the accuracy table can show caught / total per bucket.
            per_option_totals[expected] = per_option_totals.get(expected, 0) + 1
            per_option_hits.setdefault(expected, []).append(int(ok))
        s["per_option_hits"] = per_option_hits
        s["per_option_totals"] = per_option_totals
        stats[q.id] = s
    return stats


def _predicted_from_result(res) -> str | None:
    """Map a typed result to a string label for eval.

    bool → "Yes" if fired else "No".
    choice → res.top (the highest-probability option). For main_sound
    the manifest values are speech / music / noise / silence — exact
    string match with the option list.
    """
    if isinstance(res, BoolResult):
        return "Yes" if res.fired else "No"
    if isinstance(res, ChoiceResult):
        return res.top
    return None


def _format_accuracy(stats: dict[str, dict], questions) -> str:
    """Per-question accuracy. Bool questions show Yes-caught / No-correct;
    choice questions show per-option hit counts (e.g. `speech 3/3`,
    `music 2/4`). The OVERALL row is the unweighted mean of per-question
    accuracy so a 100%-trivially-clipped question can't drag it up.
    """
    out = ["per-question accuracy"]
    out.append("  bool: split by expected Yes / No (correct / total)")
    out.append("  choice: split by expected option (caught / total)")
    out.append("")
    out.append(f"  {'question':<20s}{'src':>4s}{'kind':>8s}{'correct':>9s}"
               f"{'total':>7s}{'acc':>8s}  breakdown")
    out.append("  " + "-" * 86)
    per_q_acc = []
    for q in questions:
        s = stats.get(q.id, {})
        correct = s.get("correct", 0); total = s.get("total", 0)
        acc = (correct / total) if total else 0.0
        per_q_acc.append(acc)
        src = "S" if isinstance(q, BoolQuestion) and q.source == "signal" else "M"
        kind = "choice" if isinstance(q, ChoiceQuestion) else "bool"

        if isinstance(q, ChoiceQuestion):
            parts = []
            for opt in q.options:
                tot = s.get("per_option_totals", {}).get(opt, 0)
                hits = sum(s.get("per_option_hits", {}).get(opt, []))
                parts.append(f"{opt}={hits}/{tot}")
            breakdown = "  ".join(parts)
        else:
            yes_hits = sum(s.get("per_option_hits", {}).get("Yes", []))
            no_hits = sum(s.get("per_option_hits", {}).get("No", []))
            yes_total = s.get("per_option_totals", {}).get("Yes", 0)
            no_total = s.get("per_option_totals", {}).get("No", 0)
            breakdown = f"yes_caught={yes_hits}/{yes_total}  no_correct={no_hits}/{no_total}"

        out.append(f"  {q.id:<20s}{src:>4s}{kind:>8s}{correct:>9d}{total:>7d}{acc:>8.3f}  {breakdown}")
    out.append("  " + "-" * 86)
    overall = (sum(per_q_acc) / len(per_q_acc)) if per_q_acc else 0.0
    out.append(f"  {'OVERALL (mean)':<20s}{'':>4s}{'':>8s}{'':>9s}"
               f"{'':>7s}{overall:>8.3f}")
    return "\n".join(out)


def _format_misses(clip_results, manifest_by_label, questions) -> str:
    out = ["misses (clip, question, expected, predicted, P(Yes)/P(top), source)"]
    n = 0
    for label, scores in clip_results:
        m = manifest_by_label.get(label)
        if m is None:
            continue
        for q in questions:
            col = q.manifest_col
            expected = m.get(col or "", "")
            if not expected:
                continue
            res = scores.get(q.id)
            if res is None:
                continue
            predicted = _predicted_from_result(res)
            if predicted is None:
                continue
            if predicted != expected:
                n += 1
                p = _result_p(res)
                out.append(f"  {label:<22s}{q.id:<16s}expected={expected:<8s}"
                           f"got={predicted:<8s}{p:.3f}  src={_result_source(res)}")
    if n == 0:
        return "misses (clip, question, expected, predicted, P(Yes)/P(top), source)\n  (none)"
    out.insert(1, f"  {n} miss(es)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=argparse.SUPPRESS)  # legacy; per-question `enter` thresholds in YAML are used
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Run only these clip labels (default: every label in manifest)")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Run only these question IDs (default: every question in questions.yaml)")
    parser.add_argument("--quiet", action="store_true",
                        help="don't print per-clip progress")
    parser.add_argument("--questions-path", default=None,
                        help="path to questions.yaml (default: repo root)")
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

    if args.questions_path:
        from earshot.schema import load_questions
        questions = load_questions(args.questions_path)
    else:
        questions = load_default_questions()
    if args.questions:
        wanted_q = set(args.questions)
        questions = [q for q in questions if q.id in wanted_q]
    if not questions:
        sys.stdout.write("no questions selected\n")
        return 1

    n_clips = len(labels)
    n_q = len(questions)
    bool_count = sum(1 for q in questions if isinstance(q, BoolQuestion))
    choice_count = sum(1 for q in questions if isinstance(q, ChoiceQuestion))
    sys.stdout.write(
        f"running {n_clips} clip(s) × {n_q} question(s) "
        f"({bool_count} bool + {choice_count} choice); per-question enter thresholds from YAML\n"
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

    sys.stdout.write(_format_misses(clip_results, manifest_by_label, questions))
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