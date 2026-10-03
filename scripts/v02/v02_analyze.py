"""v0.2 redo analysis: load saved earshot + CLAP data, compute AUROC and
calibrated ESC-50 accuracy. No model needed.

Outputs a single table with:
  - per-task accuracy (earshot) + AUROC
  - per-task CLAP accuracy (min/max over 3 prompt pairs) + AUROC (min/max)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SAVE_DIR = Path("bench")
TASKS = [
    ("ravdess_calm_angry", "angry", "calm", "Does the speaker sound angry?",
     "RAVDESS calm vs angry", "CC BY-NC-SA 4.0"),
    ("ravdess_speech_song", "song", "speech", "Is the person singing rather than speaking?",
     "RAVDESS speech vs song", "CC BY-NC-SA 4.0"),
    ("sc_stop_other", "stop", "other", "Did someone say the word stop?",
     "Speech Commands v0.02 stop vs other", "CC BY 4.0"),
    ("say_q_stmt", "question", "statement", "Is the person asking a question?",
     "macOS `say` statement vs question", "SYNTHETIC (n=20)"),
]


def auroc(score, true_bin):
    if len(np.unique(true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(true_bin, score))


def main():
    print("=" * 78)
    print("v0.2 redo — analysis (no model)")
    print("=" * 78)

    # ---- 1. ESC-50 with per-question calibration ----
    print("\n## ESC-50 fold 1: raw vs per-question calibrated\n")
    z = np.load(SAVE_DIR / "v02_earshot_esc50.npz", allow_pickle=True)
    P = z["P_yes"]                              # (400, 50)
    true_idx = z["true_idx"]                    # (400,)
    labels = list(z["labels"])

    bz = np.load(SAVE_DIR / "v02_earshot_bias.npz", allow_pickle=True)
    bias = bz["bias"]                           # (3, 50)
    bias_names = list(bz["names"])
    bias_mean = bias.mean(axis=0)               # (50,)

    eps = 1e-6
    # Raw argmax
    pred_raw = P.argmax(axis=1)
    acc_raw = float((pred_raw == true_idx).mean())

    # Per-question calibrated: logit(P) − logit(bias_q)
    logit = np.log(np.clip(P, eps, 1 - eps) / np.clip(1 - P, eps, 1 - eps))
    bias_logit = np.log(np.clip(bias_mean, eps, 1 - eps) /
                        np.clip(1 - bias_mean, eps, 1 - eps))
    P_cal = 1.0 / (1.0 + np.exp(-(logit - bias_logit[None, :])))
    pred_cal = P_cal.argmax(axis=1)
    acc_cal = float((pred_cal == true_idx).mean())

    # Top-5 (raw)
    top5_raw = np.argsort(-P, axis=1)[:, :5]
    top5_acc_raw = float(np.mean([true_idx[i] in top5_raw[i] for i in range(len(true_idx))]))
    top5_cal = np.argsort(-P_cal, axis=1)[:, :5]
    top5_acc_cal = float(np.mean([true_idx[i] in top5_cal[i] for i in range(len(true_idx))]))

    print(f"  bias sources: {bias_names}")
    print(f"  raw argmax:   top-1 = {acc_raw:.4f}  top-5 = {top5_acc_raw:.4f}")
    print(f"  calibrated:   top-1 = {acc_cal:.4f}  top-5 = {top5_acc_cal:.4f}")
    delta = acc_cal - acc_raw
    print(f"  delta:        {delta:+.4f}  (top-1)")
    # Show top-10 calibrated confusions
    if delta != 0:
        from collections import Counter
        cm = Counter()
        for i in range(len(true_idx)):
            if pred_cal[i] != true_idx[i]:
                cm[(labels[true_idx[i]], labels[pred_cal[i]])] += 1
        print("  top-10 calibrated confusions:")
        for (t, p), c in cm.most_common(10):
            print(f"    {c:3d}×  {t:20s} -> {p}")
    print(f"  bias stats (mean over 50 questions):")
    print(f"    min={bias_mean.min():.4f}  median={np.median(bias_mean):.4f}  "
          f"max={bias_mean.max():.4f}")
    print(f"    most-biased-to-Yes (5):")
    for j in np.argsort(bias_mean)[::-1][:5]:
        print(f"      {labels[j]:20s}  bias={bias_mean[j]:.3f}")

    # ---- 2. Part C: AUROC for both methods ----
    print("\n## Part C — accuracy (with counts) + AUROC\n")
    print(f"{'task':35s} {'n':>4s} {'base':>6s}  "
          f"{'earshot acc':>11s}  {'earshot AUROC':>13s}  "
          f"{'CLAP acc min/max':>19s}  {'CLAP AUROC min/max':>19s}")
    print("-" * 130)
    rows = []
    for task_id, pos, neg, q, label, lic in TASKS:
        # earshot
        ez = np.load(SAVE_DIR / f"v02_earshot_{task_id}.npz")
        p_yes = ez["p_yes"]
        true_bin = ez["true_bin"]
        n = int(ez["n"])
        # threshold at 0.5
        preds = (p_yes > 0.5).astype(np.int64)
        acc_e = float((preds == true_bin).mean())
        correct_e = int((preds == true_bin).sum())
        auroc_e = auroc(p_yes, true_bin)
        # baseline
        n_pos = int(true_bin.sum())
        n_neg = n - n_pos
        baseline = max(n_pos, n_neg) / n

        # CLAP — load 3 pairs
        accs, aurocs = [], []
        for k in range(3):
            cz = np.load(SAVE_DIR / f"v02_clap_{task_id}_pair{k}.npz")
            sims = cz["sims"]
            cz_true = cz["true_bin"]
            assert (cz_true == true_bin).all(), "label mismatch"
            preds_c = (sims > 0).astype(np.int64)
            accs.append(float((preds_c == true_bin).mean()))
            aurocs.append(auroc(sims, true_bin))
        a_lo, a_hi = min(accs), max(accs)
        r_lo, r_hi = min(aurocs), max(aurocs)

        print(f"{label:35s} {n:4d} {baseline:6.3f}  "
              f"{acc_e:8.4f} ({correct_e}/{n})  {auroc_e:10.4f}  "
              f"{a_lo:6.4f}-{a_hi:6.4f}     {r_lo:6.4f}-{r_hi:6.4f}")
        rows.append((task_id, label, lic, n, n_pos, n_neg, baseline,
                     acc_e, correct_e, auroc_e,
                     accs, aurocs))

    # ---- 3. Show the 3 CLAP prompts used per task ----
    print("\n## CLAP prompt pairs used (pos vs neg)\n")
    for task_id, *_ in rows:
        print(f"  {task_id}:")
        for k in range(3):
            cz = np.load(SAVE_DIR / f"v02_clap_{task_id}_pair{k}.npz")
            print(f"    pair {k}: {cz['neg_prompt']!r}  vs  {cz['pos_prompt']!r}")


if __name__ == "__main__":
    main()