"""Generate results/v02_comparison.png from the saved .npz arrays.

Two panels, readable at phone size:
  (a) ESC-50 fold 1 top-1 accuracy per method
  (b) Part C AUROC per task, earshot vs CLAP best of 3 prompts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

BENCH = Path("bench")
OUT = Path("results/v02_comparison.png")


def auroc(score, true_bin):
    if len(np.unique(true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(true_bin, score))


# ---- Panel A: ESC-50 top-1 ----
methods = ["AST+LR", "CLAP\nreadable", "CLAP\nraw",
           "earshot\ncalibrated", "earshot\nraw"]
top1 = [0.9050, 0.8825, 0.8375, 0.8500, 0.8225]
colors = ["#1b7837", "#5aae61", "#c2a5cf", "#4393c3", "#92c5de"]

# ---- Panel B: Part C AUROC ----
tasks = [
    ("calm vs angry", "ravdess_calm_angry"),
    ("speech vs song", "ravdess_speech_song"),
    ("stop vs other", "sc_stop_other"),
    ("q vs stmt", "say_q_stmt"),
]
earshot_aurocs = []
clap_best_aurocs = []
for label, task_id in tasks:
    z = np.load(BENCH / f"v02_earshot_{task_id}.npz")
    earshot_aurocs.append(auroc(z["p_yes"], z["true_bin"]))
    rs = []
    for k in range(3):
        cz = np.load(BENCH / f"v02_clap_{task_id}_pair{k}.npz")
        rs.append(auroc(cz["sims"], cz["true_bin"]))
    clap_best_aurocs.append(max(rs))


def panel_a(ax):
    bars = ax.bar(range(len(methods)), top1, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("top-1 accuracy", fontsize=10)
    ax.set_ylim(0.78, 0.94)
    ax.set_yticks(np.arange(0.80, 0.95, 0.05))
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_title("(a) ESC-50 fold 1 — top-1 accuracy",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    for bar, v in zip(bars, top1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def panel_b(ax):
    x = np.arange(len(tasks))
    w = 0.38
    bars_e = ax.bar(x - w / 2, earshot_aurocs, w, label="earshot",
                    color="#4393c3", edgecolor="black", linewidth=0.5)
    bars_c = ax.bar(x + w / 2, clap_best_aurocs, w, label="CLAP (best of 3 prompts)",
                    color="#c2a5cf", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_title("(b) Part C — AUROC per task",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    for bar, v in zip(bars_e, earshot_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars_c, clap_best_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("v0.2 comparison: earshot vs CLAP vs trained classifier — M4 MacBook Air 16 GB",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()