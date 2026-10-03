"""Render the earshot vs CLAP vs trained classifier chart.

Two panels, readable at phone size:
  (a) ESC-50 fold 1 top-1 accuracy per method (Y from 0)
  (b) AUROC on three binary speech/tone tasks (earshot vs CLAP best prompt)

Colours consistent across both panels:
  trained classifier = grey
  CLAP = purple (lighter shade for raw labels)
  earshot = blue (lighter shade for raw)
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

# Shared palette
COLOUR_TRAINED = "#777777"          # grey
COLOUR_CLAP_READABLE = "#7e4a96"    # purple
COLOUR_CLAP_RAW = "#cba6d8"        # lighter purple
COLOUR_EARSHOT_CAL = "#2c5d8f"      # blue
COLOUR_EARSHOT_RAW = "#9bbcd9"      # lighter blue


def auroc(score, true_bin):
    if len(np.unique(true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(true_bin, score))


def panel_a(ax):
    methods = [
        ("Trained classifier\n(AST + LR)\n129 ms", 0.9050, COLOUR_TRAINED),
        ("CLAP\n(readable prompts)\n26 ms", 0.8825, COLOUR_CLAP_READABLE),
        ("CLAP\n(raw labels)\n25 ms", 0.8375, COLOUR_CLAP_RAW),
        ("earshot\n(calibrated)\n4.3 s", 0.8500, COLOUR_EARSHOT_CAL),
        ("earshot\n(raw)\n4.3 s", 0.8225, COLOUR_EARSHOT_RAW),
    ]
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colours = [m[2] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Top-1 accuracy", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Recognising 50 everyday sounds (ESC-50, 400 clips)",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def panel_b(ax):
    # Three tasks only (drop the synthetic statement/question).
    tasks = [
        ("Calm vs angry\n(same sentence)", "ravdess_calm_angry"),
        ("Speaking vs singing\n(same words)", "ravdess_speech_song"),
        ("'Stop' vs other words", "sc_stop_other"),
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

    x = np.arange(len(tasks))
    w = 0.38
    bars_e = ax.bar(x - w / 2, earshot_aurocs, w, label="earshot",
                    color=COLOUR_EARSHOT_CAL, edgecolor="black", linewidth=0.5)
    bars_c = ax.bar(x + w / 2, clap_best_aurocs, w,
                    label="CLAP (best of 3 prompts)",
                    color=COLOUR_CLAP_READABLE, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7)
    ax.text(0.99, 0.52, "coin flip", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="grey", style="italic")
    ax.set_title("Questions about speech (AUROC, 0.5 = coin flip)",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    for bar, v in zip(bars_e, earshot_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(bars_c, clap_best_aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("earshot vs CLAP vs a trained classifier (M4 MacBook Air, 16 GB)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "ESC-50 fold 1; RAVDESS (n=384 per task); Speech Commands (n=500). "
             "All on one M4 MacBook Air.",
             ha="center", va="top", fontsize=8.5, color="grey", style="italic")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=180)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()