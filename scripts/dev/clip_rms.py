"""Compute RMS for every clip in the manifest and print sorted by RMS."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

CLIPS_DIR = Path("clips")


def main() -> None:
    labels = []
    with open(CLIPS_DIR / "manifest.csv") as f:
        next(f)
        for row in csv.reader(f):
            if row:
                labels.append(row[1].strip())

    rows = []
    for label in labels:
        path = CLIPS_DIR / f"{label}.wav"
        audio, sr = sf.read(str(path))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio)))
        rows.append((label, len(audio) / sr, sr, rms, peak))

    rows.sort(key=lambda r: r[3])
    print(f"{'label':<18}{'len(s)':>10}{'sr':>8}{'rms':>14}{'peak':>14}  notes")
    print("-" * 78)
    for label, dur, sr, rms, peak in rows:
        flag = ""
        if label in ("silence_1", "silence_2", "room_noise_1"):
            flag = " <-- expected silent"
        elif rms >= 0.01:
            flag = " non-silent"
        print(f"{label:<18}{dur:>10.2f}{sr:>8}{rms:>14.6f}{peak:>14.6f}{flag}")


if __name__ == "__main__":
    main()