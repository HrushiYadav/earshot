"""v0.2 redo: measure peak MPS in a FRESH process for 1-question tasks.

The earlier 10.31 GB peak was on a process that had already done the
50-question ESC-50 batch. That memory is allocator high-water — it does not
get returned even when we drop down to 1 question. This script runs
ONLY a 1-question task in a brand-new process and reports the real peak.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from earshot.scorer import score_batched  # noqa: E402
from earshot.schema import BoolQuestion  # noqa: E402

SAMPLE_RATE = 16000
_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def main():
    # Use 5 RAVDESS clips, 1 question ("Does the speaker sound angry?")
    # Skip model load cost by relying on _ensure_loaded
    audio_root = Path("data/ravdess/Audio_Speech_Actors_01-24/Actor_01")
    wavs = sorted(audio_root.glob("03-01-02-*.wav"))[:5]  # 5 calm clips
    if not wavs:
        print("no RAVDESS clips found", file=sys.stderr)
        sys.exit(1)
    import soundfile as sf
    qs = [BoolQuestion(id="q", type="bool",
                       text="Does the speaker sound angry?",
                       manifest_col=None, smoothing=0.0, enter=0.5, exit=0.5,
                       source="model")]

    # Warmup
    audio, sr = sf.read(str(wavs[0]), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio.astype(np.float32)
    print(f"  clip 0: shape={audio.shape}  rms={np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  pre-load  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB")
    _ = score_batched(audio, qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    print(f"  post-warmup  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB")

    # Run 5 clips
    mps_peaks = [mps_gb()]
    t_start = time.perf_counter()
    for i, wav in enumerate(wavs):
        a, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
        a = a.astype(np.float32)
        res = score_batched(a, qs)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        mps_peaks.append(mps_gb())
        print(f"  clip {i+1}: P(Yes)={float(res['q'].p):.3f}  "
              f"mps_now={mps_gb():.2f}GB  mps_peak={max(mps_peaks):.2f}GB")

    print()
    print(f"  fresh-process 1-question peak MPS: {max(mps_peaks):.2f} GB")
    print(f"  fresh-process 1-question peak cpu_rss: {cpu_rss_gb():.2f} GB")
    print(f"  total time: {time.perf_counter()-t_start:.1f} s")


if __name__ == "__main__":
    main()