"""v0.2 Part A baseline: CLAP zero-shot on ESC-50 fold 1.

Top-1 accuracy of `laion/clap-htsat-unfused` using prompts
"the sound of a {label}" for the 50 ESC-50 labels.

Reports:
  - top-1 accuracy
  - top-5 accuracy (bonus)
  - median per-clip latency (after warm-up)
  - peak MPS / process memory

ESC-50 is CC BY-NC. Audio is read from data/esc50/ which is gitignored.
Only the summary numbers are printed; no per-clip CSV is committed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import ClapModel, ClapProcessor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "laion/clap-htsat-unfused"
ESC_DIR = Path("data/esc50")
META_CSV = ESC_DIR / "meta" / "esc50.csv"
SAMPLE_RATE = 48000  # CLAP's expected input rate

# Memory probes
_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if DEVICE == "mps":
        # bytes
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def load_meta():
    """Return fold-1 list of (wav_path, label_name) and the 50 sorted labels."""
    import csv
    rows = []
    labels = set()
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            if int(r["fold"]) != 1:
                continue
            wav = ESC_DIR / "audio" / r["filename"]
            if not wav.exists():
                continue
            rows.append((wav, r["category"]))
            labels.add(r["category"])
    return rows, sorted(labels)


def read_audio(path: Path) -> np.ndarray:
    """Read wav → mono float32 at SAMPLE_RATE."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def main():
    rows, labels = load_meta()
    print(f"ESC-50 fold 1: {len(rows)} clips across {len(labels)} classes", flush=True)
    prompts = [f"the sound of a {lbl.replace('_', ' ')}" for lbl in labels]

    print(f"Loading {MODEL_NAME}…", flush=True)
    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    # fp32 on MPS — CLAP's BatchNorm layers need consistent fp32 numerics
    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print(f"  load={time.perf_counter()-t0:.1f}s  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB",
          flush=True)

    # Embed prompts ONCE
    t0 = time.perf_counter()
    with torch.inference_mode():
        tok = processor(text=prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        text_out = model.get_text_features(**tok)
        text_emb = text_out.pooler_output
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    print(f"  text-embed={time.perf_counter()-t0:.2f}s  ({len(labels)} prompts)",
          flush=True)

    # Warm-up on 1 clip
    wav0, _ = rows[0]
    audio0 = read_audio(wav0)
    with torch.inference_mode():
        inp = processor(audio=audio0, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        _ = model.get_audio_features(**inp)
    if DEVICE == "mps":
        torch.mps.synchronize()
    print(f"  warm-up done  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB",
          flush=True)

    # Score each clip
    per_clip_ms = []
    correct1 = 0
    correct5 = 0
    t_start = time.perf_counter()
    for i, (wav, true_lbl) in enumerate(rows):
        audio = read_audio(wav)
        with torch.inference_mode():
            t1 = time.perf_counter()
            inp = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            aud_out = model.get_audio_features(**inp)
            aud_emb = aud_out.pooler_output
            aud_emb = aud_emb / aud_emb.norm(dim=-1, keepdim=True)
            if DEVICE == "mps":
                torch.mps.synchronize()
            t2 = time.perf_counter()
            sims = (aud_emb @ text_emb.T).squeeze(0)  # (50,)
            top = torch.topk(sims, k=5).indices.cpu().tolist()
        per_clip_ms.append((t2 - t1) * 1000)
        pred1 = labels[top[0]]
        if pred1 == true_lbl:
            correct1 += 1
        if true_lbl in [labels[j] for j in top]:
            correct5 += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] med={np.median(per_clip_ms):.1f}ms  "
                  f"top1={correct1/(i+1):.3f}  cpu_rss={cpu_rss_gb():.2f}GB  "
                  f"mps={mps_gb():.2f}GB", flush=True)

    elapsed = time.perf_counter() - t_start
    n = len(rows)
    top1 = correct1 / n
    top5 = correct5 / n
    print()
    print("=" * 60)
    print(f"CLAP zero-shot on ESC-50 fold 1 ({n} clips)")
    print(f"  top-1 accuracy: {top1:.4f}  ({correct1}/{n})")
    print(f"  top-5 accuracy: {top5:.4f}  ({correct5}/{n})")
    print(f"  median per-clip latency: {np.median(per_clip_ms):.1f} ms")
    print(f"  p25 / p75: {np.percentile(per_clip_ms,25):.1f} / {np.percentile(per_clip_ms,75):.1f} ms")
    print(f"  total scoring time: {elapsed:.1f} s  ({elapsed/n*1000:.1f} ms/clip avg)")
    print(f"  peak cpu_rss: {cpu_rss_gb():.2f} GB")
    print(f"  peak mps: {mps_gb():.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()