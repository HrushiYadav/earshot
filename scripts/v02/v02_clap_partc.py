"""v0.2 Part C: CLAP zero-shot on the same 4 tasks as earshot.

For each task, the two prompts are argmax'd against audio-text similarity.
Also reports the always-one-class baseline (50%).

Datasets:
  - RAVDESS speech (CC BY-NC-SA 4.0): Zenodo 1188976
  - Speech Commands v0.02 (CC BY 4.0): storage.googleapis.com/.../speech_commands_v0.02.tar.gz
  - macOS `say` clips: SYNTHETIC, generated locally

Audio is gitignored (data/).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import ClapModel, ClapProcessor

# Reuse the earshot part C loaders
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_earshot_partc import (  # noqa: E402
    load_ravdess_calm_angry, load_ravdess_speech_song,
    load_speech_commands_stop_vs_other, load_say_statement_question,
    read_audio,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000

_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if DEVICE == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def balance_cap(rows, per_class_cap=192, seed=0):
    from collections import defaultdict
    by_lbl = defaultdict(list)
    for w, l in rows:
        by_lbl[l].append(w)
    rng = np.random.default_rng(seed)
    out = []
    for lbl, ws in by_lbl.items():
        ws = list(ws)
        if len(ws) > per_class_cap:
            idxs = rng.choice(len(ws), size=per_class_cap, replace=False)
            ws = [ws[i] for i in sorted(idxs)]
        for w in ws:
            out.append((w, lbl))
    return out


def embed_text(model, processor, prompts):
    with torch.inference_mode():
        tok = processor(text=prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        out = model.get_text_features(**tok)
        emb = out.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def embed_audio(model, processor, audio):
    with torch.inference_mode():
        inp = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        out = model.get_audio_features(**inp)
        emb = out.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        if DEVICE == "mps":
            torch.mps.synchronize()
    return emb


def run_binary_clap(model, processor, rows, pos_label, neg_label,
                    prompt_pos, prompt_neg, dataset_label):
    if not rows:
        print(f"\n=== Task: {dataset_label} — SKIPPED (no clips) ===")
        return
    print(f"\n=== Task: {dataset_label} ===")
    print(f"  {len(rows)} clips  ({sum(1 for _,l in rows if l==pos_label)} {pos_label} / "
          f"{sum(1 for _,l in rows if l==neg_label)} {neg_label})")
    print(f"  prompts: {prompt_neg!r}  vs  {prompt_pos!r}")
    text_emb = embed_text(model, processor, [prompt_neg, prompt_pos])
    # Warm
    _ = embed_audio(model, processor, read_audio(rows[0][0]))

    correct, total = 0, 0
    pos_correct, pos_total = 0, 0
    neg_correct, neg_total = 0, 0
    per_clip_ms = []
    t_start = time.perf_counter()
    for wav, true_lbl in rows:
        audio = read_audio(wav)
        t1 = time.perf_counter()
        aud_emb = embed_audio(model, processor, audio)
        sims = (aud_emb @ text_emb.T).squeeze(0)
        pred = pos_label if float(sims[1]) > float(sims[0]) else neg_label
        per_clip_ms.append((time.perf_counter() - t1) * 1000)
        total += 1
        if pred == true_lbl:
            correct += 1
        if true_lbl == pos_label:
            pos_total += 1
            if pred == true_lbl:
                pos_correct += 1
        else:
            neg_total += 1
            if pred == true_lbl:
                neg_correct += 1
    elapsed = time.perf_counter() - t_start
    baseline = max(pos_total, neg_total) / total
    print(f"  accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"    per-class: {pos_label} {pos_correct}/{pos_total} = "
          f"{pos_correct/pos_total:.3f}  |  {neg_label} {neg_correct}/{neg_total} = "
          f"{neg_correct/neg_total:.3f}")
    print(f"  majority baseline: {baseline:.4f}")
    print(f"  median per-clip latency: {np.median(per_clip_ms):.1f} ms")
    print(f"  total scoring time: {elapsed:.1f} s")


def main():
    print(f"CLAP zero-shot on Part C tasks  cpu_rss={cpu_rss_gb():.2f}GB  "
          f"mps={mps_gb():.2f}GB")
    print(f"Loading {MODEL_NAME}…")
    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print(f"  load={time.perf_counter()-t0:.1f}s")

    run_binary_clap(
        model, processor,
        balance_cap(load_ravdess_calm_angry(), 192),
        pos_label="angry", neg_label="calm",
        prompt_pos="an angry voice", prompt_neg="a calm voice",
        dataset_label="RAVDESS calm vs angry (CC BY-NC-SA 4.0)",
    )
    run_binary_clap(
        model, processor,
        balance_cap(load_ravdess_speech_song(), 192),
        pos_label="song", neg_label="speech",
        prompt_pos="a person singing", prompt_neg="a person speaking",
        dataset_label="RAVDESS speech vs song (CC BY-NC-SA 4.0)",
    )
    run_binary_clap(
        model, processor,
        load_speech_commands_stop_vs_other(),
        pos_label="stop", neg_label="other",
        prompt_pos="a person saying the word stop",
        prompt_neg="a person saying a word",
        dataset_label="Speech Commands v0.02 stop vs other (CC BY 4.0)",
    )
    run_binary_clap(
        model, processor,
        load_say_statement_question(),
        pos_label="question", neg_label="statement",
        prompt_pos="a person asking a question",
        prompt_neg="a person making a statement",
        dataset_label="macOS `say` statement vs question (SYNTHETIC)",
    )
    print(f"\npeak mps={mps_gb():.2f} GB  cpu_rss={cpu_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()