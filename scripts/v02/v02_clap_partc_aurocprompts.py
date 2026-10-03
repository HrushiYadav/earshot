"""v0.2 Part C redo: CLAP with 3 prompt pairs per task + AUROC.

Saves per-clip sims to bench/v02_clap_<task>_<pair>.npz so the analysis step
can compute accuracy (min/max over the 3 pairs) and AUROC for each task.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import ClapModel, ClapProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_earshot_save_all import (  # noqa: E402
    load_ravdess_calm_angry, load_ravdess_speech_song,
    load_speech_commands_stop_vs_other, load_say_statement_question,
    read_audio, balance_cap,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000
SAVE_DIR = Path("bench")

_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if DEVICE == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


# 3 prompt pairs per task
TASK_PROMPTS = {
    "ravdess_calm_angry": [
        ("an angry voice", "a calm voice"),
        ("a person speaking with anger", "a person speaking calmly"),
        ("an aggressive tone of voice", "a peaceful tone of voice"),
    ],
    "ravdess_speech_song": [
        ("a person singing", "a person speaking"),
        ("someone singing a song", "someone speaking words"),
        ("a singer performing", "a person talking"),
    ],
    "sc_stop_other": [
        ("a person saying the word stop", "a person saying a different word"),
        ("someone says stop", "someone says a word"),
        ("the word stop", "a spoken word"),
    ],
    "say_q_stmt": [
        ("a person asking a question", "a person making a statement"),
        ("a spoken question", "a spoken statement"),
        ("someone asking a question", "someone making a statement"),
    ],
}


def embed_text(model, processor, prompts):
    with torch.inference_mode():
        tok = processor(text=prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        out = model.get_text_features(**tok)
        emb = out.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def run_task(model, processor, rows, task_name, cap):
    rows = balance_cap(rows, cap)
    pos_label, neg_label = (("angry", "calm") if task_name == "ravdess_calm_angry"
                            else ("song", "speech") if task_name == "ravdess_speech_song"
                            else ("stop", "other") if task_name == "sc_stop_other"
                            else ("question", "statement"))
    true_bin = np.array([1 if l == pos_label else 0 for _, l in rows], dtype=np.int64)
    n = len(rows)
    n_pos = int(true_bin.sum())
    n_neg = n - n_pos
    print(f"\n=== CLAP × {task_name}: {n} clips ({n_pos} {pos_label} / "
          f"{n_neg} {neg_label}) ===", flush=True)

    audios = [read_audio(w) for w, _ in rows]

    # Warm
    _ = embed_text(model, processor, TASK_PROMPTS[task_name][0])
    t_start = time.perf_counter()
    for pair_idx, (pos, neg) in enumerate(TASK_PROMPTS[task_name]):
        text_emb = embed_text(model, processor, [neg, pos])
        sims = np.zeros(n, dtype=np.float32)
        for i, audio in enumerate(audios):
            with torch.inference_mode():
                inp = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                out = model.get_audio_features(**inp)
                aud_emb = out.pooler_output
                aud_emb = aud_emb / aud_emb.norm(dim=-1, keepdim=True)
                if DEVICE == "mps":
                    torch.mps.synchronize()
            s_neg = float((aud_emb @ text_emb[0]).item())
            s_pos = float((aud_emb @ text_emb[1]).item())
            sims[i] = s_pos - s_neg
        out = SAVE_DIR / f"v02_clap_{task_name}_pair{pair_idx}.npz"
        np.savez(out, sims=sims, true_bin=true_bin,
                 pos_label=pos_label, neg_label=neg_label,
                 pos_prompt=pos, neg_prompt=neg, n=n)
        # quick accuracy
        preds = (sims > 0).astype(np.int64)
        acc = float((preds == true_bin).mean())
        print(f"  pair {pair_idx}: {neg!r}  vs  {pos!r}  acc={acc:.4f}", flush=True)
    print(f"  total time: {time.perf_counter()-t_start:.1f}s")


def main():
    print(f"CLAP — Part C redo (3 prompt pairs, AUROC)")
    print(f"  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB")
    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print(f"  load={time.perf_counter()-t0:.1f}s")
    run_task(model, processor, load_ravdess_calm_angry(),
             "ravdess_calm_angry", 192)
    run_task(model, processor, load_ravdess_speech_song(),
             "ravdess_speech_song", 192)
    run_task(model, processor, load_speech_commands_stop_vs_other(),
             "sc_stop_other", 250)
    run_task(model, processor, load_say_statement_question(),
             "say_q_stmt", 10)
    print(f"\npeak mps={mps_gb():.2f}GB  cpu_rss={cpu_rss_gb():.2f}GB")


if __name__ == "__main__":
    main()