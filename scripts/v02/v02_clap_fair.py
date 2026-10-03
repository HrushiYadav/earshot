"""v0.2 Part A fairness: CLAP with readable phrases (matches earshot wording).

Same model, same fold 1 (400 clips), but prompts use the richer readable
phrases instead of the raw snake_case labels. Reports both versions side by
side so we can isolate the prompt-style effect from the model effect.
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
SAMPLE_RATE = 48000

_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if DEVICE == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


# (label, readable phrase) — kept in the same order as v02_clap_baseline.py
LABEL_PHRASES = [
    ("dog", "a dog barking"),
    ("rooster", "a rooster crowing"),
    ("pig", "a pig snorting"),
    ("cow", "a cow mooing"),
    ("frog", "a frog croaking"),
    ("cat", "a cat meowing"),
    ("hen", "a hen clucking"),
    ("insects", "insects buzzing"),
    ("sheep", "a sheep bleating"),
    ("crow", "a crow cawing"),
    ("rain", "rain falling"),
    ("sea_waves", "ocean waves"),
    ("crackling_fire", "a fire crackling"),
    ("crickets", "crickets chirping"),
    ("chirping_birds", "birds chirping"),
    ("water_drops", "water dripping"),
    ("wind", "wind blowing"),
    ("pouring_water", "water pouring"),
    ("toilet_flush", "a toilet flushing"),
    ("thunderstorm", "a thunderstorm"),
    ("crying_baby", "a baby crying"),
    ("sneezing", "a person sneezing"),
    ("clapping", "people clapping"),
    ("breathing", "a person breathing"),
    ("coughing", "a person coughing"),
    ("footsteps", "footsteps"),
    ("laughing", "a person laughing"),
    ("brushing_teeth", "someone brushing their teeth"),
    ("snoring", "a person snoring"),
    ("drinking_sipping", "someone sipping a drink"),
    ("door_wood_knock", "knocking on a wooden door"),
    ("mouse_click", "a computer mouse clicking"),
    ("keyboard_typing", "typing on a keyboard"),
    ("door_wood_creaks", "a wooden door creaking"),
    ("can_opening", "a can being opened"),
    ("washing_machine", "a washing machine running"),
    ("vacuum_cleaner", "a vacuum cleaner running"),
    ("clock_alarm", "an alarm clock ringing"),
    ("clock_tick", "a clock ticking"),
    ("glass_breaking", "glass breaking"),
    ("helicopter", "a helicopter"),
    ("chainsaw", "a chainsaw running"),
    ("siren", "a siren wailing"),
    ("car_horn", "a car horn honking"),
    ("engine", "an engine running"),
    ("train", "a train"),
    ("church_bells", "church bells ringing"),
    ("airplane", "an airplane"),
    ("fireworks", "fireworks exploding"),
    ("hand_saw", "a hand saw cutting"),
]


def load_meta():
    import csv as csvmod
    rows, labels = [], set()
    with META_CSV.open() as f:
        for r in csvmod.DictReader(f):
            if int(r["fold"]) != 1:
                continue
            wav = ESC_DIR / "audio" / r["filename"]
            if not wav.exists():
                continue
            rows.append((wav, r["category"]))
            labels.add(r["category"])
    return rows, sorted(labels)


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def score_with_prompts(model, processor, rows, labels, prompts, label_to_idx):
    # Warm-up on clip 0
    wav0, _ = rows[0]
    audio0 = read_audio(wav0)
    with torch.inference_mode():
        inp = processor(audio=audio0, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        _ = model.get_audio_features(**inp)
    if DEVICE == "mps":
        torch.mps.synchronize()

    per_clip_ms, correct1, correct5 = [], 0, 0
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
            sims = (aud_emb @ prompts.T).squeeze(0)
            top = torch.topk(sims, k=5).indices.cpu().tolist()
        per_clip_ms.append((t2 - t1) * 1000)
        pred1 = labels[top[0]]
        if pred1 == true_lbl:
            correct1 += 1
        if true_lbl in [labels[j] for j in top]:
            correct5 += 1
    return correct1, correct5, per_clip_ms


def main():
    rows, labels = load_meta()
    phrase_by_label = dict(LABEL_PHRASES)
    assert set(phrase_by_label) == set(labels), f"label set mismatch: " \
        f"{set(phrase_by_label) ^ set(labels)}"
    print(f"ESC-50 fold 1: {len(rows)} clips across {len(labels)} classes", flush=True)

    print(f"Loading {MODEL_NAME}…", flush=True)
    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    print(f"  load={time.perf_counter()-t0:.1f}s  cpu_rss={cpu_rss_gb():.2f}GB  "
          f"mps={mps_gb():.2f}GB", flush=True)

    # ---- variant A: raw-label prompts ----
    raw_prompts = [f"the sound of a {lbl.replace('_', ' ')}" for lbl in labels]
    with torch.inference_mode():
        tok = processor(text=raw_prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        tout = model.get_text_features(**tok)
        raw_text_emb = tout.pooler_output
        raw_text_emb = raw_text_emb / raw_text_emb.norm(dim=-1, keepdim=True)

    print("\n--- variant A: raw labels (baseline) ---", flush=True)
    c1, c5, ms = score_with_prompts(model, processor, rows, labels, raw_text_emb, None)
    n = len(rows)
    print(f"  top-1={c1/n:.4f}  top-5={c5/n:.4f}  med={np.median(ms):.1f}ms")

    # ---- variant B: readable-phrase prompts ----
    phrase_prompts = [f"the sound of {phrase_by_label[l]}" for l in labels]
    with torch.inference_mode():
        tok = processor(text=phrase_prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        tout = model.get_text_features(**tok)
        phrase_text_emb = tout.pooler_output
        phrase_text_emb = phrase_text_emb / phrase_text_emb.norm(dim=-1, keepdim=True)

    print("\n--- variant B: readable phrases (fair vs earshot) ---", flush=True)
    c1, c5, ms = score_with_prompts(model, processor, rows, labels, phrase_text_emb, None)
    print(f"  top-1={c1/n:.4f}  top-5={c5/n:.4f}  med={np.median(ms):.1f}ms")
    print(f"  peak cpu_rss={cpu_rss_gb():.2f}GB  peak mps={mps_gb():.2f}GB")


if __name__ == "__main__":
    main()