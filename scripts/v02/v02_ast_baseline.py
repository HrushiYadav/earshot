"""v0.2 Part A baseline: AST embeddings + sklearn logistic regression.

Trained on folds 2–5 (1600 clips), tested on fold 1 (400 clips).
Reports top-1, top-5 accuracy, median per-clip inference latency, peak memory.

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
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.linear_model import LogisticRegression

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
ESC_DIR = Path("data/esc50")
META_CSV = ESC_DIR / "meta" / "esc50.csv"
SAMPLE_RATE = 16000  # AST's expected rate

_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if DEVICE == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def load_meta():
    import csv
    by_fold = {1: [], 2: [], 3: [], 4: [], 5: []}
    labels = set()
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            fold = int(r["fold"])
            wav = ESC_DIR / "audio" / r["filename"]
            if not wav.exists():
                continue
            by_fold[fold].append((wav, r["category"]))
            labels.add(r["category"])
    return by_fold, sorted(labels)


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def main():
    by_fold, labels = load_meta()
    label_to_idx = {l: i for i, l in enumerate(labels)}
    train = by_fold[2] + by_fold[3] + by_fold[4] + by_fold[5]
    test = by_fold[1]
    print(f"train: {len(train)} clips   test: {len(test)} clips   classes: {len(labels)}",
          flush=True)

    CACHE = Path("bench/v02_ast_emb.npz")
    CACHE.parent.mkdir(exist_ok=True, parents=True)
    cache_train = CACHE.with_suffix(".train.npz")
    cache_test = CACHE.with_suffix(".test.npz")
    train_keys = [str(p) for p, _ in train]
    test_keys = [str(p) for p, _ in test]

    print(f"Loading {MODEL_NAME}…", flush=True)
    t0 = time.perf_counter()
    fex = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to(DEVICE).eval()
    print(f"  load={time.perf_counter()-t0:.1f}s  cpu_rss={cpu_rss_gb():.2f}GB  "
          f"mps={mps_gb():.2f}GB", flush=True)

    def embed_batch(items, cache_path, bs=8):
        # Try cache first
        if cache_path.exists():
            z = np.load(cache_path)
            if list(z["keys"]) == [str(p) for p, _ in items]:
                print(f"    cache hit {cache_path.name}", flush=True)
                return z["X"], z["y"]
        embs, ys, keys = [], [], []
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            audios = [read_audio(w) for w, _ in chunk]
            with torch.inference_mode():
                inp = fex(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inp = {k: v.to(DEVICE).to(torch.float16) for k, v in inp.items()}
                out = model(**inp)
                feat = out.last_hidden_state.mean(dim=1)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().float().numpy())
            ys.extend([label_to_idx[l] for _, l in chunk])
            keys.extend([str(p) for p, _ in chunk])
            if (i // bs) % 20 == 0:
                print(f"    embed {i+len(chunk)}/{len(items)}  cpu_rss={cpu_rss_gb():.2f}GB  "
                      f"mps={mps_gb():.2f}GB", flush=True)
        X = np.concatenate(embs, axis=0)
        y = np.array(ys, dtype=np.int64)
        np.savez(cache_path, X=X, y=y, keys=np.array(keys))
        return X, y

    print("Embedding train (folds 2-5)…", flush=True)
    t0 = time.perf_counter()
    Xtr, ytr = embed_batch(train, cache_train)
    print(f"  Xtr={Xtr.shape}  elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    print("Embedding test (fold 1)…", flush=True)
    t0 = time.perf_counter()
    Xte, yte = embed_batch(test, cache_test)
    print(f"  Xte={Xte.shape}  elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    print("Training logistic regression…", flush=True)
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, solver="lbfgs")
    clf.fit(Xtr, ytr)
    print(f"  fit={time.perf_counter()-t0:.1f}s", flush=True)

    # Score per-clip for latency
    correct1 = 0
    correct5 = 0
    per_clip_ms = []
    t_start = time.perf_counter()
    for i in range(0, len(test), 8):
        chunk = test[i:i + 8]
        audios = [read_audio(w) for w, _ in chunk]
        with torch.inference_mode():
            t1 = time.perf_counter()
            inp = fex(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inp = {k: v.to(DEVICE).to(torch.float16) for k, v in inp.items()}
            out = model(**inp)
            feat = out.last_hidden_state.mean(dim=1)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            if DEVICE == "mps":
                torch.mps.synchronize()
            t2 = time.perf_counter()
        Xb = feat.cpu().float().numpy()
        # Distribute latency across clips in the batch (approximate)
        clip_ms = (t2 - t1) * 1000 / len(chunk)
        per_clip_ms.extend([clip_ms] * len(chunk))
        proba = clf.predict_proba(Xb)
        top = np.argsort(-proba, axis=1)[:, :5]
        for j, (_, lbl) in enumerate(chunk):
            pred1 = labels[top[j, 0]]
            if pred1 == lbl:
                correct1 += 1
            if lbl in [labels[k] for k in top[j]]:
                correct5 += 1

    elapsed = time.perf_counter() - t_start
    n = len(test)
    top1 = correct1 / n
    top5 = correct5 / n
    print()
    print("=" * 60)
    print(f"AST embeddings + LR on ESC-50 fold 1 ({n} clips)")
    print(f"  top-1 accuracy: {top1:.4f}  ({correct1}/{n})")
    print(f"  top-5 accuracy: {top5:.4f}  ({correct5}/{n})")
    print(f"  median per-clip inference latency: {np.median(per_clip_ms):.1f} ms "
          f"(batch-of-8 estimate)")
    print(f"  p25 / p75: {np.percentile(per_clip_ms,25):.1f} / "
          f"{np.percentile(per_clip_ms,75):.1f} ms")
    print(f"  total scoring time: {elapsed:.1f} s")
    print(f"  peak cpu_rss: {cpu_rss_gb():.2f} GB")
    print(f"  peak mps: {mps_gb():.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()