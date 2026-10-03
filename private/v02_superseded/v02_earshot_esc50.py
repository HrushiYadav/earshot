"""v0.2 Part B: earshot (Qwen2.5-Omni-3B thinker) on ESC-50 fold 1.

For each of the 400 fold-1 clips, build 50 BoolQuestion objects using the
readable phrases, run them in ONE batched call to `score_batched`, and take
argmax P(Yes) as the prediction.

Reports:
  - top-1 / top-5 accuracy
  - median per-clip latency (warm)
  - peak memory (cpu_rss + MPS)
  - total time
  - how often max P(Yes) < 0.5 ("model says No to everything")
  - 10 most-confused (true_class, pred_class) pairs

Any clip that throws is logged to bench/v02_earshot_failures.csv and the
loop continues — partial results are still useful.

ESC-50 is CC BY-NC. Audio is read from data/esc50/ which is gitignored.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from earshot.scorer import score_batched  # noqa: E402
from earshot.schema import BoolQuestion  # noqa: E402

ESC_DIR = Path("data/esc50")
META_CSV = ESC_DIR / "meta" / "esc50.csv"
SAMPLE_RATE = 16000
FAIL_LOG = Path("bench/v02_earshot_failures.csv")

_proc = psutil.Process(os.getpid())


def cpu_rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3


def mps_gb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


LABEL_PHRASES = {
    "dog": "a dog barking",
    "rooster": "a rooster crowing",
    "pig": "a pig snorting",
    "cow": "a cow mooing",
    "frog": "a frog croaking",
    "cat": "a cat meowing",
    "hen": "a hen clucking",
    "insects": "insects buzzing",
    "sheep": "a sheep bleating",
    "crow": "a crow cawing",
    "rain": "rain falling",
    "sea_waves": "ocean waves",
    "crackling_fire": "a fire crackling",
    "crickets": "crickets chirping",
    "chirping_birds": "birds chirping",
    "water_drops": "water dripping",
    "wind": "wind blowing",
    "pouring_water": "water pouring",
    "toilet_flush": "a toilet flushing",
    "thunderstorm": "a thunderstorm",
    "crying_baby": "a baby crying",
    "sneezing": "a person sneezing",
    "clapping": "people clapping",
    "breathing": "a person breathing",
    "coughing": "a person coughing",
    "footsteps": "footsteps",
    "laughing": "a person laughing",
    "brushing_teeth": "someone brushing their teeth",
    "snoring": "a person snoring",
    "drinking_sipping": "someone sipping a drink",
    "door_wood_knock": "knocking on a wooden door",
    "mouse_click": "a computer mouse clicking",
    "keyboard_typing": "typing on a keyboard",
    "door_wood_creaks": "a wooden door creaking",
    "can_opening": "a can being opened",
    "washing_machine": "a washing machine running",
    "vacuum_cleaner": "a vacuum cleaner running",
    "clock_alarm": "an alarm clock ringing",
    "clock_tick": "a clock ticking",
    "glass_breaking": "glass breaking",
    "helicopter": "a helicopter",
    "chainsaw": "a chainsaw running",
    "siren": "a siren wailing",
    "car_horn": "a car horn honking",
    "engine": "an engine running",
    "train": "a train",
    "church_bells": "church bells ringing",
    "airplane": "an airplane",
    "fireworks": "fireworks exploding",
    "hand_saw": "a hand saw cutting",
}


def load_meta():
    rows, labels = [], set()
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
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def make_questions(labels):
    """50 BoolQuestion objects in the same order as `labels`.
    The id is the ESC-50 category so we can map back from the scorer dict."""
    return [
        BoolQuestion(
            id=lbl,
            type="bool",
            text=f"Is there the sound of {LABEL_PHRASES[lbl]}?",
            manifest_col=None,
            smoothing=0.0,   # no smoothing for a single-shot eval
            enter=0.5,
            exit=0.5,
            source="model",
        )
        for lbl in labels
    ]


def main():
    rows, labels = load_meta()
    assert set(LABEL_PHRASES) == set(labels), f"label mismatch: {set(LABEL_PHRASES) ^ set(labels)}"
    print(f"ESC-50 fold 1: {len(rows)} clips × {len(labels)} bool questions = "
          f"{len(rows)*len(labels)} model calls (batched 50/clips)", flush=True)
    print(f"  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB", flush=True)

    # Force model load by warming up on clip 0 (will populate _ensure_loaded cache).
    wav0, _ = rows[0]
    audio0 = read_audio(wav0)
    qs = make_questions(labels)
    print("Loading Qwen2.5-Omni-3B thinker + warm-up on clip 0…", flush=True)
    t0 = time.perf_counter()
    _ = score_batched(audio0, qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    warm = time.perf_counter() - t0
    print(f"  warm-up done in {warm:.1f}s  cpu_rss={cpu_rss_gb():.2f}GB  "
          f"mps={mps_gb():.2f}GB", flush=True)

    # Run the loop
    per_clip_ms = []
    correct1 = 0
    correct5 = 0
    max_below_05 = 0
    confusion: Counter = Counter()
    failures: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    for i, (wav, true_lbl) in enumerate(rows):
        try:
            audio = read_audio(wav)
            t1 = time.perf_counter()
            res = score_batched(audio, qs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t2 = time.perf_counter()
            per_clip_ms.append((t2 - t1) * 1000)

            # P(Yes) per question, in label order
            p_yes = np.array([float(res[lbl].p) for lbl in labels])
            top5 = np.argsort(-p_yes)[:5]
            pred1 = labels[top5[0]]
            if pred1 == true_lbl:
                correct1 += 1
            if true_lbl in [labels[j] for j in top5]:
                correct5 += 1
            if p_yes.max() < 0.5:
                max_below_05 += 1
            if pred1 != true_lbl:
                confusion[(true_lbl, pred1)] += 1

        except Exception as exc:
            failures.append((str(wav), f"{type(exc).__name__}: {exc}"))
            print(f"  ! {wav.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] med={np.median(per_clip_ms):.1f}ms  "
                  f"top1={correct1/(i+1):.3f}  top5={correct5/(i+1):.3f}  "
                  f"max<0.5={max_below_05}/{i+1}  cpu_rss={cpu_rss_gb():.2f}GB  "
                  f"mps={mps_gb():.2f}GB", flush=True)

    elapsed = time.perf_counter() - t_start
    n = len(per_clip_ms)
    print()
    print("=" * 60)
    print(f"earshot (Qwen2.5-Omni-3B thinker) on ESC-50 fold 1 ({n} clips, "
          f"{len(failures)} failures)")
    if n:
        print(f"  top-1 accuracy: {correct1/(n+len(failures)):.4f}  ({correct1}/{n+len(failures)})")
        print(f"  top-5 accuracy: {correct5/(n+len(failures)):.4f}  ({correct5}/{n+len(failures)})")
        print(f"  median per-clip latency (50 qs, batched): {np.median(per_clip_ms):.1f} ms")
        print(f"  p25 / p75: {np.percentile(per_clip_ms,25):.1f} / "
              f"{np.percentile(per_clip_ms,75):.1f} ms")
        print(f"  total scoring time: {elapsed:.1f} s  "
              f"({elapsed/(n+len(failures))*1000:.1f} ms/clip avg)")
    print(f"  clips where max P(Yes) < 0.5: {max_below_05} "
          f"({max_below_05/(n+len(failures))*100:.1f}%)")
    print(f"  peak cpu_rss: {cpu_rss_gb():.2f} GB")
    print(f"  peak mps: {mps_gb():.2f} GB")

    print("\nTop-10 most-confused (true_class, pred_class) pairs:")
    for (t, p), c in confusion.most_common(10):
        print(f"  {c:3d}×  {t:20s} -> {p}")

    if failures:
        FAIL_LOG.parent.mkdir(exist_ok=True, parents=True)
        with FAIL_LOG.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "error"])
            for path, err in failures:
                w.writerow([path, err])
        print(f"\n{len(failures)} failures logged to {FAIL_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()