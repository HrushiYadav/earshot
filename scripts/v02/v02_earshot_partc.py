"""v0.2 Part B extension + Part C: earshot on multiple small datasets.

Loads the Qwen2.5-Omni-3B thinker once and runs:

  1. ESC-50 fold 1 with contextual calibration (Part B extension)
  2. RAVDESS calm vs angry (Part C task 1)
  3. RAVDESS speech vs song (Part C task 2)
  4. Speech Commands stop vs other (Part C task 3)
  5. macOS `say` statement vs question (Part C task 4, SYNTHETIC)

Reports accuracy for each, plus counts, sample size, and license.

Contextual calibration: for each (clip, label), we compute
  logit(P(Yes | audio, "Is there the sound of X?"))
  - logit(P(Yes | audio, "Is there any sound at all?"))
The prior question's P(Yes) is read at the same last-non-pad position as
the label questions; the prior and label suffixes are batched together so
the prefix forward is reused. Calibration is essentially free per clip.

Datasets are CC BY-NC (RAVDESS) or CC BY (Speech Commands v0.02); the
macsay clips are synthetic. Audio stays in data/ (gitignored); only the
summary numbers are printed.
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


def read_audio(path: Path, target_sr=SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def bool_q(qid: str, text: str) -> BoolQuestion:
    return BoolQuestion(
        id=qid, type="bool", text=text,
        manifest_col=None, smoothing=0.0,
        enter=0.5, exit=0.5, source="model",
    )


def accuracy(preds, labels, pos_label):
    """Top-1 accuracy for a binary task where `pos_label` is the "yes" class."""
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds), correct, len(preds)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_esc50_fold1():
    """Return [(wav_path, label_str), ...] for ESC-50 fold 1."""
    esc_dir = Path("data/esc50")
    meta = esc_dir / "meta" / "esc50.csv"
    rows = []
    labels = set()
    with meta.open() as f:
        for r in csv.DictReader(f):
            if int(r["fold"]) != 1:
                continue
            wav = esc_dir / "audio" / r["filename"]
            if not wav.exists():
                continue
            rows.append((wav, r["category"]))
            labels.add(r["category"])
    return rows, sorted(labels)


ESC_PHRASES = {
    "dog": "a dog barking", "rooster": "a rooster crowing", "pig": "a pig snorting",
    "cow": "a cow mooing", "frog": "a frog croaking", "cat": "a cat meowing",
    "hen": "a hen clucking", "insects": "insects buzzing", "sheep": "a sheep bleating",
    "crow": "a crow cawing", "rain": "rain falling", "sea_waves": "ocean waves",
    "crackling_fire": "a fire crackling", "crickets": "crickets chirping",
    "chirping_birds": "birds chirping", "water_drops": "water dripping",
    "wind": "wind blowing", "pouring_water": "water pouring",
    "toilet_flush": "a toilet flushing", "thunderstorm": "a thunderstorm",
    "crying_baby": "a baby crying", "sneezing": "a person sneezing",
    "clapping": "people clapping", "breathing": "a person breathing",
    "coughing": "a person coughing", "footsteps": "footsteps",
    "laughing": "a person laughing", "brushing_teeth": "someone brushing their teeth",
    "snoring": "a person snoring", "drinking_sipping": "someone sipping a drink",
    "door_wood_knock": "knocking on a wooden door",
    "mouse_click": "a computer mouse clicking", "keyboard_typing": "typing on a keyboard",
    "door_wood_creaks": "a wooden door creaking", "can_opening": "a can being opened",
    "washing_machine": "a washing machine running",
    "vacuum_cleaner": "a vacuum cleaner running", "clock_alarm": "an alarm clock ringing",
    "clock_tick": "a clock ticking", "glass_breaking": "glass breaking",
    "helicopter": "a helicopter", "chainsaw": "a chainsaw running",
    "siren": "a siren wailing", "car_horn": "a car horn honking",
    "engine": "an engine running", "train": "a train",
    "church_bells": "church bells ringing", "airplane": "an airplane",
    "fireworks": "fireworks exploding", "hand_saw": "a hand saw cutting",
}


def load_ravdess_calm_angry():
    """RAVDESS speech audio, emotion 02 (calm) vs 05 (angry), all 24 actors.
    Returns [(path, 'angry'|'calm'), ...] balanced."""
    root = Path("data/ravdess/Audio_Speech_Actors_01-24")
    if not root.exists():
        return []
    rows = []
    for wav in sorted(root.glob("*/*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal, emotion = parts[0], parts[1], parts[2]
        if modality != "03" or vocal != "01":
            continue
        if emotion == "05":
            rows.append((wav, "angry"))
        elif emotion == "02":
            rows.append((wav, "calm"))
    return rows


def load_ravdess_speech_song():
    """RAVDESS audio speech vs song, both modalities 03, all actors.
    Returns [(path, 'speech'|'song'), ...] balanced."""
    speech_root = Path("data/ravdess/Audio_Speech_Actors_01-24")
    song_root = Path("data/ravdess/Audio_Song_Actors_01-24")
    rows = []
    if speech_root.exists():
        for wav in sorted(speech_root.glob("*/*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            if parts[0] != "03" or parts[1] != "01":
                continue
            rows.append((wav, "speech"))
    if song_root.exists():
        for wav in sorted(song_root.glob("*/*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            if parts[0] != "03" or parts[1] != "02":
                continue
            rows.append((wav, "song"))
    return rows


def load_speech_commands_stop_vs_other():
    """Speech Commands v0.02: stop vs a mix of {go, no, yes, up, down}.
    Returns [(path, 'stop'|'other'), ...] balanced.

    Each word folder has thousands of files; we cap at 30 per class for
    balance and to keep earshot run time bounded."""
    root = Path("data/speech_commands_v2")
    out = []
    for label, folder in [("stop", "stop"), ("other", None)]:
        if label == "stop":
            d = root / "stop"
            if not d.exists():
                continue
            files = list(d.glob("*.wav"))
            files = sorted(files)[:30]
            for f in files:
                out.append((f, label))
        else:
            # Mix from go, no, yes, up, down
            for word in ["go", "no", "yes", "up", "down"]:
                d = root / word
                if not d.exists():
                    continue
                files = sorted(d.glob("*.wav"))[:6]  # 5 words × 6 = 30
                for f in files:
                    out.append((f, label))
    return out


def load_say_statement_question():
    """macOS say synthetic clips: 10 statements + 10 questions."""
    root = Path("data/say_clips")
    rows = []
    for wav in sorted(root.glob("stmt_*.wav")):
        rows.append((wav, "statement"))
    for wav in sorted(root.glob("ques_*.wav")):
        rows.append((wav, "question"))
    return rows


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_esc50_with_calibration(rows, labels):
    print("\n=== Task: ESC-50 fold 1 with contextual calibration ===", flush=True)
    print(f"  {len(rows)} clips × {len(labels)} labels  +  1 prior per clip", flush=True)
    # Build 50 label questions + 1 prior "Is there any sound at all?"
    label_qs = [bool_q(lbl, f"Is there the sound of {ESC_PHRASES[lbl]}?") for lbl in labels]
    prior_q = bool_q("__prior__", "Is there any sound at all?")
    all_qs = label_qs + [prior_q]

    # Pre-warm with clip 0
    audio0 = read_audio(rows[0][0])
    _ = score_batched(audio0, all_qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    per_clip_ms, correct_raw, correct_cal, max_below05 = [], 0, 0, 0
    confusion_raw, confusion_cal = Counter(), Counter()
    t_start = time.perf_counter()

    for i, (wav, true_lbl) in enumerate(rows):
        try:
            audio = read_audio(wav)
            t1 = time.perf_counter()
            res = score_batched(audio, all_qs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t2 = time.perf_counter()
            per_clip_ms.append((t2 - t1) * 1000)

            p_yes = np.array([float(res[lbl].p) for lbl in labels])
            prior_p = float(res["__prior__"].p)

            # Convert to logits, subtract prior logit (contextual calibration)
            eps = 1e-6
            logit = np.log(np.clip(p_yes, eps, 1 - eps) /
                           np.clip(1 - p_yes, eps, 1 - eps))
            prior_logit = np.log(np.clip(prior_p, eps, 1 - eps) /
                                 np.clip(1 - prior_p, eps, 1 - eps))
            cal_logit = logit - prior_logit
            cal_p_yes = 1 / (1 + np.exp(-cal_logit))

            top5_raw = np.argsort(-p_yes)[:5]
            top5_cal = np.argsort(-cal_p_yes)[:5]
            pred_raw, pred_cal = labels[top5_raw[0]], labels[top5_cal[0]]

            if pred_raw == true_lbl:
                correct_raw += 1
            if pred_cal == true_lbl:
                correct_cal += 1
            if p_yes.max() < 0.5:
                max_below05 += 1
            if pred_raw != true_lbl:
                confusion_raw[(true_lbl, pred_raw)] += 1
            if pred_cal != true_lbl:
                confusion_cal[(true_lbl, pred_cal)] += 1

        except Exception as exc:
            print(f"  ! {wav.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(rows)}] med={np.median(per_clip_ms):.0f}ms  "
                  f"raw={correct_raw/(i+1):.3f}  cal={correct_cal/(i+1):.3f}  "
                  f"max<0.5={max_below05}/{i+1}  elapsed={elapsed:.0f}s  "
                  f"mps={mps_gb():.2f}GB", flush=True)

    n = len(per_clip_ms)
    print()
    print(f"  raw P(Yes) argmax:    top-1 = {correct_raw/n:.4f}  ({correct_raw}/{n})")
    print(f"  calibrated argmax:    top-1 = {correct_cal/n:.4f}  ({correct_cal}/{n})")
    print(f"  median per-clip latency: {np.median(per_clip_ms):.0f} ms")
    print(f"  clips with max P(Yes) < 0.5: {max_below05} ({max_below05/n*100:.1f}%)")
    print(f"  peak mps: {mps_gb():.2f} GB")
    print(f"\n  top-10 confusions (raw):")
    for (t, p), c in confusion_raw.most_common(10):
        print(f"    {c:3d}×  {t:20s} -> {p}")
    print(f"\n  top-10 confusions (calibrated):")
    for (t, p), c in confusion_cal.most_common(10):
        print(f"    {c:3d}×  {t:20s} -> {p}")


def run_binary(rows, pos_label, neg_label, question_text, dataset_label):
    """Run a binary earshot task and report."""
    if not rows:
        print(f"\n=== Task: {dataset_label} — SKIPPED (no clips) ===", flush=True)
        return
    print(f"\n=== Task: {dataset_label} ===", flush=True)
    print(f"  {len(rows)} clips  ({sum(1 for _,l in rows if l==pos_label)} {pos_label} / "
          f"{sum(1 for _,l in rows if l==neg_label)} {neg_label})", flush=True)
    qs = [bool_q("q", question_text)]

    audio0 = read_audio(rows[0][0])
    _ = score_batched(audio0, qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    per_clip_ms, preds, labels_out, p_yes_all = [], [], [], []
    for wav, true_lbl in rows:
        try:
            audio = read_audio(wav)
            t1 = time.perf_counter()
            res = score_batched(audio, qs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t2 = time.perf_counter()
            per_clip_ms.append((t2 - t1) * 1000)
            p = float(res["q"].p)
            p_yes_all.append(p)
            pred = pos_label if p > 0.5 else neg_label
            preds.append(pred)
            labels_out.append(true_lbl)
        except Exception as exc:
            print(f"  ! {wav.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

    n = len(preds)
    correct = sum(1 for p, l in zip(preds, labels_out) if p == l)
    pos_correct = sum(1 for p, l in zip(preds, labels_out) if p == l and l == pos_label)
    neg_correct = sum(1 for p, l in zip(preds, labels_out) if p == l and l == neg_label)
    pos_total = sum(1 for l in labels_out if l == pos_label)
    neg_total = sum(1 for l in labels_out if l == neg_label)
    baseline = max(pos_total, neg_total) / n
    print(f"  accuracy: {correct}/{n} = {correct/n:.4f}")
    print(f"    per-class: {pos_label} {pos_correct}/{pos_total} = "
          f"{pos_correct/pos_total:.3f}  |  {neg_label} {neg_correct}/{neg_total} = "
          f"{neg_correct/neg_total:.3f}")
    print(f"  majority baseline: {baseline:.4f}")
    print(f"  median P(Yes) when true={pos_label}: "
          f"{np.median([p for p,l in zip(p_yes_all,labels_out) if l==pos_label]):.3f}")
    print(f"  median P(Yes) when true={neg_label}: "
          f"{np.median([p for p,l in zip(p_yes_all,labels_out) if l==neg_label]):.3f}")
    print(f"  median per-clip latency: {np.median(per_clip_ms):.0f} ms  "
          f"(peak mps {mps_gb():.2f} GB)")


def main():
    print("earshot v0.2 — Part B extension (calibration) + Part C", flush=True)
    print(f"  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB", flush=True)

    # ESC-50 fold 1 with contextual calibration
    esc_rows, esc_labels = load_esc50_fold1()
    run_esc50_with_calibration(esc_rows, esc_labels)

    # Part C tasks — cap sample sizes for a tractable earshot run
    rng = np.random.default_rng(0)

    def balance_cap(rows, per_class_cap=192):
        """Cap each class to `per_class_cap` clips, randomly sampled but
        preserving the (wav, label) tuples."""
        from collections import defaultdict
        by_lbl = defaultdict(list)
        for w, l in rows:
            by_lbl[l].append(w)
        out = []
        for lbl, ws in by_lbl.items():
            ws = list(ws)
            if len(ws) > per_class_cap:
                idxs = rng.choice(len(ws), size=per_class_cap, replace=False)
                ws = [ws[i] for i in sorted(idxs)]
            for w in ws:
                out.append((w, lbl))
        return out

    ca = balance_cap(load_ravdess_calm_angry(), 192)
    run_binary(
        ca,
        pos_label="angry", neg_label="calm",
        question_text="Does the speaker sound angry?",
        dataset_label="RAVDESS calm vs angry (CC BY-NC-SA 4.0)",
    )
    ss = balance_cap(load_ravdess_speech_song(), 192)
    run_binary(
        ss,
        pos_label="song", neg_label="speech",
        question_text="Is the person singing rather than speaking?",
        dataset_label="RAVDESS speech vs song (CC BY-NC-SA 4.0)",
    )
    run_binary(
        load_speech_commands_stop_vs_other(),
        pos_label="stop", neg_label="other",
        question_text="Did someone say the word stop?",
        dataset_label="Speech Commands v0.02 stop vs other (CC BY 4.0)",
    )
    run_binary(
        load_say_statement_question(),
        pos_label="question", neg_label="statement",
        question_text="Is the person asking a question?",
        dataset_label="macOS `say` statement vs question (SYNTHETIC)",
    )


if __name__ == "__main__":
    main()