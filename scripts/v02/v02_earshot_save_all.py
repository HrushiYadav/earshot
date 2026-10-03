"""v0.2 redo: earshot on ESC-50 + 3 neutral inputs (for per-question bias)
+ 4 Part C tasks. Saves raw P(Yes) to disk for off-line calibration and
AUROC analysis.

Per-question bias for calibration is computed from 3 neutral inputs:
  - silence_1 (clips/silence_1.wav)
  - low white noise (5 s, RMS ~ 1e-4, generated)
  - room_noise_1 (clips/room_noise_1.wav)
Bias_q = mean over the 3 neutrals of P(Yes | audio, "Is there the sound of X_q?")
The calibrated score is logit(P(Yes)) − logit(bias_q).

AUROC analysis is done in v02_analyze.py (pure Python, no model).
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from earshot.scorer import score_batched  # noqa: E402
from earshot.schema import BoolQuestion  # noqa: E402

SAMPLE_RATE = 16000
SAVE_DIR = Path("bench")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

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


# ---------------------------------------------------------------------------
# ESC-50 fold 1
# ---------------------------------------------------------------------------

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


def load_esc50_fold1():
    esc_dir = Path("data/esc50")
    meta = esc_dir / "meta" / "esc50.csv"
    rows, labels = [], set()
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


def run_esc50_save(rows, labels):
    print(f"\n=== ESC-50 fold 1 — save P(Yes) ===", flush=True)
    print(f"  {len(rows)} clips × {len(labels)} labels", flush=True)
    label_qs = [bool_q(lbl, f"Is there the sound of {ESC_PHRASES[lbl]}?") for lbl in labels]
    audio0 = read_audio(rows[0][0])
    _ = score_batched(audio0, label_qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    print(f"  warmup done  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB", flush=True)

    P = np.zeros((len(rows), len(labels)), dtype=np.float32)
    true_idx = np.array([labels.index(lbl) for _, lbl in rows], dtype=np.int64)
    t_start = time.perf_counter()
    for i, (wav, true_lbl) in enumerate(rows):
        try:
            audio = read_audio(wav)
            res = score_batched(audio, label_qs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            for j, lbl in enumerate(labels):
                P[i, j] = float(res[lbl].p)
        except Exception as exc:
            print(f"  ! {wav.name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            med = elapsed / (i + 1) * 1000
            print(f"  [{i+1}/{len(rows)}] med={med:.0f}ms  elapsed={elapsed:.0f}s  "
                  f"mps={mps_gb():.2f}GB", flush=True)

    out = SAVE_DIR / "v02_earshot_esc50.npz"
    np.savez(out, P_yes=P, true_idx=true_idx, labels=np.array(labels))
    print(f"  saved {out}  P.shape={P.shape}", flush=True)


# ---------------------------------------------------------------------------
# Neutral inputs (per-question bias)
# ---------------------------------------------------------------------------

def build_neutral_inputs():
    """Return list of (name, np.ndarray) for 3 neutral audio inputs."""
    inputs = []
    # 1) silence — read from disk to be sure
    sil = read_audio(Path("clips/silence_1.wav"))
    # Trim/pad to 5s
    target = 5 * SAMPLE_RATE
    if len(sil) > target:
        sil = sil[:target]
    elif len(sil) < target:
        sil = np.concatenate([sil, np.zeros(target - len(sil), dtype=np.float32)])
    inputs.append(("silence", sil))
    # 2) low white noise — RMS ~ 1e-4
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(target).astype(np.float32) * 1e-4
    inputs.append(("low_white_noise", noise))
    # 3) room_noise_1
    rn = read_audio(Path("clips/room_noise_1.wav"))
    if len(rn) > target:
        rn = rn[:target]
    elif len(rn) < target:
        rn = np.concatenate([rn, np.zeros(target - len(rn), dtype=np.float32)])
    inputs.append(("room_noise_1", rn))
    return inputs


def run_neutral_bias_save(labels):
    print(f"\n=== Neutral inputs for per-question bias ===", flush=True)
    inputs = build_neutral_inputs()
    for name, audio in inputs:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"  {name}: shape={audio.shape}  rms={rms:.6f}")
    label_qs = [bool_q(lbl, f"Is there the sound of {ESC_PHRASES[lbl]}?") for lbl in labels]
    # warmup
    _ = score_batched(inputs[0][1], label_qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    bias = np.zeros((len(inputs), len(labels)), dtype=np.float32)
    for k, (name, audio) in enumerate(inputs):
        res = score_batched(audio, label_qs)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        for j, lbl in enumerate(labels):
            bias[k, j] = float(res[lbl].p)
    out = SAVE_DIR / "v02_earshot_bias.npz"
    np.savez(out, bias=bias,
             names=np.array([n for n, _ in inputs]),
             labels=np.array(labels))
    print(f"  saved {out}  bias.shape={bias.shape}")
    print("  per-question mean bias (over 3 neutrals):")
    sorted_idx = np.argsort(bias.mean(axis=0))[::-1]
    for j in sorted_idx[:8]:
        print(f"    {labels[j]:20s}  mean={bias.mean(axis=0)[j]:.3f}  "
              f"min={bias.min(axis=0)[j]:.3f}  max={bias.max(axis=0)[j]:.3f}")


# ---------------------------------------------------------------------------
# Part C tasks
# ---------------------------------------------------------------------------

def load_ravdess_calm_angry():
    root = Path("data/ravdess/Audio_Speech_Actors_01-24")
    if not root.exists():
        return []
    rows = []
    for wav in sorted(root.glob("*/*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        if parts[0] != "03" or parts[1] != "01":
            continue
        if parts[2] == "05":
            rows.append((wav, "angry"))
        elif parts[2] == "02":
            rows.append((wav, "calm"))
    return rows


def load_ravdess_speech_song():
    rows = []
    sp = Path("data/ravdess/Audio_Speech_Actors_01-24")
    sg = Path("data/ravdess/Audio_Song_Actors_01-24")
    if sp.exists():
        for wav in sorted(sp.glob("*/*.wav")):
            p = wav.stem.split("-")
            if len(p) == 7 and p[0] == "03" and p[1] == "01":
                rows.append((wav, "speech"))
    if sg.exists():
        for wav in sorted(sg.glob("*/*.wav")):
            p = wav.stem.split("-")
            if len(p) == 7 and p[0] == "03" and p[1] == "02":
                rows.append((wav, "song"))
    return rows


def load_speech_commands_stop_vs_other(cap_per_word=50, words=("go", "no", "yes", "up", "down")):
    root = Path("data/speech_commands_v2")
    rows = []
    d = root / "stop"
    if d.exists():
        for f in sorted(d.glob("*.wav"))[:cap_per_word * 5]:  # enough; we sub-sample
            rows.append((f, "stop"))
    for w in words:
        d = root / w
        if d.exists():
            for f in sorted(d.glob("*.wav"))[:cap_per_word]:
                rows.append((f, "other"))
    return rows


def load_say_statement_question():
    root = Path("data/say_clips")
    rows = []
    for wav in sorted(root.glob("stmt_*.wav")):
        rows.append((wav, "statement"))
    for wav in sorted(root.glob("ques_*.wav")):
        rows.append((wav, "question"))
    return rows


def balance_cap(rows, per_class_cap, seed=0):
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


def run_binary_save(rows, pos_label, neg_label, question_text, task_name, cap=192):
    if not rows:
        print(f"\n=== {task_name} — SKIPPED (no clips) ===")
        return
    rows = balance_cap(rows, cap)
    pos = sum(1 for _, l in rows if l == pos_label)
    neg = sum(1 for _, l in rows if l == neg_label)
    print(f"\n=== {task_name} ===", flush=True)
    print(f"  {len(rows)} clips ({pos} {pos_label} / {neg} {neg_label})", flush=True)
    qs = [bool_q("q", question_text)]
    audio0 = read_audio(rows[0][0])
    _ = score_batched(audio0, qs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    p_yes = np.zeros(len(rows), dtype=np.float32)
    true_bin = np.array([1 if l == pos_label else 0 for _, l in rows], dtype=np.int64)
    t_start = time.perf_counter()
    for i, (wav, _) in enumerate(rows):
        try:
            audio = read_audio(wav)
            t1 = time.perf_counter()
            res = score_batched(audio, qs)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            p_yes[i] = float(res["q"].p)
        except Exception as exc:
            print(f"  ! {wav.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
    elapsed = time.perf_counter() - t_start
    out = SAVE_DIR / f"v02_earshot_{task_name}.npz"
    np.savez(out, p_yes=p_yes, true_bin=true_bin,
             pos_label=pos_label, neg_label=neg_label,
             question=question_text, n=len(rows))
    med = elapsed / len(rows) * 1000
    print(f"  saved {out}  med={med:.0f}ms/clip  total={elapsed:.0f}s")


def main():
    print(f"earshot save-all  cpu_rss={cpu_rss_gb():.2f}GB  mps={mps_gb():.2f}GB")
    esc_rows, esc_labels = load_esc50_fold1()
    run_esc50_save(esc_rows, esc_labels)
    run_neutral_bias_save(esc_labels)
    # Part C — cap Speech Commands at 250 per class to hit the user's 500-clip total
    run_binary_save(
        load_ravdess_calm_angry(),
        pos_label="angry", neg_label="calm",
        question_text="Does the speaker sound angry?",
        task_name="ravdess_calm_angry", cap=192,
    )
    run_binary_save(
        load_ravdess_speech_song(),
        pos_label="song", neg_label="speech",
        question_text="Is the person singing rather than speaking?",
        task_name="ravdess_speech_song", cap=192,
    )
    run_binary_save(
        load_speech_commands_stop_vs_other(),
        pos_label="stop", neg_label="other",
        question_text="Did someone say the word stop?",
        task_name="sc_stop_other", cap=250,
    )
    run_binary_save(
        load_say_statement_question(),
        pos_label="question", neg_label="statement",
        question_text="Is the person asking a question?",
        task_name="say_q_stmt", cap=10,
    )
    print(f"\npeak mps={mps_gb():.2f}GB  cpu_rss={cpu_rss_gb():.2f}GB")


if __name__ == "__main__":
    main()