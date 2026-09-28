"""Stage 0 follow-up — bench the thinker's per-pass cost on a 3 s clip.

Loads Qwen2.5-Omni-3B (thinker-only) once, does 1 warm-up pass, then 5 timed
passes. Each pass = feature extraction + single-question forward pass (logits
only, no generation). Prints per-pass timings and the median of each.

Audio kwargs truncate input_features to the real audio length (48000 samples
@ 16 kHz) so the padded 30 s default never reaches the GPU.
"""
from __future__ import annotations

import os, statistics, time, traceback
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
CLIP = _REPO_ROOT / "clips" / "voice_user_3s.wav"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
QUESTION = "Is someone speaking?"
AUDIO_SR = 16000
N_WARMUP = 1
N_TIMED = 5


def load_audio_16k_mono(path: Path, max_seconds: float = 3.0) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != AUDIO_SR:
        n_target = int(round(len(audio) * AUDIO_SR / sr))
        audio = np.interp(
            np.linspace(0.0, 1.0, n_target, endpoint=False),
            np.linspace(0.0, 1.0, len(audio), endpoint=False),
            audio,
        ).astype(np.float32)
    n_target = int(AUDIO_SR * max_seconds)
    if len(audio) >= n_target:
        audio = audio[:n_target]
    else:
        audio = np.concatenate([audio, np.zeros(n_target - len(audio), dtype=np.float32)])
    return audio


def main() -> int:
    print(f"loading {MODEL_ID} (thinker-only)...", flush=True)
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.eval()
    model.to(torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print("model loaded", flush=True)

    audio = load_audio_16k_mono(CLIP)
    n_samples = len(audio)
    print(f"clip: {CLIP.name}  len={len(audio)/AUDIO_SR:.2f}s  n_samples={n_samples}", flush=True)

    convo = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": f"{QUESTION} Answer Yes or No."},
        ]},
    ]
    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    def one_pass() -> tuple[float, float]:
        """Return (feature_extract_seconds, forward_pass_seconds)."""
        # --- feature extraction (CPU, includes mel-spectrogram) ---
        t0 = time.perf_counter()
        inputs = processor(
            text=text,
            audio=audio,
            sampling_rate=AUDIO_SR,
            return_tensors="pt",
            padding=True,
            audio_kwargs={"max_length": n_samples, "truncation": True},
        )
        # Move tensors to MPS and cast floats.
        inputs = {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in inputs.items()}
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(torch.float16)
        t_feat = time.perf_counter() - t0

        # --- forward pass (MPS, logits only, no generate) ---
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
        # Force sync so the timer includes GPU completion, not just dispatch.
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_fwd = time.perf_counter() - t0

        # Touch the output so it isn't optimised away.
        _ = out.logits.shape
        return t_feat, t_fwd

    print(f"\n--- {N_WARMUP} warm-up pass(es) ---", flush=True)
    for i in range(N_WARMUP):
        t_feat, t_fwd = one_pass()
        print(f"  warm-up {i+1}: feat={t_feat*1000:.1f}ms  fwd={t_fwd*1000:.1f}ms", flush=True)

    print(f"\n--- {N_TIMED} timed passes ---", flush=True)
    feat_times: list[float] = []
    fwd_times: list[float] = []
    total_times: list[float] = []
    for i in range(N_TIMED):
        t_feat, t_fwd = one_pass()
        t_total = t_feat + t_fwd
        feat_times.append(t_feat)
        fwd_times.append(t_fwd)
        total_times.append(t_total)
        print(f"  pass {i+1}: feat={t_feat*1000:6.1f}ms  fwd={t_fwd*1000:6.1f}ms  "
              f"total={t_total*1000:6.1f}ms", flush=True)

    print("\n--- median over {} timed passes ---".format(N_TIMED), flush=True)
    med_feat = statistics.median(feat_times) * 1000
    med_fwd = statistics.median(fwd_times) * 1000
    med_total = statistics.median(total_times) * 1000
    print(f"  feature extraction median: {med_feat:.1f} ms", flush=True)
    print(f"  forward pass median:       {med_fwd:.1f} ms", flush=True)
    print(f"  total per-pass median:     {med_total:.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
