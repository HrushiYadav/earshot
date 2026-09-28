"""Stage 0 follow-up — silence sanity check.

Loads Qwen2.5-Omni-3B (thinker-only) once, asks "Is someone speaking?" on:
  1. clips/silence_3s.wav  (3 s of digital silence)
  2. clips/voice_user_3s.wav  (3 s of your voice, baseline)

P(Yes) on silence must be clearly lower than on the voice clip.
"""
from __future__ import annotations

import os, sys, traceback
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import numpy as np
import soundfile as sf
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS = {
    "silence": _REPO_ROOT / "clips" / "silence_3s.wav",
    "voice":   _REPO_ROOT / "clips" / "voice_user_3s.wav",
}
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
QUESTION = "Is someone speaking?"
AUDIO_SR = 16000


def load_audio_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def find_yes_no_ids(tokenizer):
    yes_ids, no_ids = [], []
    for w in ("Yes", " yes", "Yes "):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for w in ("No", " no", "No "):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return list(dict.fromkeys(yes_ids)), list(dict.fromkeys(no_ids))


def p_yes_from_logits(last, yes_ids, no_ids) -> float:
    ids = torch.tensor(yes_ids + no_ids, dtype=torch.long)
    logits = last[ids].float()
    n_yes = len(yes_ids)
    yes_lse = torch.logsumexp(logits[:n_yes], dim=0)
    no_lse = torch.logsumexp(logits[n_yes:], dim=0)
    pair = torch.stack([yes_lse, no_lse])
    probs = torch.softmax(pair, dim=0)
    return float(probs[0].item())


def main() -> int:
    print(f"loading {MODEL_ID} (thinker-only)...", flush=True)
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map=None,
    )
    model.eval()
    model.to(torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print("model loaded", flush=True)

    yes_ids, no_ids = find_yes_no_ids(processor.tokenizer)

    print(f"\n{'clip':<10} {'len(s)':<8} {'rms':<8} {'P(Yes)':<8} {'answer':<6} {'yes_ids':<22} {'no_ids':<22}", flush=True)
    results = {}
    for label, path in CLIPS.items():
        audio = load_audio_16k_mono(path)
        n_samples = len(audio)
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

        convo = [{"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": f"{QUESTION} Answer Yes or No."},
        ]}]
        text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, audio=audio, sampling_rate=AUDIO_SR,
            return_tensors="pt", padding=True,
            audio_kwargs={"max_length": n_samples, "truncation": True},
        )
        inputs = {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in inputs.items()}
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(torch.float16)
        with torch.inference_mode():
            out = model(**inputs)
        last = out.logits[0, -1].float().cpu()
        p_yes = p_yes_from_logits(last, yes_ids, no_ids)
        answer = "Yes" if p_yes >= 0.5 else "No"
        results[label] = p_yes
        print(f"{label:<10} {len(audio)/AUDIO_SR:<8.2f} {rms:<8.4f} {p_yes:<8.3f} {answer:<6} "
              f"{str(yes_ids):<22} {str(no_ids):<22}", flush=True)

    p_silence = results["silence"]
    p_voice = results["voice"]
    print(f"\nratio P(Yes)_voice / P(Yes)_silence = {p_voice / max(p_silence, 1e-9):.2f}x",
          flush=True)
    print(f"P(Yes)_silence ({p_silence:.3f}) is {'LOWER' if p_silence < p_voice else 'NOT lower'} "
          f"than P(Yes)_voice ({p_voice:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
