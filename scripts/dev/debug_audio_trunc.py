"""Verify truncation: input_features should be (1, 128, 300) for 3 s clip, not (1, 128, 30000)."""
import os, sys
from pathlib import Path
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import numpy as np
import soundfile as sf
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[2]
audio, sr = sf.read(str(_REPO_ROOT / "clips" / "voice_user_3s.wav"))
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
print(f"audio: sr={sr}, len={len(audio)/sr:.2f}s, frames@hop160={int(len(audio)/160)}",
      flush=True)

model_id = "Qwen/Qwen2.5-Omni-3B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

convo = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Is someone speaking? Answer Yes or No."},
    ]},
]
text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

print("\n--- audio_kwargs with max_length=48000 (3 s @ 16 kHz), truncation=True ---", flush=True)
inputs = processor(
    text=text,
    audio=audio,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True,
    audio_kwargs={"max_length": 48000, "truncation": True},
)
for k, v in inputs.items():
    try:
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}", flush=True)
    except Exception:
        print(f"  {k}: shape={tuple(v.shape)}", flush=True)
