"""Debug: verify audio is actually being processed by Qwen2.5-Omni thinker.
Two-step approach: render text with placeholders via apply_chat_template(tokenize=False),
then pass text + audio to processor separately.
"""
import os, sys, traceback
from pathlib import Path
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import numpy as np
import soundfile as sf
import torch

print("start", flush=True)

# Resolve clip path relative to repo root (this file lives in scripts/dev/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
clip_path = _REPO_ROOT / "clips" / "voice_user_3s.wav"
audio, sr = sf.read(str(clip_path))
print(f"loaded audio: sr={sr}, shape={audio.shape}, dtype={audio.dtype}, "
      f"rms={float(np.sqrt(np.mean(audio.astype(np.float32)**2))):.4f}", flush=True)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor
model_id = "Qwen/Qwen2.5-Omni-3B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Method B: render text via chat template (tokenize=False), then processor(text=, audio=)
convo = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Is someone speaking? Answer Yes or No."},
    ]},
]
text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
print("\n--- Method B: apply_chat_template(tokenize=False) + processor(text=, audio=) ---", flush=True)
print(f"rendered text (first 300 chars): {text[:300]!r}", flush=True)
print(f"  contains <|AUDIO|>: {'<|AUDIO|>' in text}", flush=True)
print(f"  contains <|audio_bos|>: {'<|audio_bos|>' in text}", flush=True)

try:
    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    print(f"\nprocessor output keys: {list(inputs.keys())}", flush=True)
    for k, v in inputs.items():
        try:
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                  f"min={float(v.min()):.3f}, max={float(v.max()):.3f}", flush=True)
        except Exception:
            print(f"  {k}: shape={tuple(v.shape)}", flush=True)
    if "input_ids" in inputs:
        ids = inputs["input_ids"][0].tolist()
        print(f"\n  decoded (first 60): {processor.tokenizer.decode(ids[:60])!r}", flush=True)
        n_audio_tokens = sum(1 for i in ids if i == 151646)
        n_audio_bos = sum(1 for i in ids if i == 151647)
        n_audio_eos = sum(1 for i in ids if i == 151648)
        print(f"  <|AUDIO|> tokens (151646): {n_audio_tokens}", flush=True)
        print(f"  <|audio_bos|> tokens (151647): {n_audio_bos}", flush=True)
        print(f"  <|audio_eos|> tokens (151648): {n_audio_eos}", flush=True)
except Exception as e:
    print(f"Method B failed: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()

print("\nDONE", flush=True)
