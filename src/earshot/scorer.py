"""Stage 2 — sequential single-question scorer (no batching).

For each question:
  1. Build a fresh chat conversation (system + audio + question text).
  2. Tokenize via the processor with `audio_kwargs` truncating the audio
     features to the real length (no 30-s padding default).
  3. Run ONE forward pass. No generation.
  4. Read P(Yes) by log-sum-exp across the Yes and No token-id groups at
     the last position.

The model is loaded lazily on first call (cached at module level so a loop
over many clips doesn't reload it).
"""
from __future__ import annotations

import threading
import time
from typing import Iterable

import numpy as np
import torch

from .model import load_model
from .prompts import SYSTEM_MESSAGE, QUESTION_TEMPLATE


# ---------------------------------------------------------------------------
# Token-id helpers
# ---------------------------------------------------------------------------

# From the Stage 0 spike on voice_user_3s.wav:
#   yes_ids = [9454, 9834, 9693, 7414]
#   no_ids  = [2753, 902, 2152, 2308]
# These were chosen by collecting all single-token variants of "Yes" / "No"
# (with leading-space / case variants) the Qwen tokenizer emits. Using a
# group of ids makes P(Yes) more stable than a single "Yes" id.
DEFAULT_YES_IDS = [9454, 9834, 9693, 7414]
DEFAULT_NO_IDS = [2753, 902, 2152, 2308]


def find_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Re-derive (yes_ids, no_ids) from a tokenizer. Used by tests / fresh
    setups; the Stage 2 defaults are fine for Qwen2.5-Omni-3B."""
    yes_ids: list[int] = []
    no_ids: list[int] = []
    for w in ("Yes", " yes", "yes", " Yes"):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for w in ("No", " no", "no", " No"):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return list(dict.fromkeys(yes_ids)), list(dict.fromkeys(no_ids))


def p_yes_from_logits(
    last_logits: torch.Tensor,
    yes_ids: list[int] = DEFAULT_YES_IDS,
    no_ids: list[int] = DEFAULT_NO_IDS,
) -> float:
    """P(Yes) from the last-position logits. Log-sum-exp within each group,
    then softmax over the two groups."""
    ids = torch.tensor(yes_ids + no_ids, device=last_logits.device, dtype=torch.long)
    logits = last_logits[ids].float()
    n_yes = len(yes_ids)
    yes_lse = torch.logsumexp(logits[:n_yes], dim=0)
    no_lse = torch.logsumexp(logits[n_yes:], dim=0)
    pair = torch.stack([yes_lse, no_lse])
    probs = torch.softmax(pair, dim=0)
    return float(probs[0].item())


# ---------------------------------------------------------------------------
# Single-question scoring
# ---------------------------------------------------------------------------

def score_one(
    processor,
    model,
    device: torch.device,
    audio: np.ndarray,
    question_text: str,
    yes_ids: list[int] = DEFAULT_YES_IDS,
    no_ids: list[int] = DEFAULT_NO_IDS,
    n_samples: int | None = None,
) -> float:
    """One question, one audio. Returns P(Yes)."""
    convo = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": QUESTION_TEMPLATE.format(question=question_text)},
        ]},
    ]
    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]

    if n_samples is None:
        n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)

    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        audio_kwargs={"max_length": n_samples, "truncation": True},
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(torch.float16)

    with torch.inference_mode():
        out = model(**inputs)
    last = out.logits[0, -1].float().cpu()
    return p_yes_from_logits(last, yes_ids=yes_ids, no_ids=no_ids)


# ---------------------------------------------------------------------------
# Sequential API — model is lazy-loaded once and reused
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {}  # {"processor": ..., "model": ..., "device": ...,
                   #  "yes_ids": [...], "no_ids": [...]}


def _ensure_loaded():
    """Load and cache the model. Safe across threads / processes-as-threads."""
    with _CACHE_LOCK:
        if "model" not in _CACHE:
            processor, model, device = load_model()
            yes_ids, no_ids = find_yes_no_ids(processor.tokenizer)
            _CACHE["processor"] = processor
            _CACHE["model"] = model
            _CACHE["device"] = device
            _CACHE["yes_ids"] = yes_ids
            _CACHE["no_ids"] = no_ids
        return _CACHE


def score_sequential(
    audio: np.ndarray,
    questions: Iterable[dict] | Iterable[tuple[str, str]],
) -> dict[str, float]:
    """Score each question against `audio` sequentially.

    Args:
        audio: 1-D float32 numpy array at 16 kHz mono. The model will truncate
            the feature extractor to the real length (no 30-s padding).
        questions: either dicts (must have "id" and "text") or (id, text) pairs.

    Returns:
        dict mapping question id to P(Yes), in float. Call order does not
        affect the result, but Python insertion order is preserved.

    The model is loaded on the first call and reused on subsequent calls
    (cheaper when scoring the same model against many clips).
    """
    cache = _ensure_loaded()
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    yes_ids = cache["yes_ids"]
    no_ids = cache["no_ids"]

    out: dict[str, float] = {}
    n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    for q in questions:
        if isinstance(q, dict):
            qid = q["id"]
            text = q["text"]
        else:
            qid, text = q
        out[qid] = score_one(
            processor, model, device, audio, text,
            yes_ids=yes_ids, no_ids=no_ids, n_samples=n_samples,
        )
    return out
