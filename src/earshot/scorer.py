"""Stage 2 — sequential single-question scorer (no batching).
Stage 3 — fork-and-score: one prefix forward, broadcast KV cache across N suffixes.

For each model-driven question:
  1. Build a fresh chat conversation (system + audio + question text).
  2. Tokenize via the processor with `audio_kwargs` truncating the audio
     features to the real length (no 30-s padding default).
  3. Run ONE forward pass. No generation.
  4. Read P(Yes) by log-sum-exp across the Yes and No token-id groups at
     the last position.

Stage 2.5 (pre-Stage 3 tuning pass) added signal-derived questions. A
question with `source="signal"` skips the model and is answered from the raw
audio (RMS check for `silent`). All other questions go to the model.

Stage 3 — score_batched:
  - Signal questions are computed up front and skipped from the model batch.
  - Prefix = (system message + audio tokens, up to and including <|audio_eos|>).
    The question text is replaced with a SENTINEL placeholder so the chat
    template renders everything up to <|audio_eos|> verbatim, then the
    SENTINEL is sliced out. That gives us a prefix that is identical across
    all questions.
  - Prefix runs once with use_cache=True, batch size 1. Past_key_values is
    then expanded to batch N with batch_repeat_interleave(N) (a view, no
    real memory copy).
  - Each question suffix is the question text + the chat-tail ("\nassistant\n").
    Each suffix is tokenized separately and right-padded to the max length.
  - Position ids for the suffix are computed via the model's get_rope_index
    helper applied to the FULL (prefix + suffix) sequence, then sliced to the
    suffix portion. For Qwen2.5-Omni the suffix is pure text so all three
    rotary axes are identical and consecutive.
  - One batched forward pass. Per-row, logits at the last NON-pad position
    are fed to the same Yes/No log-sum-exp softmax as score_one.

The model is loaded lazily on first call (cached at module level so a loop
over many clips doesn't reload it).
"""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
import torch

from .model import load_model
from .prompts import (
    QUESTION_TEMPLATE,
    SIGNAL_SILENT_RMS_THRESHOLD,
    SYSTEM_MESSAGE,
)


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
# Result type — shared by signal and model scorers
# ---------------------------------------------------------------------------

# A ScoreResult is what every question reduces to, regardless of source. The
# fields are kept as a plain dict (not a dataclass) until Stage 4 turns them
# into a real typed schema. The keys here are stable; Stage 4 should not
# rename them without a migration.
#
#   id      — question id (matches the input dict's "id")
#   text    — question text (matches the input dict's "text")
#   prob    — P(Yes), float in [0, 1]
#   answer  — "Yes" if prob >= 0.5 else "No"
#   source  — "model" or "signal" (the producer of `prob`)
#
def score_result(qid: str, text: str, prob: float, source: str) -> dict:
    return {
        "id": qid,
        "text": text,
        "prob": prob,
        "answer": "Yes" if prob >= 0.5 else "No",
        "source": source,
    }


# ---------------------------------------------------------------------------
# Signal-derived scoring (no model)
# ---------------------------------------------------------------------------

def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of a 1-D float audio buffer."""
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def score_signal_silent(audio: np.ndarray, qid: str = "silent", text: str = "Is the room silent?") -> dict:
    """Score `silent` from the audio RMS.

    Threshold chosen in prompts.py from the manifest data. Returns a
    ScoreResult with prob = 1.0 for Yes (room is silent) or 0.0 for No.
    """
    r = rms(audio)
    is_silent = r < SIGNAL_SILENT_RMS_THRESHOLD
    return score_result(qid, text, prob=1.0 if is_silent else 0.0, source="signal")


# ---------------------------------------------------------------------------
# Single-question scoring (model)
# ---------------------------------------------------------------------------

def _build_one(processor, model, device, audio, question_text, yes_ids, no_ids, n_samples):
    """Build inputs for a single question + audio, run model, return P(Yes)."""
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

    feat_mask = inputs.get("feature_attention_mask")
    audio_seqlen = (
        int(feat_mask.sum().item()) if feat_mask is not None
        else inputs["input_features"].shape[-1]
    )
    pos_ids, _ = model.get_rope_index(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long, device=device),
    )
    inputs["position_ids"] = pos_ids

    with torch.inference_mode():
        out = model(**inputs)
    last = out.logits[0, -1].float().cpu()
    return p_yes_from_logits(last, yes_ids=yes_ids, no_ids=no_ids)


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
    if n_samples is None:
        n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    return _build_one(processor, model, device, audio, question_text, yes_ids, no_ids, n_samples)


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
    questions: Iterable[dict],
) -> dict[str, dict]:
    """Score each question against `audio` sequentially (one forward pass per
    question). See module docstring for the per-question dict shape and the
    ScoreResult fields."""
    cache = _ensure_loaded()
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    yes_ids = cache["yes_ids"]
    no_ids = cache["no_ids"]

    out: dict[str, dict] = {}
    n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    for q in questions:
        qid = q["id"]
        text = q["text"]
        source = q.get("source", "model")
        if source == "signal":
            if qid == "silent":
                out[qid] = score_signal_silent(audio, qid=qid, text=text)
            else:
                raise ValueError(f"unknown signal question id: {qid!r}")
        else:
            prob = _build_one(processor, model, device, audio, text, yes_ids, no_ids, n_samples)
            out[qid] = score_result(qid, text, prob, source="model")
    return out


# ---------------------------------------------------------------------------
# Stage 3 — fork-and-score batched scorer
# ---------------------------------------------------------------------------

# SENTINEL placeholder inserted where the question text would go. The chat
# template renders it verbatim, sitting between <|audio_eos|> and "\nassistant\n".
# We slice the rendered template at the SENTINEL to produce a question-free
# prefix that is identical across all questions.
_PREFIX_SENTINEL = "\u2e80EARSHOT_Q\u2e80"


def score_batched(
    audio: np.ndarray,
    questions: Iterable[dict],
    batch_size: int | None = None,
) -> dict[str, dict]:
    """Stage 3 — fork-and-score: one prefix forward pass, broadcast across N
    question suffixes in a single batched pass.

    Signal questions (`source="signal"`) are computed up front via the
    matching scorer (currently only `silent` -> RMS check) and skipped from
    the model batch. Model-driven questions are scored in chunks of
    `batch_size` rows per forward pass (default: all in one batch).

    Returns:
        dict mapping question id -> ScoreResult dict. Same shape as
        score_sequential, so callers can swap one for the other.
    """
    questions = list(questions)
    cache = _ensure_loaded()
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    yes_ids = cache["yes_ids"]
    no_ids = cache["no_ids"]

    out: dict[str, dict] = {}
    signal_qs = [q for q in questions if q.get("source") == "signal"]
    model_qs = [q for q in questions if q.get("source", "model") == "model"]
    for q in signal_qs:
        if q["id"] == "silent":
            out[q["id"]] = score_signal_silent(audio, qid=q["id"], text=q["text"])
        else:
            raise ValueError(f"unknown signal question id: {q['id']!r}")
    if not model_qs:
        return out

    n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    bs = batch_size or len(model_qs)

    # --- 1. Build the prefix via the SENTINEL trick -------------------------
    convo_prefix = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": _PREFIX_SENTINEL},
        ]},
    ]
    rendered = processor.apply_chat_template(convo_prefix, tokenize=False, add_generation_prompt=True)
    if isinstance(rendered, list):
        rendered = rendered[0]
    s_idx = rendered.find(_PREFIX_SENTINEL)
    if s_idx < 0:
        raise RuntimeError(f"SENTINEL {_PREFIX_SENTINEL!r} not found in rendered template:\n{rendered}")
    prefix_text = rendered[:s_idx]                # up to and including <|audio_eos|>
    suffix_tail = rendered[s_idx + len(_PREFIX_SENTINEL):]   # "\nassistant\n"

    # --- 2. Tokenize the prefix (with audio) --------------------------------
    prefix_inputs = processor(
        text=prefix_text, audio=audio, sampling_rate=16_000,
        return_tensors="pt", padding=True,
        audio_kwargs={"max_length": n_samples, "truncation": True},
    )
    feat_mask = prefix_inputs.get("feature_attention_mask")
    audio_seqlen = (
        int(feat_mask.sum().item()) if feat_mask is not None
        else prefix_inputs["input_features"].shape[-1]
    )

    prefix_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in prefix_inputs.items()}
    for k, v in list(prefix_inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            prefix_inputs[k] = v.to(torch.float16)

    pos_ids, _ = model.get_rope_index(
        input_ids=prefix_inputs["input_ids"],
        attention_mask=prefix_inputs["attention_mask"],
        audio_seqlens=torch.tensor([audio_seqlen], dtype=torch.long, device=device),
    )
    prefix_inputs["position_ids"] = pos_ids

    # --- 3. Run the prefix once with use_cache=True --------------------------
    with torch.inference_mode():
        prefix_out = model(**prefix_inputs, use_cache=True)
    past_kv = prefix_out.past_key_values
    prefix_seq_len = prefix_inputs["input_ids"].shape[1]

    # --- 4. Tokenize all suffix texts and right-pad -------------------------
    # suffix = "<question>? Answer Yes or No." + suffix_tail ("\nassistant\n")
    suffix_texts = [
        QUESTION_TEMPLATE.format(question=q["text"]) + suffix_tail for q in model_qs
    ]
    suffix_token_lists = [
        processor.tokenizer.encode(t, add_special_tokens=False) for t in suffix_texts
    ]

    pad_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
    )
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")

    # --- 5. Score in chunks of `batch_size` rows ----------------------------
    total = len(model_qs)
    for chunk_begin in range(0, total, bs):
        chunk = list(range(chunk_begin, min(chunk_begin + bs, total)))
        N = len(chunk)
        token_lists = [suffix_token_lists[i] for i in chunk]
        max_len = max(len(ids) for ids in token_lists)
        suffix_ids = torch.full((N, max_len), pad_id, dtype=torch.long)
        suffix_attn = torch.zeros((N, max_len), dtype=torch.long)
        last_non_pad: list[int] = []
        for i, ids in enumerate(token_lists):
            L = len(ids)
            suffix_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            suffix_attn[i, :L] = 1
            last_non_pad.append(L - 1)
        suffix_ids = suffix_ids.to(device)
        suffix_attn = suffix_attn.to(device)

        # 5a. Expand the prefix KV cache to batch N. With prefix batch=1, this
        # is a no-copy repeat_interleave(N, dim=0) on every layer's K/V.
        # IMPORTANT: DynamicCache.batch_repeat_interleave mutates in place
        # and returns None — passing the return value as past_key_values
        # silently forwards None to the model and the audio prefix is lost.
        # Use the (now-mutated) past_kv itself.
        past_kv.batch_repeat_interleave(N)

        # 5b. Position ids for the suffix: the suffix is pure text on top of an
        # already-cached prefix, so all three rotary axes are identical and
        # consecutive, starting from `prefix_seq_len`. (Qwen2.5-Omni reuses the
        # cached `rope_deltas` from the prefix forward and derives positions
        # from the FULL attention mask's cumsum; we pass our own position_ids
        # here so the model doesn't have to recompute them.)
        suffix_pos = (
            prefix_seq_len
            + torch.arange(max_len, device=device).unsqueeze(0).expand(3, N, max_len)
        ).contiguous()

        # 5c. Full attention mask = prefix (all 1s) ++ suffix with right-pad
        # zeros. The model's 4D causal-mask construction uses this to know how
        # much cached prefix to attend over. Passing only the suffix mask
        # makes the model think total length == suffix length and the prefix
        # audio gets masked out.
        attn_mask = torch.cat(
            [
                torch.ones((N, prefix_seq_len), dtype=suffix_attn.dtype, device=device),
                suffix_attn,
            ],
            dim=1,
        )

        # 5d. Run the batched forward pass.
        with torch.inference_mode():
            chunk_out = model(
                input_ids=suffix_ids,
                attention_mask=attn_mask,
                position_ids=suffix_pos,
                past_key_values=past_kv,
            )
        logits = chunk_out.logits  # (N, max_len, vocab)

        # 5e. Per row, read logits at the last non-pad position.
        row_idx = torch.arange(N, device=device)
        last_logits = logits[row_idx, torch.tensor(last_non_pad, device=device)].float().cpu()
        for i, q_idx in enumerate(chunk):
            prob = p_yes_from_logits(last_logits[i], yes_ids=yes_ids, no_ids=no_ids)
            q = model_qs[q_idx]
            out[q["id"]] = score_result(q["id"], q["text"], prob, source="model")

    return out