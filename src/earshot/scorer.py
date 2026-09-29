"""Stage 2 — sequential single-question scorer (no batching).
Stage 3 — fork-and-score: one prefix forward, broadcast KV cache across N suffixes.
Stage 4 — typed schema: bool P(Yes) and choice P(letter), discriminated on question type.

For each model-driven bool question:
  1. Build a fresh chat conversation (system + audio + question text).
  2. Tokenize via the processor with `audio_kwargs` truncating the audio
     features to the real length (no 30-s padding default).
  3. Run ONE forward pass. No generation.
  4. Read P(Yes) by log-sum-exp across the Yes and No token-id groups at
     the last position.

For each model-driven choice question:
  1. Build the chat conversation with the question text, an "Answer with
     one letter" instruction, and the options presented as lettered lines:
         What is the main speaker doing? Answer with one letter (A, B, ...).
         A. talking
         B. laughing
         …
  2. Run the same prefix+suffix forward as for bool.
  3. At the last non-pad position, take logits for the letter tokens A/B/C…
     and softmax over the options.

Signal-derived questions (`source="signal"`, currently just `silent`) skip
the model entirely — `score_signal_silent()` answers from the audio RMS.

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
    are fed to the same Yes/No log-sum-exp softmax as score_one (bool), or
    to a letter-token softmax across the options (choice).

The model is loaded lazily on first call (cached at module level so a loop
over many clips doesn't reload it).
"""
from __future__ import annotations

import threading
from typing import Iterable, Union

import numpy as np
import torch

from .model import load_model
from .prompts import (
    QUESTION_TEMPLATE,
    SIGNAL_SILENT_RMS_THRESHOLD,
    SYSTEM_MESSAGE,
)
from .schema import (
    LETTERS,
    BoolQuestion,
    BoolResult,
    ChoiceQuestion,
    ChoiceResult,
    letter_for_index,
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

# Letter token IDs in the Qwen2.5-Omni tokenizer. Each uppercase letter is a
# single token: 'A'..'Z' map to consecutive IDs 32..57 (verified at
# module-load time below). We rely on this so choice scoring can pluck
# one logit per option directly without splitting or joining subword pieces.
LETTER_BASE_ID = 32  # token id of 'A'


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


def letter_token_ids(n: int) -> list[int]:
    """Return the token ids for the first `n` uppercase letters: [A, B, …].

    The Qwen2.5-Omni tokenizer assigns 'A'..'Z' to consecutive ids 32..57
    (one token per letter, no leading-space variants). Callers building a
    choice suffix can format lines like `"A. talking"` and trust the
    letter tokenization is single-id.
    """
    if not 1 <= n <= 26:
        raise ValueError(f"letter_token_ids({n}): need 1..26")
    return [LETTER_BASE_ID + i for i in range(n)]


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


def probs_from_letter_logits(
    last_logits: torch.Tensor, options: list[str]
) -> dict[str, float]:
    """P(option) from last-position logits for the letter tokens A/B/C….

    Each letter is a single token in the Qwen tokenizer (verified at
    module load); we pluck one logit per option and softmax. If an option
    somehow got a multi-token letter (e.g. lowercased) the caller must
    normalise first.
    """
    if not 1 <= len(options) <= 26:
        raise ValueError(f"probs_from_letter_logits: need 2..26 options, got {len(options)}")
    ids = torch.tensor(letter_token_ids(len(options)),
                       device=last_logits.device, dtype=torch.long)
    logits = last_logits[ids].float()
    probs = torch.softmax(logits, dim=0).cpu().tolist()
    return {opt: float(p) for opt, p in zip(options, probs)}


# ---------------------------------------------------------------------------
# Suffix text builders
# ---------------------------------------------------------------------------

def _bool_suffix_text(question: str) -> str:
    """The text fed to the model after <|audio_eos|> for a bool question.
    Stays the same shape as Stage 2 — `"<question>? Answer Yes or No."` —
    so the chat-tail `"\\n<|im_start|>assistant\\n"` is appended by the
    score_batched split (suffix_tail)."""
    return QUESTION_TEMPLATE.format(question=question)


def _choice_suffix_text(question: str, options: list[str]) -> str:
    """The text fed to the model for a choice question. Options are
    presented as lettered lines so the model can answer with a single
    letter token; that letter is then softmaxed across options.
    """
    lines = [
        f"{question} Answer with one letter (A, B, C, ...).",
    ]
    for i, opt in enumerate(options):
        lines.append(f"{letter_for_index(i)}. {opt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal-derived scoring (no model)
# ---------------------------------------------------------------------------

def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of a 1-D float audio buffer."""
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def score_signal_silent(
    audio: np.ndarray,
    qid: str = "silent",
    text: str = "Is the room silent?",
    threshold: float = SIGNAL_SILENT_RMS_THRESHOLD,
    enter: float = 0.5,
) -> BoolResult:
    """Score `silent` from the audio RMS.

    Threshold chosen in prompts.py from the manifest data. Returns a
    BoolResult with prob = 1.0 for Yes (room is silent) or 0.0 for No.
    `enter` is the question's enter threshold; `fired` is set when
    p crosses it. (For the default enter=0.5, fired == is_silent.)
    """
    r = rms(audio)
    p = 1.0 if r < threshold else 0.0
    return BoolResult(
        id=qid, text=text,
        p=p, fired=(p >= enter), source="signal",
    )


# ---------------------------------------------------------------------------
# Single-question scoring (model) — kept for completeness; the live path
# uses score_batched.
# ---------------------------------------------------------------------------

def _build_one(processor, model, device, audio, question: BoolQuestion | ChoiceQuestion,
               yes_ids, no_ids, n_samples, *, dtype=torch.float16):
    """One full forward for a single question. Returns a BoolResult or
    ChoiceResult depending on the question type."""
    if isinstance(question, ChoiceQuestion):
        suffix_text = _choice_suffix_text(question.text, question.options)
    else:
        suffix_text = _bool_suffix_text(question.text)

    convo = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": suffix_text},
        ]},
    ]
    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]
    inputs = processor(
        text=text, audio=audio, sampling_rate=16_000,
        return_tensors="pt", padding=True,
        audio_kwargs={"max_length": n_samples, "truncation": True},
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(dtype)

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

    return _result_from_logits(question, last, yes_ids=yes_ids, no_ids=no_ids)


def _result_from_logits(question, last_logits, yes_ids, no_ids):
    """Pick the right read-out (Yes/No or letter-token softmax) and wrap
    it in the matching pydantic result."""
    if isinstance(question, ChoiceQuestion):
        probs = probs_from_letter_logits(last_logits, question.options)
        top = max(probs, key=probs.get)
        return ChoiceResult(
            id=question.id, text=question.text,
            probs=probs, top=top, source="model",
        )
    p = p_yes_from_logits(last_logits, yes_ids=yes_ids, no_ids=no_ids)
    return BoolResult(
        id=question.id, text=question.text,
        p=p, fired=(p >= question.enter), source="model",
    )


def score_one(
    processor, model, device, audio, question: BoolQuestion | ChoiceQuestion,
    yes_ids=None, no_ids=None, n_samples: int | None = None,
):
    """Public entry point for a single (audio, question) score. Returns a
    BoolResult or ChoiceResult."""
    if n_samples is None:
        n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    if yes_ids is None:
        yes_ids = DEFAULT_YES_IDS
    if no_ids is None:
        no_ids = DEFAULT_NO_IDS
    return _build_one(processor, model, device, audio, question, yes_ids, no_ids, n_samples)


# ---------------------------------------------------------------------------
# Sequential API — model is lazy-loaded once and reused
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {}


def _ensure_loaded(dtype: torch.dtype = torch.float16):
    """Lazy-load and cache the model. If a different dtype is requested
    after a previous load, the cache is rebuilt."""
    with _CACHE_LOCK:
        if "model" not in _CACHE or _CACHE.get("dtype") != dtype:
            processor, model, device = load_model(dtype=dtype)
            yes_ids, no_ids = find_yes_no_ids(processor.tokenizer)
            _CACHE["processor"] = processor
            _CACHE["model"] = model
            _CACHE["device"] = device
            _CACHE["dtype"] = dtype
            _CACHE["yes_ids"] = yes_ids
            _CACHE["no_ids"] = no_ids
        return _CACHE


def score_sequential(
    audio: np.ndarray,
    questions: Iterable[BoolQuestion | ChoiceQuestion],
    *,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Union[BoolResult, ChoiceResult]]:
    """Score each question against `audio` sequentially (one forward pass per
    question). Returns dict[id -> BoolResult | ChoiceResult]. `dtype`
    overrides the production fp16 default; pass torch.float32 for the
    exactness proof (see test_equivalence.py --fp32)."""
    cache = _ensure_loaded(dtype=dtype)
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    yes_ids = cache["yes_ids"]
    no_ids = cache["no_ids"]

    out: dict[str, Union[BoolResult, ChoiceResult]] = {}
    n_samples = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
    for q in questions:
        if q.source == "signal":
            if q.signal == "rms" and q.id == "silent":
                out[q.id] = score_signal_silent(
                    audio, qid=q.id, text=q.text,
                    threshold=q.signal_threshold or SIGNAL_SILENT_RMS_THRESHOLD,
                    enter=q.enter,
                )
            else:
                raise ValueError(f"unknown signal question: id={q.id!r} signal={q.signal!r}")
            continue
        out[q.id] = _build_one(processor, model, device, audio, q, yes_ids, no_ids, n_samples,
                              dtype=dtype)
    return out


# ---------------------------------------------------------------------------
# Stage 3 — fork-and-score batched scorer
# ---------------------------------------------------------------------------

# SENTINEL placeholder inserted where the question text would go. The chat
# template renders it verbatim, sitting between <|audio_eos|> and "\nassistant\n".
# We slice the rendered template at the SENTINEL to produce a question-free
# prefix that is identical across all questions. The NUL-byte delimiters
# make it unambiguous that this is a control token, not real text (the
# tokenizer would never emit NULs in a normal conversation).
_PREFIX_SENTINEL = "\x00QUERY\x00"


def score_batched(
    audio: np.ndarray,
    questions: Iterable[BoolQuestion | ChoiceQuestion],
    batch_size: int | None = None,
    *,
    suffix_pos_offset: int = 0,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Union[BoolResult, ChoiceResult]]:
    """Stage 3 — fork-and-score: one prefix forward pass, broadcast across N
    question suffixes in a single batched pass. Returns dict[id ->
    BoolResult | ChoiceResult], matching `score_sequential`'s shape so
    callers can swap one for the other.

    `suffix_pos_offset` is a control knob for the equivalence test: when
    set to a non-zero value, every suffix position id is shifted by that
    amount (intended use: +1). This deliberately breaks the test so we
    can prove the test is sensitive to position bugs. Defaults to 0
    (correct). Never set this from production code.

    `dtype` overrides the production fp16 default; pass
    torch.float32 for the exactness proof (see test_equivalence.py
    --fp32)."""
    questions = list(questions)
    cache = _ensure_loaded(dtype=dtype)
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    yes_ids = cache["yes_ids"]
    no_ids = cache["no_ids"]

    out: dict[str, Union[BoolResult, ChoiceResult]] = {}
    signal_qs = [q for q in questions if q.source == "signal"]
    model_qs = [q for q in questions if q.source == "model"]
    for q in signal_qs:
        if q.signal == "rms" and q.id == "silent":
            out[q.id] = score_signal_silent(
                audio, qid=q.id, text=q.text,
                threshold=q.signal_threshold or SIGNAL_SILENT_RMS_THRESHOLD,
                enter=q.enter,
            )
        else:
            raise ValueError(f"unknown signal question: id={q.id!r} signal={q.signal!r}")
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
    prefix_text = rendered[:s_idx]
    suffix_tail = rendered[s_idx + len(_PREFIX_SENTINEL):]

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
            prefix_inputs[k] = v.to(dtype)

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
    suffix_texts: list[str] = []
    for q in model_qs:
        if isinstance(q, ChoiceQuestion):
            body = _choice_suffix_text(q.text, q.options)
        else:
            body = _bool_suffix_text(q.text)
        suffix_texts.append(body + suffix_tail)

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

        # 5b. Position ids for the suffix: pure text continuing the cached
        # prefix, so all three rotary axes are identical and consecutive,
        # starting from `prefix_seq_len`. `suffix_pos_offset` is the test
        # knob that deliberately breaks positions; production always uses 0.
        suffix_pos = (
            prefix_seq_len + suffix_pos_offset
            + torch.arange(max_len, device=device).unsqueeze(0).expand(3, N, max_len)
        ).contiguous()

        # 5c. Full attention mask = prefix (all 1s) ++ suffix with right-pad
        # zeros. The model's 4D causal-mask construction uses this to know
        # how much cached prefix to attend over.
        attn_mask = torch.cat(
            [
                torch.ones((N, prefix_seq_len), dtype=suffix_attn.dtype, device=device),
                suffix_attn,
            ],
            dim=1,
        )

        with torch.inference_mode():
            chunk_out = model(
                input_ids=suffix_ids,
                attention_mask=attn_mask,
                position_ids=suffix_pos,
                past_key_values=past_kv,
            )
        logits = chunk_out.logits  # (N, max_len, vocab)

        # 5d. Per row, read logits at the last non-pad position.
        row_idx = torch.arange(N, device=device)
        last_logits = logits[row_idx, torch.tensor(last_non_pad, device=device)].float().cpu()
        for i, q_idx in enumerate(chunk):
            q = model_qs[q_idx]
            out[q.id] = _result_from_logits(q, last_logits[i], yes_ids=yes_ids, no_ids=no_ids)

    return out


__all__ = [
    "DEFAULT_YES_IDS", "DEFAULT_NO_IDS", "LETTERS",
    "find_yes_no_ids", "letter_token_ids",
    "p_yes_from_logits", "probs_from_letter_logits",
    "rms", "score_signal_silent",
    "score_one", "score_sequential", "score_batched",
]
