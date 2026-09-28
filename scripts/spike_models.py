"""
Stage 0 — spike_models.py

Try to load each candidate audio LM, feed it a 3 s WAV, ask one yes/no question,
and print load time, peak memory and the answer. Skip any that fails to load and
print why.

Candidates, in order:
    1. Gemma 3n E2B (gated)
    2. Qwen2.5-Omni-3B (thinker only, no talker)
    3. Voxtral Mini 3B

Usage:
    uv run python scripts/spike_models.py --clip clips/voice_3s.wav
    uv run python scripts/spike_models.py --clip clips/voice_3s.wav --only gemma3n
    uv run python scripts/spike_models.py --clip clips/voice_3s.wav --question "Is someone speaking?"

Acceptance (per the Stage 0 prompt):
    At least one model loads, answers "Is someone speaking?" sensibly on a clip of
    you talking, and fits under ~6 GB.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Use HF Xet high-performance transfer for faster downloads when available.
# The legacy hf_transfer was deprecated in huggingface_hub 1.x; HF_XET_HIGH_PERFORMANCE
# replaces it. The HF token is read from ~/.cache/huggingface/token (set by `hf auth login`).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 16000  # all candidate processors want 16 kHz mono
DEFAULT_QUESTION = "Is someone speaking?"
MEM_BUDGET_BYTES = 6 * 1024**3  # 6 GB — Stage 0 acceptance budget
PROBE_SECONDS = 3.0  # the spec says 3 s


@dataclass
class ProbeResult:
    name: str
    loaded: bool
    skip_reason: str | None
    load_seconds: float | None
    peak_mem_gb: float | None
    answer: str | None
    p_yes: float | None
    yes_token_id: int | None
    no_token_id: int | None
    error: str | None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_clip_as_16k_mono(path: Path, max_seconds: float = PROBE_SECONDS) -> np.ndarray:
    """Load a WAV, downmix to mono, resample to 16 kHz, truncate/pad to max_seconds."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        # Linear resample — adequate for a 3 s probe.
        n_target = int(round(len(audio) * TARGET_SR / sr))
        audio = np.interp(
            np.linspace(0.0, 1.0, n_target, endpoint=False),
            np.linspace(0.0, 1.0, len(audio), endpoint=False),
            audio,
        ).astype(np.float32)
    n_target = int(TARGET_SR * max_seconds)
    if len(audio) >= n_target:
        audio = audio[:n_target]
    else:
        audio = np.concatenate([audio, np.zeros(n_target - len(audio), dtype=np.float32)])
    return audio


def synthesize_silence(seconds: float = PROBE_SECONDS) -> np.ndarray:
    """Fallback clip: 3 s of digital silence — used when no clip is provided."""
    return np.zeros(int(TARGET_SR * seconds), dtype=np.float32)


# ---------------------------------------------------------------------------
# Yes/No token id helper
# ---------------------------------------------------------------------------


def find_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Return token ids for variants of 'Yes' and 'No'. The scorer uses
    log-sum-exp across each group so leading-space and case variants both vote."""
    yes_candidates = ["Yes", " yes", "yes", " Yes"]
    no_candidates = ["No", " no", "no", " No"]
    yes_ids: list[int] = []
    no_ids: list[int] = []
    for w in yes_candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for w in no_candidates:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    # Deduplicate but preserve order
    yes_ids = list(dict.fromkeys(yes_ids))
    no_ids = list(dict.fromkeys(no_ids))
    return yes_ids, no_ids


def p_yes_from_logits(last_logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    """P(Yes) = softmax over the union of Yes/No token logits."""
    ids = torch.tensor(yes_ids + no_ids, device=last_logits.device, dtype=torch.long)
    logits = last_logits[ids]
    # Group 0 = Yes, Group 1 = No; log-sum-exp within each, then softmax across the two groups.
    n_yes = len(yes_ids)
    yes_lse = torch.logsumexp(logits[:n_yes], dim=0)
    no_lse = torch.logsumexp(logits[n_yes:], dim=0)
    pair = torch.stack([yes_lse, no_lse])
    probs = torch.softmax(pair, dim=0)
    return float(probs[0].item())


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def _peak_mem_gb() -> float:
    """Best-effort peak RSS. On macOS ru_maxrss is in bytes; on Linux it's in KB."""
    try:
        import resource  # type: ignore[import-not-found]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_gb = rss / (1024**3)
        else:
            rss_gb = rss / (1024 * 1024)
    except Exception:
        rss_gb = 0.0
    return rss_gb


# ---------------------------------------------------------------------------
# Per-model probes
# ---------------------------------------------------------------------------


def _pick_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def probe_gemma3n(audio: np.ndarray, question: str, device: torch.device, dtype: torch.dtype):
    """Gemma 3n E2B-it. Gated repo — caller must have access."""
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel

    model_id = "google/gemma-3n-E2B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True,
        )
    except (ValueError, KeyError, ImportError):
        model = AutoModel.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True,
        )
    model.eval()
    model.to(device)

    convo = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio, "sample_rate": TARGET_SR},
            {"type": "text", "text": f"{question} Answer Yes or No."},
        ]},
    ]
    inputs = processor.apply_chat_template(
        convo, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        out = model(**inputs)
    last = out.logits[0, -1].float().cpu()
    return last, processor.tokenizer, None


def probe_qwen(audio: np.ndarray, question: str, device: torch.device, dtype: torch.dtype):
    """Qwen2.5-Omni-3B THINKER only. Sidesteps the talker/code2wav weights that
    account for ~1.5 GB of the full model. Loads to CPU first then moves to MPS,
    which avoids a SIGSEGV in `at::native::mps::copy_cast_kernel_mps` triggered
    by `device_map={"": "mps"}` during from_pretrained."""
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2.5-Omni-3B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,  # CPU first; explicit move below avoids the MPS bug
    )
    model.eval()
    model.to(device)

    convo = [
        {"role": "system", "content": [{"type": "text", "text": "You are an audio listener. Answer the user's yes/no question about the audio with a single word: Yes or No."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": f"{question} Answer Yes or No."},
        ]},
    ]
    text = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list):
        text = text[0]

    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        out = model(**inputs)
    last = out.logits[0, -1].float().cpu()
    return last, processor.tokenizer, None


def probe_voxtral(audio: np.ndarray, question: str, device: torch.device, dtype: torch.dtype):
    """Voxtral Mini 3B. mistralai/Voxtral-Mini-3B-2507."""
    from transformers import VoxtralForConditionalGeneration, AutoProcessor

    model_id = "mistralai/Voxtral-Mini-3B-2507"
    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map=None,
    )
    model.eval()
    model.to(device)

    convo = [
        {"role": "user", "content": [
            {"type": "audio", "path": None, "data": audio, "sampling_rate": TARGET_SR},
            {"type": "text", "text": f"{question} Answer Yes or No."},
        ]},
    ]
    inputs = processor.apply_chat_template(
        convo, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        out = model(**inputs)
    last = out.logits[0, -1].float().cpu()
    return last, processor.tokenizer, None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CANDIDATES: dict[str, Callable] = {
    "gemma3n": probe_gemma3n,
    "qwen": probe_qwen,
    "voxtral": probe_voxtral,
}


def run_one(name: str, probe: Callable, clip: Path, question: str) -> ProbeResult:
    print(f"\n=== {name} ===", flush=True)
    if not clip.exists():
        return ProbeResult(name=name, loaded=False, skip_reason=f"clip not found: {clip}",
                           load_seconds=None, peak_mem_gb=None, answer=None, p_yes=None,
                           yes_token_id=None, no_token_id=None, error=None)
    audio = load_clip_as_16k_mono(clip)
    print(f"  clip: {clip}  len={len(audio)/TARGET_SR:.2f}s  rms={float(np.sqrt(np.mean(audio**2))):.4f}",
          flush=True)
    device, dtype = _pick_device_dtype()
    print(f"  device={device}  dtype={dtype}", flush=True)

    t0 = time.perf_counter()
    try:
        last_logits, tokenizer, skip = probe(audio, question, device, dtype)
    except Exception as e:  # noqa: BLE001 — surface every failure for the acceptance check
        print(f"  SKIP — {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(limit=4)
        return ProbeResult(name=name, loaded=False, skip_reason=f"{type(e).__name__}: {e}",
                           load_seconds=None, peak_mem_gb=None, answer=None, p_yes=None,
                           yes_token_id=None, no_token_id=None, error=traceback.format_exc())
    load_seconds = time.perf_counter() - t0

    yes_ids, no_ids = find_yes_no_ids(tokenizer)
    if not yes_ids or not no_ids:
        return ProbeResult(name=name, loaded=True, skip_reason="could not locate Yes/No token ids",
                           load_seconds=load_seconds, peak_mem_gb=_peak_mem_gb(),
                           answer=None, p_yes=None, yes_token_id=None, no_token_id=None,
                           error=None)
    p_yes = p_yes_from_logits(last_logits, yes_ids, no_ids)
    answer = "Yes" if p_yes >= 0.5 else "No"

    peak = _peak_mem_gb()
    fits = peak <= MEM_BUDGET_BYTES / 1024**3
    print(f"  load_seconds={load_seconds:.2f}  peak_mem≈{peak:.2f} GB  fits_6GB={fits}",
          flush=True)
    print(f"  yes_ids={yes_ids}  no_ids={no_ids}  P(Yes)={p_yes:.3f}  -> answer={answer}",
          flush=True)
    return ProbeResult(
        name=name, loaded=True, skip_reason=None, load_seconds=load_seconds,
        peak_mem_gb=peak, answer=answer, p_yes=p_yes,
        yes_token_id=yes_ids[0], no_token_id=no_ids[0], error=None,
    )


def summarise(results: list[ProbeResult]) -> None:
    print("\n=== Summary ===", flush=True)
    print(f"{'model':<10} {'loaded':<7} {'load_s':<8} {'peak_GB':<8} {'P(Yes)':<8} {'answer':<6} {'note'}", flush=True)
    for r in results:
        note = r.skip_reason or ""
        print(f"{r.name:<10} {str(r.loaded):<7} "
              f"{(f'{r.load_seconds:.2f}' if r.load_seconds is not None else '-'):<8} "
              f"{(f'{r.peak_mem_gb:.2f}' if r.peak_mem_gb is not None else '-'):<8} "
              f"{(f'{r.p_yes:.3f}' if r.p_yes is not None else '-'):<8} "
              f"{(r.answer or '-'):<6} {note}", flush=True)

    winners = [r for r in results if r.loaded and r.answer is not None
               and (r.peak_mem_gb or 0) <= MEM_BUDGET_BYTES / 1024**3]
    if not winners:
        print("\nNo candidate passed. See the fallback in the plan's risks table:", flush=True)
        print("  Try MLX builds of the same models, or fall back to Whisper-tiny + a small text LLM.", flush=True)
        return
    # Fastest winner that answers sensibly. For now, "sensibly" = we trust the model's softmax;
    # the acceptance check is the human listening to the answer in the terminal.
    winner = min(winners, key=lambda r: r.load_seconds or 1e9)
    print(f"\nStage 0 candidate selected: {winner.name}  "
          f"(P(Yes)={winner.p_yes:.3f}, load={winner.load_seconds:.1f}s, peak≈{winner.peak_mem_gb:.2f} GB)", flush=True)
    print("Write this into AGENTS.md under 'Selected model (Stage 0)' after running:", flush=True)
    print(f"  {winner.name} — selected by spike_models.py on {time.strftime('%Y-%m-%d')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 0 model-load spike.")
    p.add_argument("--clip", type=Path, default=None,
                   help="Path to a 3 s WAV of you talking. If omitted, silence is used.")
    p.add_argument("--question", type=str, default=DEFAULT_QUESTION,
                   help="Yes/no question to ask.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma list of candidate names (gemma3n, qwen, voxtral). Default: all.")
    args = p.parse_args(argv)

    clip: Path = args.clip if args.clip else Path("clips/_silence_3s.wav")
    if not clip.exists():
        clip.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(clip), synthesize_silence(), TARGET_SR)
        print(f"  no --clip given; wrote fallback silence to {clip}", flush=True)

    # Default order: gemma3n first (smallest), then qwen thinker-only, then voxtral.
    order = ["gemma3n", "qwen", "voxtral"]
    if args.only:
        order = [x.strip() for x in args.only.split(",") if x.strip()]

    results: list[ProbeResult] = []
    for name in order:
        probe = CANDIDATES.get(name)
        if probe is None:
            print(f"unknown candidate: {name}", file=sys.stderr)
            continue
        results.append(run_one(name, probe, clip, args.question))
    summarise(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
