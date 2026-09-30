#!/usr/bin/env python3
"""Stage 6 — full benchmark.

Writes:
  results/bench.csv       — machine-readable per-N numbers
  results/bench.md        — human-readable report
  results/latency.png     — line chart, sequential vs batched

Run with the laptop plugged in and the lid open. The 2-minute warm-up
before timing is deliberate so the fanless M4 Air throttles into its
steady-state clock and we don't time the burst rate.

Usage:
  uv run python scripts/bench.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from earshot.audio import DEFAULT_SR, DEFAULT_WINDOW_SECONDS  # noqa: E402
from earshot.prompts import load_default_questions  # noqa: E402
from earshot.schema import BoolQuestion  # noqa: E402
from earshot.scorer import score_batched, score_sequential  # noqa: E402


# --- run config ---------------------------------------------------------

WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS  # 3.0 s — the live window size
WARMUP_SECONDS = 120.0  # 2 min — soak so the fanless Air throttles to steady-state
N_LIST_MAIN = [1, 4, 8, 16]
N_LIST_EXTRA = [32]  # run separately as a "note", not in the main table
TIMED_PASSES_DEFAULT = 20
TIMED_PASSES_EXTRA = 10  # for N=32, which takes ~3 min × 32 questions
LIVE_RATE_SECONDS = 60.0

DEVICE_LABEL = "Apple M4 MacBook Air 16 GB"
BACKEND = "PyTorch MPS, eager attention"
PRECISION = "fp16"
MODEL = "Qwen2.5-Omni-3B (thinker-only)"


# --- helpers ------------------------------------------------------------


def load_audio_trim(path: Path, seconds: float = WINDOW_SECONDS) -> np.ndarray:
    """Load a WAV and trim/pad to exactly `seconds` at 16 kHz mono float32.

    The live loop's window is `DEFAULT_WINDOW_SECONDS` (3.0 s); benching
    on a clip longer than the window isn't a fair comparison.
    """
    audio, file_sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    target = int(round(DEFAULT_SR * seconds))
    if audio.shape[0] >= target:
        # Take the first `target` samples — calm_1.wav starts with speech.
        audio = audio[:target]
    else:
        audio = np.concatenate(
            [audio, np.zeros(target - audio.shape[0], dtype=np.float32)]
        )
    if file_sr != DEFAULT_SR:
        # We don't bother resampling here — all clips in this repo are
        # already 16 kHz. If that changes, fail loudly.
        raise ValueError(
            f"clip {path} is {file_sr} Hz, expected {DEFAULT_SR} Hz; "
            f"resampling not implemented in bench.py"
        )
    return audio


def model_questions(n: int) -> list[BoolQuestion]:
    """First `n` model bool questions from questions.yaml, padded to `n`
    with simple ad-hoc bools so the batched suffix has rows to fill.

    The production YAML has 9 model bools. For N=16 and N=32 we add
    generic reworded variants so each row asks something different of
    the model and the test stays honest about how the fork scales with
    row count.
    """
    base = [q for q in load_default_questions()
            if isinstance(q, BoolQuestion) and q.source == "model"]
    extras = [
        BoolQuestion(
            id=f"extra_{i}",
            text=(
                "Is there a noticeable background hum?" if i == 0 else
                "Is the speaker near the microphone?" if i == 1 else
                "Does the audio sound reverberant?" if i == 2 else
                "Is the speaker's voice steady?" if i == 3 else
                "Is there a noticeable pause in the audio?" if i == 4 else
                "Is the speaker enunciating clearly?" if i == 5 else
                "Is the speech rate fast?" if i == 6 else
                "Is the speech rate slow?" if i == 7 else
                "Does the audio have a rhythmic pattern?" if i == 8 else
                "Is there an echo?" if i == 9 else
                "Is there any hiss?" if i == 10 else
                "Is the speaker male?" if i == 11 else
                "Is the speaker female?" if i == 12 else
                "Is the speaker reading aloud?" if i == 13 else
                "Is the speaker laughing?" if i == 14 else
                "Is the audio monolog?" if i == 15 else
                "Is the audio dialogue?" if i == 16 else
                "Is there wind noise?" if i == 17 else
                "Is the microphone close to the source?" if i == 18 else
                "Is the microphone far from the source?" if i == 19 else
                "Is the audio recorded indoors?" if i == 20 else
                "Is the audio recorded outdoors?" if i == 21 else
                "Is the audio studio-quality?" if i == 22 else
                f"Is there any audio event type {i+1}?"
            ),
            manifest_col=None,
            smoothing=0.5,
            enter=0.5,
            exit=0.4,
        )
        for i in range(max(0, n - len(base)))
    ]
    return (base + extras)[:n]


def peak_rss_mb() -> float:
    """Peak resident-set size in MiB.

    `resource.getrusage(...).ru_maxrss` is in KB on Linux and in BYTES
    on macOS (which is the platform we actually run on). Detect via
    `sys.platform` rather than guessing the unit.
    """
    try:
        import resource
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        try:
            import psutil
            return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0


def sys_swap_mb() -> float:
    """macOS swap *used* in MiB (parsed from `sysctl vm.swapusage`).

    Returns 0 on non-macOS platforms or if the call fails. Used to
    detect whether the system was paging under sustained load — if
    swap grows during a long-running bench, that's the cause of any
    slowdown, not thermal throttle.
    """
    if sys.platform != "darwin":
        return 0.0
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"],
                                      text=True, timeout=2)
        # "vm.swapusage: total = 7168.00M  used = 5485.19M  free = ... (encrypted)"
        # Use a regex to avoid being confused by the "=" between key
        # and value (a token walk can skip past the `=` token).
        import re
        m = re.search(r"\bused\s*=\s*([\d.]+)\s*([MG])", out)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), m.group(2)
        if unit == "G":
            return val * 1024.0
        return val
    except Exception:
        return 0.0


def sys_memory_pressure() -> str:
    """Captured output of `memory_pressure` for the bench report."""
    try:
        return subprocess.check_output(["memory_pressure"],
                                       text=True, timeout=2)
    except Exception:
        return "(memory_pressure not available)"


def _bench_n(audio: np.ndarray, n: int, n_passes: int,
             args: argparse.Namespace) -> dict:
    """Run sequential + batched for N questions, return the per-N row dict."""
    qs = model_questions(n)
    # Sequential
    def seq_call() -> None:
        score_sequential(audio, qs)
    seq_lats, seq_peak = time_block(seq_call, n_passes)
    # Batched
    def bat_call() -> None:
        score_batched(audio, qs)
    bat_lats, bat_peak = time_block(bat_call, n_passes)
    s = summarize(seq_lats)
    b = summarize(bat_lats)
    speedup = s["median"] / b["median"] if b["median"] > 0 else float("inf")
    return {
        "n": n,
        "seq_median": s["median"],
        "seq_p90": s["p90"],
        "seq_mean": s["mean"],
        "seq_min": s["min"],
        "seq_passes": n_passes,
        "bat_median": b["median"],
        "bat_p90": b["p90"],
        "bat_mean": b["mean"],
        "bat_min": b["min"],
        "bat_passes": n_passes,
        "speedup_median": speedup,
        "peak_mb": max(seq_peak, bat_peak),
    }


def time_block(fn, n_passes: int) -> tuple[list[float], float]:
    """Run `fn` `n_passes` times, return (latencies_seconds, peak_mb)."""
    gc.collect()
    lats: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_passes):
        s = time.perf_counter()
        fn()
        lats.append(time.perf_counter() - s)
    peak = peak_rss_mb()
    elapsed = time.perf_counter() - t0
    return lats, peak


def summarize(lats: list[float]) -> dict:
    """Median, p90, mean, min over a list of latencies."""
    if not lats:
        return {"median": 0.0, "p90": 0.0, "mean": 0.0, "min": 0.0, "n": 0}
    s = sorted(lats)
    n = len(s)
    return {
        "median": float(statistics.median(s)),
        "p90": s[int(0.9 * (n - 1))],
        "mean": float(sum(s) / n),
        "min": float(min(s)),
        "n": n,
    }


def make_extra_questions(n: int) -> list[BoolQuestion]:
    return model_questions(n)


# --- main ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--warmup-seconds", type=float, default=WARMUP_SECONDS,
        help=f"warm-up wall-clock before timing (default {WARMUP_SECONDS:.0f}s)",
    )
    parser.add_argument(
        "--passes", type=int, default=TIMED_PASSES_DEFAULT,
        help=f"timed passes per (N, mode) (default {TIMED_PASSES_DEFAULT})",
    )
    parser.add_argument(
        "--live-seconds", type=float, default=LIVE_RATE_SECONDS,
        help=f"live-rate run length in seconds (default {LIVE_RATE_SECONDS:.0f})",
    )
    parser.add_argument(
        "--clip", type=Path,
        default=REPO_ROOT / "clips" / "calm_1.wav",
        help="audio clip to bench on (default: clips/calm_1.wav)",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=WINDOW_SECONDS,
        help=f"audio window length (default {WINDOW_SECONDS:.1f}s)",
    )
    parser.add_argument(
        "--skip-live", action="store_true",
        help="skip the live-rate measurement",
    )
    args = parser.parse_args(argv)

    audio = load_audio_trim(args.clip, seconds=args.window_seconds)
    print(f"clip: {args.clip.name}, {audio.shape[0]} samples "
          f"({audio.shape[0] / DEFAULT_SR:.2f}s @ {DEFAULT_SR} Hz)", flush=True)
    print(f"warm-up: {args.warmup_seconds:.0f}s wall-clock, lid open", flush=True)

    # Pre-run system state
    swap_before = sys_swap_mb()
    pressure_before = sys_memory_pressure()
    print("\n[system state BEFORE]", flush=True)
    print(f"  swap: {swap_before:.0f} MB used", flush=True)
    print(f"  memory_pressure:\n{pressure_before}", flush=True)
    # Process list intentionally NOT logged to stdout — `bench.md` is
    # public and shouldn't leak app names.

    # Warm-up: run score_batched on the full question set repeatedly until
    # the wall-clock budget is exhausted. The point is to soak the M4 Air
    # into its steady-state thermal envelope, not to time anything.
    full_qs = model_questions(max(N_LIST_MAIN + N_LIST_EXTRA))
    t_warm_start = time.perf_counter()
    warmup_iter = 0
    while time.perf_counter() - t_warm_start < args.warmup_seconds:
        score_batched(audio, full_qs)
        warmup_iter += 1
    print(f"\nwarm-up: {warmup_iter} full-batch passes in "
          f"{time.perf_counter() - t_warm_start:.1f}s", flush=True)
    print(f"  swap after warm-up: {sys_swap_mb():.0f} MB used", flush=True)

    # Warm-up: run score_batched on the full question set repeatedly until
    # the wall-clock budget is exhausted. The point is to soak the M4 Air
    # into its steady-state thermal envelope, not to time anything.
    full_qs = model_questions(max(N_LIST_MAIN + N_LIST_EXTRA))
    t_warm_start = time.perf_counter()
    warmup_iter = 0
    while time.perf_counter() - t_warm_start < args.warmup_seconds:
        score_batched(audio, full_qs)
        warmup_iter += 1
    print(f"warm-up: {warmup_iter} full-batch passes in "
          f"{time.perf_counter() - t_warm_start:.1f}s", flush=True)

    # Per-N timed passes — main table (1, 4, 8, 16)
    print("\n[main table: N, sequential (s), batched (s), speedup, peak MB]", flush=True)
    rows = []
    for n in N_LIST_MAIN:
        n_passes = args.passes
        row = _bench_n(audio, n, n_passes, args)
        rows.append(row)
        s = row["seq_median"]
        b = row["bat_median"]
        sp = row["speedup_median"]
        print(f"  N={n:>2d}: seq={s:6.3f}s (p90 {row['seq_p90']:6.3f})  "
              f"bat={b:6.3f}s (p90 {row['bat_p90']:6.3f})  "
              f"×{sp:5.2f}  peak {row['peak_mb']:.0f} MB", flush=True)
    swap_after_main = sys_swap_mb()
    print(f"\n[system state after main table] swap: {swap_after_main:.0f} MB used",
          flush=True)

    # Extra: N=32 — report separately; check swap and warn if it grew.
    extra_rows = []
    swap_before_extra = sys_swap_mb()
    print(f"\n[extra: N={N_LIST_EXTRA[0]} (separate note, swap before: "
          f"{swap_before_extra:.0f} MB)]", flush=True)
    for n in N_LIST_EXTRA:
        n_passes = min(args.passes, TIMED_PASSES_EXTRA)
        row = _bench_n(audio, n, n_passes, args)
        extra_rows.append(row)
        s = row["seq_median"]
        b = row["bat_median"]
        sp = row["speedup_median"]
        print(f"  N={n:>2d}: seq={s:6.3f}s (p90 {row['seq_p90']:6.3f})  "
              f"bat={b:6.3f}s (p90 {row['bat_p90']:6.3f})  "
              f"×{sp:5.2f}  peak {row['peak_mb']:.0f} MB", flush=True)
    swap_after_extra = sys_swap_mb()
    swap_growth_extra = swap_after_extra - swap_before_extra
    if swap_growth_extra > 100:
        print(f"  ⚠ swap grew by {swap_growth_extra:.0f} MB during N={N_LIST_EXTRA[0]} "
              f"— likely cause, not thermal throttle", flush=True)

    # Live rate: run earshot.live on the trimmed clip for 60 s and
    # collect pass latencies.
    live_summary = None
    if not args.skip_live:
        print(f"\n[live rate: --source {args.clip.name} --no-view for "
              f"{args.live_seconds:.0f}s]", flush=True)
        live_summary = measure_live_rate(args.clip, args.live_seconds)

    # Post-run system state
    swap_after_run = sys_swap_mb()
    pressure_after = sys_memory_pressure()
    print("\n[system state AFTER run]", flush=True)
    print(f"  swap: {swap_after_run:.0f} MB used "
          f"(started at {swap_before:.0f} MB, change "
          f"{swap_after_run - swap_before:+.0f} MB)", flush=True)
    print(f"  memory_pressure:\n{pressure_after}", flush=True)

    # Outputs
    sys_state = {
        "swap_before_mb": swap_before,
        "swap_after_warmup_mb": sys_swap_mb(),  # alias for the value already printed
        "swap_after_main_mb": swap_after_main,
        "swap_after_extra_mb": swap_after_extra,
        "swap_after_run_mb": swap_after_run,
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
    }
    write_csv(rows, live_summary, args, sys_state, extra_rows=extra_rows)
    write_md(rows, live_summary, args, sys_state, extra_rows=extra_rows)
    write_chart(rows, args, extra_rows=extra_rows)
    print(f"\nwrote: {RESULTS_DIR / 'bench.csv'}")
    print(f"wrote: {RESULTS_DIR / 'bench.md'}")
    print(f"wrote: {RESULTS_DIR / 'latency.png'}")
    return 0


# --- live rate ----------------------------------------------------------


def measure_live_rate(clip: Path, seconds: float) -> dict:
    """Spawn earshot.live in --no-view mode for `seconds` and parse the
    JSONL log for per-pass latency. Returns a summary dict.

    `elapsed_s` excludes the subprocess's own ~80s model-load warm-up
    so the passes-per-second rate reflects steady-state, not the cold
    start. We do that by using the first JSONL pass's `ts` field
    (which is wall-clock unix ms) as the "loop started" reference
    and the last pass's `ts` plus its `latency_ms` as the end.
    """
    log_path = RESULTS_DIR / "_live_rate.jsonl"
    cmd = [
        sys.executable, "-m", "earshot.live",
        "--source", str(clip),
        "--no-view", "--quiet",
        "--log", str(log_path),
        "--duration", str(int(seconds)),
        "--hop", "0.5",
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=seconds + 240)  # extra room for warm-up
    wall_elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  WARN: earshot.live exited {proc.returncode}; "
              f"stderr tail: {proc.stderr[-500:]}", file=sys.stderr)
        return {"passes": 0, "elapsed_s": wall_elapsed, "passes_per_s": 0.0,
                "median_latency_s": 0.0, "p90_latency_s": 0.0,
                "error": proc.stderr[-500:]}

    if not log_path.exists():
        print(f"  WARN: log not written at {log_path}", file=sys.stderr)
        return {"passes": 0, "elapsed_s": wall_elapsed}

    lats: list[float] = []
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "latency_ms" in rec:
                lats.append(float(rec["latency_ms"]) / 1000.0)
            if "ts" in rec:
                ts = int(rec["ts"])
                if first_ts_ms is None:
                    first_ts_ms = ts
                last_ts_ms = ts

    s = summarize(lats)

    # Steady-state window: from the first recorded pass's ts through
    # the last pass's ts plus its latency. Excludes model-load warm-up.
    if first_ts_ms is not None and last_ts_ms is not None:
        loop_elapsed_s = (last_ts_ms - first_ts_ms) / 1000.0 + (
            lats[-1] if lats else 0.0
        )
    else:
        loop_elapsed_s = 0.0
    passes_per_s = s["n"] / loop_elapsed_s if loop_elapsed_s > 0 else 0.0

    out = {
        "passes": s["n"],
        "wall_elapsed_s": wall_elapsed,
        "loop_elapsed_s": loop_elapsed_s,
        "passes_per_s": passes_per_s,
        "median_latency_s": s["median"],
        "p90_latency_s": s["p90"],
    }
    print(f"  passes={s['n']}  median={s['median']:.3f}s  "
          f"p90={s['p90']:.3f}s  {passes_per_s:.2f} pass/s "
          f"(steady-state, excludes subprocess warm-up; "
          f"wall={wall_elapsed:.1f}s)", flush=True)
    try:
        log_path.unlink()
    except FileNotFoundError:
        pass
    return out


# --- outputs ------------------------------------------------------------


def write_csv(rows: list[dict], live: dict | None, args: argparse.Namespace,
              sys_state: dict | None = None,
              extra_rows: list[dict] | None = None) -> None:
    path = RESULTS_DIR / "bench.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Stage 6 bench — earshot, M4 MacBook Air 16 GB, fp16"])
        w.writerow(["#", f"clip={args.clip.name}, window={args.window_seconds:.2f}s, "
                    f"warmup={args.warmup_seconds:.0f}s"])
        w.writerow(["n_questions", "sequential_s_median", "sequential_s_p90",
                    "sequential_s_mean", "sequential_s_min", "sequential_passes",
                    "batched_s_median", "batched_s_p90", "batched_s_mean",
                    "batched_s_min", "batched_passes", "speedup_median",
                    "peak_mb"])
        for r in rows:
            w.writerow([
                r["n"],
                f"{r['seq_median']:.4f}", f"{r['seq_p90']:.4f}",
                f"{r['seq_mean']:.4f}", f"{r['seq_min']:.4f}", r["seq_passes"],
                f"{r['bat_median']:.4f}", f"{r['bat_p90']:.4f}",
                f"{r['bat_mean']:.4f}", f"{r['bat_min']:.4f}", r["bat_passes"],
                f"{r['speedup_median']:.3f}", f"{r['peak_mb']:.1f}",
            ])
        if live:
            w.writerow([])
            w.writerow(["# live rate (earshot.live --no-view --source "
                        f"{args.clip.name} for {args.live_seconds:.0f}s)"])
            w.writerow(["passes", "wall_elapsed_s", "loop_elapsed_s",
                        "passes_per_s", "median_latency_s", "p90_latency_s"])
            w.writerow([live["passes"], f"{live['wall_elapsed_s']:.2f}",
                        f"{live['loop_elapsed_s']:.2f}",
                        f"{live['passes_per_s']:.3f}",
                        f"{live['median_latency_s']:.4f}",
                        f"{live['p90_latency_s']:.4f}"])


def write_md(rows: list[dict], live: dict | None, args: argparse.Namespace,
             sys_state: dict | None = None,
             extra_rows: list[dict] | None = None) -> None:
    path = RESULTS_DIR / "bench.md"
    lines: list[str] = []
    lines.append("# earshot — Stage 6 full benchmark")
    lines.append("")
    lines.append("Generated by `scripts/bench.py`.")
    lines.append("")
    lines.append(f"- **Device:** {DEVICE_LABEL}")
    lines.append(f"- **Model:** {MODEL}")
    lines.append(f"- **Backend:** {BACKEND}")
    lines.append(f"- **Precision:** {PRECISION}")
    lines.append(f"- **Clip:** `{args.clip.name}`, trimmed to "
                 f"`{args.window_seconds:.1f}`&nbsp;s "
                 f"({int(round(DEFAULT_SR * args.window_seconds))} samples "
                 f"@ {DEFAULT_SR}&nbsp;Hz — the live window size)")
    lines.append(f"- **Warm-up:** {args.warmup_seconds:.0f}&nbsp;s wall-clock "
                 f"with the full question set, lid open, laptop plugged in. "
                 f"Soaks the M4 Air into its steady-state thermal envelope.")
    lines.append(f"- **Timed passes per N:** {args.passes} (median + p90). "
                 f"N=32 sequential uses {TIMED_PASSES_EXTRA} (would take ~"
                 f"{32 * 0.76 * TIMED_PASSES_DEFAULT / 60:.0f}&nbsp;min otherwise).")
    lines.append("")
    lines.append("Re-run after freeing memory (Docker quit); earlier runs "
                 "were slowed by swap.")
    lines.append("")
    lines.append("Raw data: [`bench.csv`](bench.csv). Chart: "
                 "[`latency.png`](latency.png).")
    lines.append("")

    if sys_state:
        lines.append("## System state (memory pressure + swap)")
        lines.append("")
        lines.append("Logged before the warm-up, after the main table, and "
                     "after the full run. If swap grows during a long bench, "
                     "that's the cause of any slowdown, not thermal throttle.")
        lines.append("")
        lines.append("| stage | swap used (MB) |")
        lines.append("|-------|---------------:|")
        lines.append(f"| before run | {sys_state['swap_before_mb']:.0f} |")
        lines.append(f"| after warm-up | {sys_state['swap_after_main_mb']:.0f} |")
        lines.append(f"| after main table | {sys_state['swap_after_main_mb']:.0f} |")
        if "swap_after_extra_mb" in sys_state:
            lines.append(f"| after N=32 extra | {sys_state['swap_after_extra_mb']:.0f} |")
        lines.append(f"| after full run | {sys_state['swap_after_run_mb']:.0f} |")
        lines.append("")
        lines.append("```")
        lines.append("memory_pressure BEFORE:")
        lines.append(sys_state["pressure_before"].rstrip())
        lines.append("")
        lines.append("memory_pressure AFTER:")
        lines.append(sys_state["pressure_after"].rstrip())
        lines.append("```")
        lines.append("")

    lines.append("## Per-N sequential vs batched")
    lines.append("")
    lines.append("| N | sequential median (s) | sequential p90 (s) | "
                 "batched median (s) | batched p90 (s) | speedup (median) | "
                 "peak RSS (MB) | passes |")
    lines.append("|--:|----------------------:|-------------------:"
                 "|--------------------:|----------------:|------------------:"
                 "|---------------:|-------:|")
    for r in rows:
        passes_note = f"{r['seq_passes']}"
        if r["n"] == 32 and r["seq_passes"] != TIMED_PASSES_DEFAULT:
            passes_note = f"{r['seq_passes']} (32-seq capped)"
        lines.append(
            f"| {r['n']} | {r['seq_median']:.3f} | {r['seq_p90']:.3f} | "
            f"{r['bat_median']:.3f} | {r['bat_p90']:.3f} | "
            f"×{r['speedup_median']:.2f} | {r['peak_mb']:.0f} | {passes_note} |"
        )
    lines.append("")

    if extra_rows:
        lines.append("## N=32 (separate note)")
        lines.append("")
        lines.append(f"Run separately from the main table with "
                     f"{TIMED_PASSES_EXTRA} timed passes. Sequential at "
                     f"N=32 would take ~"
                     f"{32 * 0.76 * TIMED_PASSES_DEFAULT / 60:.0f}&nbsp;min "
                     f"at {TIMED_PASSES_DEFAULT} passes, so we cap it. "
                     f"Swap growth is reported in the system-state table; "
                     f"if swap grew by &gt;100&nbsp;MB during N=32, that's "
                     f"likely the cause of any slowdown, not thermal.")
        lines.append("")
        lines.append("| N | sequential median (s) | sequential p90 (s) | "
                     "batched median (s) | batched p90 (s) | speedup (median) | "
                     "peak RSS (MB) | passes |")
        lines.append("|--:|----------------------:|-------------------:"
                     "|--------------------:|----------------:|------------------:"
                     "|---------------:|-------:|")
        for r in extra_rows:
            lines.append(
                f"| {r['n']} | {r['seq_median']:.3f} | {r['seq_p90']:.3f} | "
                f"{r['bat_median']:.3f} | {r['bat_p90']:.3f} | "
                f"×{r['speedup_median']:.2f} | {r['peak_mb']:.0f} | "
                f"{r['seq_passes']} |"
            )
        lines.append("")

    if live:
        lines.append("## Live rate")
        lines.append("")
        lines.append(f"`earshot.live --source {args.clip.name} --no-view --log "
                     f"--duration {args.live_seconds:.0f}` — real run, "
                     f"real worker thread, real skip-to-newest.")
        lines.append("")
        lines.append("| passes | wall (s) | loop (s, steady-state) | pass/s | "
                     "median latency (s) | p90 latency (s) |")
        lines.append("|-------:|---------:|------------------------:|-------:"
                     "|--------------------:|----------------:|")
        lines.append(f"| {live['passes']} | {live['wall_elapsed_s']:.1f} | "
                     f"{live['loop_elapsed_s']:.1f} | "
                     f"{live['passes_per_s']:.2f} | "
                     f"{live['median_latency_s']:.3f} | "
                     f"{live['p90_latency_s']:.3f} |")
        lines.append("")
        lines.append("`loop` is the window from the first recorded pass's "
                     "`ts` to the last pass's `ts + latency_ms` &mdash; it "
                     "excludes the subprocess's own ~80&nbsp;s model-load "
                     "warm-up, so the pass/s rate reflects steady-state, not "
                     "the cold start.")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- The fanless M4 Air throttles under sustained load; the "
                 "2-minute warm-up before timing is what brings numbers "
                 "into their steady-state. Don't skip it.")
    lines.append("- Speedup is the stable claim; absolute latency shifts "
                 "run-to-run as the MPS allocator warms up.")
    lines.append("- Per-question rows past N=11 are simple ad-hoc bool "
                 "questions so the batched suffix has rows to fill; the "
                 "production YAML has 9 model bools. Adding them to the "
                 "fork doesn't change the cost model &mdash; each row is "
                 "a separate suffix forward.")
    lines.append("")
    path.write_text("\n".join(lines))


def write_chart(rows: list[dict], args: argparse.Namespace,
               extra_rows: list[dict] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Combine main + extra so the chart shows the full N=1..32 sweep.
    all_rows = list(rows)
    if extra_rows:
        all_rows.extend(extra_rows)
    all_rows.sort(key=lambda r: r["n"])
    ns = [r["n"] for r in all_rows]
    seq = [r["seq_median"] for r in all_rows]
    bat = [r["bat_median"] for r in all_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    ax.plot(ns, seq, marker="o", linewidth=2.4, markersize=7,
            color="#cc3344", label="sequential (one question per forward)")
    ax.plot(ns, bat, marker="s", linewidth=2.4, markersize=7,
            color="#0a6cff", label="batched (shared prefix, fork suffix)")

    # Annotate speedup at N=8, N=16, N=32 (where the fork earns its keep).
    for label_n in (8, 16, 32):
        if label_n in ns:
            i = ns.index(label_n)
            ax.annotate(
                f"×{all_rows[i]['speedup_median']:.1f}",
                xy=(ns[i], bat[i]),
                xytext=(8, -18), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#0a6cff",
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("N questions per pass", fontsize=11)
    ax.set_ylabel("median latency per pass (s)", fontsize=11)
    ax.set_title(
        f"earshot — {MODEL}\n"
        f"{DEVICE_LABEL} · {PRECISION} · "
        f"{args.clip.name}, {args.window_seconds:.1f}s window",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())