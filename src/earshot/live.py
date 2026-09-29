"""Stage 5 — live loop and dashboard.

The loop runs every `hop` seconds (default 0.5). The audio source (mic or
file) is read on the main thread; the model inference runs in a worker
thread so audio capture never stalls. If a pass takes longer than the
hop, the next pass drops the in-flight request and starts over with
the newest window (skip-to-newest; never queue old audio).

Per question: EMA smoothing (alpha from YAML) followed by enter/exit
hysteresis to decide the fired state — no flicker around the threshold.

Tier rule: if the silent signal fires, main_sound is shown with top =
"silence" and the model's probs are dimmed (greyed). The model still
runs and the log captures its full distribution.

CLI:
  python -m earshot.live                            # mic, default YAML, 0.5s
  python -m earshot.live --source clips/calm_1.wav   # file mode (loop)
  python -m earshot.live --ask "Is someone laughing?" # ad-hoc extra question
  python -m earshot.live --log run.jsonl --no-view   # headless, log only

A typed JSONL line is written per pass when --log is set; format:

  {"ts": <unix_ms>, "latency_ms": <float>,
   "results": {"<qid>": {"p": <float>, "fired": <bool>, "top": <str>, ...}}}
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .audio import (
    DEFAULT_SR,
    DEFAULT_WINDOW_SECONDS,
    AudioCapture,
)
from .prompts import load_default_questions
from .schema import BoolQuestion, ChoiceQuestion, Question
from .scorer import (
    BoolResult,
    ChoiceResult,
    score_batched,
)


# ---------------------------------------------------------------------------
# Audio sources — both expose the same get_window() shape
# ---------------------------------------------------------------------------


class AudioSource:
    """Abstract: returns a fresh copy of the last `window_seconds` of audio."""

    def get_window(self) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class MicSource(AudioSource):
    """Live microphone capture via sounddevice's PortAudio binding."""

    def __init__(self, sr: int = DEFAULT_SR, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self._cap = AudioCapture(sr=sr, window_seconds=window_seconds)

    def start(self) -> "MicSource":
        try:
            self._cap.start()
        except Exception as e:  # noqa: BLE001
            from .audio import mic_permission_instructions
            print(f"could not open mic: {type(e).__name__}: {e}", file=sys.stderr)
            print(mic_permission_instructions())
            raise
        return self

    def get_window(self) -> np.ndarray:
        return self._cap.get_window()

    def stop(self) -> None:
        self._cap.stop()


class FileSource(AudioSource):
    """Looped WAV-file source for reproducible demos and debugging.

    A background thread pushes 50 ms chunks into a ring buffer; get_window
    returns the last `window_seconds` of audio exactly like the mic source.
    The file is looped (when the file ends it restarts from the start).
    """

    CHUNK_MS = 50

    def __init__(self, path: str, sr: int = DEFAULT_SR, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.path = path
        self.sr = sr
        self.window_seconds = window_seconds
        self.window_samples = int(round(sr * window_seconds))
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._lock = threading.Lock()
        audio, file_sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self._audio = audio.astype(np.float32, copy=False)
        self._pos = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._feed_loop, daemon=True, name="FileSource")

    def start(self) -> "FileSource":
        if not self._thread.is_alive():
            self._thread.start()
        return self

    def _feed_loop(self) -> None:
        chunk_size = max(1, int(self.sr * self.CHUNK_MS / 1000))
        while not self._stopped:
            chunk = self._read_chunk(chunk_size)
            with self._lock:
                if chunk.shape[0] >= self.window_samples:
                    self._buffer = chunk[-self.window_samples:].copy()
                else:
                    self._buffer = np.concatenate(
                        [self._buffer[chunk.shape[0]:], chunk]
                    ).astype(np.float32, copy=False)
            time.sleep(self.CHUNK_MS / 1000.0)

    def _read_chunk(self, n: int) -> np.ndarray:
        if self._pos + n > len(self._audio):
            # Loop: take the tail of the file then wrap.
            tail = self._audio[self._pos:]
            head_needed = n - len(tail)
            head = self._audio[:head_needed] if head_needed > 0 else np.empty(0, dtype=np.float32)
            self._pos = head_needed
            return np.concatenate([tail, head]).astype(np.float32, copy=False)
        chunk = self._audio[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def get_window(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    def stop(self) -> None:
        self._stopped = True


def open_audio_source(spec: str, sr: int = DEFAULT_SR,
                     window_seconds: float = DEFAULT_WINDOW_SECONDS) -> AudioSource:
    """Resolve --source argument. "mic" -> MicSource; anything else -> FileSource."""
    if spec == "mic":
        return MicSource(sr=sr, window_seconds=window_seconds)
    return FileSource(spec, sr=sr, window_seconds=window_seconds)


# ---------------------------------------------------------------------------
# Inference worker — model runs here so the audio loop never stalls
# ---------------------------------------------------------------------------


class InferenceWorker(threading.Thread):
    """Runs score_batched on a single request slot. Latest request wins; the
    main loop only ever has one outstanding result to collect."""

    def __init__(self, questions: list[Question]):
        super().__init__(daemon=True, name="InferenceWorker")
        self._cond = threading.Condition()
        self._req_audio: np.ndarray | None = None
        self._req_questions: list[Question] | None = None
        self._request_pending = False
        self._result = None  # dict[qid, BoolResult|ChoiceResult]
        self._latency_ms: float = 0.0
        self._result_ready = False
        self._stopped = False
        self._questions = questions
        self.errors: int = 0  # how many inference calls have raised

    def submit(self, audio: np.ndarray) -> None:
        """Drop the latest audio window into the slot. Overwrites any pending
        request — that's the skip-to-newest semantics.

        NOTE: do NOT reset `_result_ready` here. The worker may have just
        published a finished result that the main thread hasn't collected yet;
        resetting the flag would silently drop it. Skip-to-newest is on the
        AUDIO side; results are reported as they finish.
        """
        with self._cond:
            self._req_audio = audio
            self._request_pending = True
            self._cond.notify()

    def try_collect(self) -> tuple[dict, float] | None:
        """Non-blocking: return (results, latency_ms) if the worker just
        finished a pass; otherwise None (worker is still busy)."""
        with self._cond:
            if not self._result_ready:
                return None
            res = (self._result, self._latency_ms)
            self._result_ready = False
            return res

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._request_pending and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                audio = self._req_audio
                self._request_pending = False
                questions = self._questions
            # Run inference OUTSIDE the lock so a fresh submit() during
            # this pass can drop a new request in without blocking.
            t0 = time.perf_counter()
            try:
                results = score_batched(audio, questions)
            except Exception as e:  # noqa: BLE001 — surface in counter
                self.errors += 1
                print(f"[worker] inference error: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0
            with self._cond:
                self._result = results
                self._latency_ms = latency_ms
                self._result_ready = True
                self._cond.notify()


# ---------------------------------------------------------------------------
# Per-question live state: EMA + hysteresis
# ---------------------------------------------------------------------------


@dataclass
class QuestionState:
    """Per-question smoothing + fired state + event log."""

    smoothed_p: float = 0.5
    fired: bool = False
    smoothed_top: str | None = None  # for choice questions
    smoothed_probs: dict[str, float] = field(default_factory=dict)

    def update(self, res: BoolResult | ChoiceResult, alpha: float,
              enter: float | None = None, exit: float | None = None) -> str | None:
        """Apply EMA + hysteresis. Returns the event string ("YES→NO", "NO→YES",
        "<top1>→<top2>") if the fired state or top choice changed, else None.
        `enter`/`exit` are only consulted for BoolResult; pass None for
        ChoiceResult."""
        prev_fired = self.fired
        prev_top = self.smoothed_top
        if isinstance(res, BoolResult):
            self.smoothed_p = alpha * res.p + (1 - alpha) * self.smoothed_p
            # Hysteresis: above enter fires (when off); below exit un-fires
            # (when on); in between stays. enter/exit default to 0.5/0.4 if
            # not provided (shouldn't happen for bool, but be safe).
            _enter = enter if enter is not None else 0.5
            _exit = exit if exit is not None else 0.4
            if not prev_fired and self.smoothed_p >= _enter:
                self.fired = True
            elif prev_fired and self.smoothed_p <= _exit:
                self.fired = False
        else:
            self.smoothed_p = alpha * res.probs.get(res.top, 0.0) + (1 - alpha) * self.smoothed_p
            # EMA each option's probability.
            for opt, p in res.probs.items():
                self.smoothed_probs[opt] = alpha * p + (1 - alpha) * self.smoothed_probs.get(opt, p)
            self.smoothed_top = max(self.smoothed_probs, key=self.smoothed_probs.get)

        if isinstance(res, BoolResult):
            if self.fired != prev_fired:
                return f"{'YES' if self.fired else 'NO'}"
        else:
            if self.smoothed_top != prev_top:
                return f"{prev_top or '?'}→{self.smoothed_top}"
        return None


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------


@dataclass
class LiveStats:
    passes: int = 0
    last_latency_ms: float = 0.0
    rolling_latencies: deque = field(default_factory=lambda: deque(maxlen=20))


def run_live(
    questions: list[Question],
    audio_source: AudioSource,
    hop: float = 0.5,
    log_path: str | None = None,
    view: bool = True,
    duration_seconds: float | None = None,
    quiet: bool = False,
) -> int:
    """Main entry: run the live loop until interrupted (or `duration_seconds`
    elapses for the test harness).

    `view=True` -> render the rich dashboard.
    `log_path` -> JSONL log file (one line per pass).
    """
    # Build per-question state and lookup by id
    states: dict[str, QuestionState] = {q.id: QuestionState() for q in questions}
    questions_by_id: dict[str, Question] = {q.id: q for q in questions}
    # Warm state with one neutral pass so first-pass EMA isn't a fade-in from 0.5.
    for q in questions:
        if isinstance(q, ChoiceQuestion):
            states[q.id].smoothed_probs = {opt: 1.0 / len(q.options) for opt in q.options}
        else:
            states[q.id].smoothed_p = 0.5

    log_fh = None
    if log_path:
        log_fh = open(log_path, "w")

    # Start the audio source and the inference worker
    audio_source.start()
    worker = InferenceWorker(questions)
    worker.start()

    # Single-shot Ctrl-C: register a SIGINT handler that flips a stop flag.
    # Python's default raises KeyboardInterrupt; rich's Live display eats
    # the first SIGINT mid-render, so 4 presses were needed. With a flag,
    # the next loop tick exits cleanly.
    stop_requested = threading.Event()
    _prev_sigint = signal.getsignal(signal.SIGINT)

    def _request_stop(signum, frame):
        stop_requested.set()

    try:
        signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, OSError):
        # Not on the main thread (shouldn't happen) or unsupported
        # platform; fall back to KeyboardInterrupt.
        pass

    # Initial event-line stream
    events: deque = deque(maxlen=5)
    stats = LiveStats()
    last_tick = 0.0

    # Compute a single warm-up pass so MPS kernels are compiled before we
    # start showing the dashboard. Per Stage 0 the first forward takes
    # ~75 s of compilation.
    print("loading model and warming up …", flush=True)
    warm_audio = audio_source.get_window()
    worker.submit(warm_audio)
    # Block until the warm-up is done so the dashboard's first frame
    # is meaningful.
    while True:
        res = worker.try_collect()
        if res is not None:
            warm_results, warm_latency = res
            # Seed the EMA state from the warm-up so the first live frame
            # doesn't open with everything at 0.5.
            for q in questions:
                r = warm_results.get(q.id)
                if r is None:
                    continue
                st = states[q.id]
                if isinstance(r, BoolResult):
                    st.smoothed_p = r.p
                    st.fired = (r.p >= q.enter)
                else:
                    st.smoothed_probs = dict(r.probs)
                    st.smoothed_top = r.top
                    st.smoothed_p = r.probs.get(r.top, 0.0)
            break
        time.sleep(0.05)
    print(f"warm-up done in {warm_latency:.0f} ms", flush=True)

    # Start the loop clock AFTER warm-up so --duration counts only live passes.
    loop_start_t = time.monotonic()
    end_at = (loop_start_t + duration_seconds) if duration_seconds else None

    # Dashboard — only built if view=True
    if view:
        from rich.console import Console
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich.align import Align

        console = Console()

        # Sized panels: each Panel draws top + bottom border + 1 content
        # row minimum, so title/footer need 3 lines, events need
        # (5 events + 2 borders) = 7 lines, questions fill the rest.
        title_h = 3
        events_h = 7
        footer_h = 3
        questions_h = max(8, len(questions) + 2)  # 1 row per q + 2 borders

        def render() -> Layout:
            root = Layout()
            root.split_column(
                Layout(name="title", size=title_h),
                Layout(name="questions", size=questions_h),
                Layout(name="events", size=events_h),
                Layout(name="footer", size=footer_h),
            )
            title = Text.assemble(
                ("earshot — ", "bold magenta"),
                (f"{len(questions)} questions ", "bold"),
                ("every ", "dim"),
                (f"{hop:.2f}s, ", "bold"),
                ("fully local on M4 MacBook Air", "dim"),
            )
            root["title"].update(Panel(
                Align.left(title, vertical="middle"),
                style="white on blue", border_style="blue",
            ))

            # Question rows
            table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
            table.add_column("label", style="bold", width=18)
            table.add_column("bar", width=22)
            table.add_column("p", width=14, justify="right")
            table.add_column("answer", width=12)
            table.add_column("source", width=8)

            for q in questions:
                st = states[q.id]
                src_tag = "signal" if isinstance(q, BoolQuestion) and q.source == "signal" else "model"
                src_styled = f"[dim cyan]{src_tag}[/dim cyan]"

                if isinstance(q, BoolQuestion):
                    p = st.smoothed_p
                    bar_len = 20
                    filled = max(0, min(bar_len, int(round(p * bar_len))))
                    # Visible styled bar: green when fired, dim when not.
                    bar_style = "bold green" if st.fired else "dim white"
                    bar_text = Text()
                    bar_text.append("█" * filled, style=bar_style)
                    bar_text.append("░" * (bar_len - filled), style="dim")
                    fired_styled = ("[bold reverse green] YES [/bold reverse green]"
                                    if st.fired else "[dim]no[/dim]")
                    table.add_row(
                        q.id, bar_text,
                        f"{p:.3f}",
                        fired_styled, src_styled,
                    )
                else:
                    # main_sound (or any choice): show top + probabilities
                    top = st.smoothed_top or q.options[0]
                    top_styled = f"[bold reverse cyan] {top} [/bold reverse cyan]"
                    probs_str = "  ".join(
                        f"[{'bold' if o == top else 'dim'}]{o}={st.smoothed_probs.get(o, 0):.2f}[/]"
                        for o in q.options
                    )
                    table.add_row(
                        q.id, probs_str, f"{st.smoothed_probs.get(top, 0):.3f}",
                        top_styled, src_styled,
                    )

            root["questions"].update(Panel(
                table, title="questions", border_style="blue",
            ))

            # Events panel
            ev_lines = []
            for ts, msg in events:
                ev_lines.append(f"[dim]{ts}[/dim]  {msg}")
            if not ev_lines:
                ev_lines.append("[dim](no events yet)[/dim]")
            ev_text = Text.from_markup("\n".join(ev_lines))
            root["events"].update(Panel(
                ev_text,
                title="recent events", border_style="blue",
            ))

            # Footer (always visible — fixed 3 rows)
            passes = max(1, stats.passes)
            avg_latency = (sum(stats.rolling_latencies) / len(stats.rolling_latencies)
                           if stats.rolling_latencies else 0.0)
            elapsed = max(1e-3, time.monotonic() - loop_start_t)
            passes_per_sec = passes / elapsed
            footer = Text.assemble(
                (f"latency {stats.last_latency_ms:.0f} ms  ", "dim"),
                ("·  ", "dim"),
                (f"avg {avg_latency:.0f} ms  ", "dim"),
                ("·  ", "dim"),
                (f"passes {stats.passes}  ", "dim"),
                ("·  ", "dim"),
                (f"~{passes_per_sec:.2f} pass/s  ", "dim"),
                ("·  ", "dim"),
                (f"{len(questions)} q/pass", "dim"),
            )
            root["footer"].update(Panel(
                Align.left(footer, vertical="middle"),
                style="white on blue", border_style="blue",
            ))
            return root

        try:
            with Live(render(), console=console, refresh_per_second=12,
                      screen=False, transient=False) as live_ctx:
                while not stop_requested.is_set():
                    now = time.monotonic()
                    if end_at and now >= end_at:
                        break
                    if now - last_tick < hop:
                        time.sleep(max(0.0, hop - (now - last_tick)))
                        continue
                    last_tick = now
                    window = audio_source.get_window()
                    rms = float(np.sqrt(np.mean(window ** 2)))
                    worker.submit(window)
                    # Non-blocking collect of previous result; if still
                    # busy we just render the dashboard with stale state
                    # (skip-to-newest).
                    res = worker.try_collect()
                    if res is None:
                        live_ctx.update(render())
                        continue
                    results, latency = res
                    stats.passes += 1
                    stats.last_latency_ms = latency
                    stats.rolling_latencies.append(latency)

                    # Detect silent-fired (tier override signal) BEFORE
                    # updating question state, so main_sound can be dimmed.
                    silent_fired = False
                    if "silent" in results:
                        st = states["silent"]
                        old = st.fired
                        new_event = st.update(results["silent"], alpha=questions_by_id["silent"].smoothing,
                                              enter=questions_by_id["silent"].enter,
                                              exit=questions_by_id["silent"].exit)
                        silent_fired = st.fired
                        if new_event is not None:
                            events.append((time.strftime("%H:%M:%S"), f"silent {old and 'YES' or 'no'}→{'YES' if st.fired else 'no'}"))

                    # Update all other questions
                    for q in questions:
                        if q.id == "silent":
                            continue
                        r = results.get(q.id)
                        if r is None:
                            continue
                        st = states[q.id]
                        old_fired = st.fired
                        old_top = st.smoothed_top
                        if isinstance(r, BoolResult):
                            evt = st.update(r, alpha=q.smoothing, enter=q.enter, exit=q.exit)
                        else:
                            evt = st.update(r, alpha=q.smoothing)
                        if evt is not None:
                            if isinstance(q, BoolQuestion):
                                events.append((time.strftime("%H:%M:%S"),
                                               f"{q.id} {old_fired and 'YES' or 'no'}→{'YES' if st.fired else 'no'}"))
                            else:
                                events.append((time.strftime("%H:%M:%S"),
                                               f"{q.id} top {old_top or '?'}→{st.smoothed_top}"))
                        # Tier rule: if silent fires, mark main_sound
                        # as overridden (renderer dims it).
                        if q.id == "main_sound" and silent_fired:
                            st.smoothed_top = "silence"

                    # Log line (p_raw vs p_smoothed for hysteresis auditability)
                    if log_fh is not None:
                        line = {
                            "ts": int(time.time() * 1000),
                            "latency_ms": round(latency, 2),
                            "rms": round(rms, 5),
                            "results": {
                                qid: _result_to_json(results[qid], states.get(qid)) for qid in results
                            },
                        }
                        log_fh.write(json.dumps(line) + "\n")
                        log_fh.flush()

                    live_ctx.update(render())
        finally:
            signal.signal(signal.SIGINT, _prev_sigint)
    else:
        # Headless (no-view) mode — same logic, no rendering.
        try:
            while not stop_requested.is_set():
                now = time.monotonic()
                if end_at and now >= end_at:
                    break
                if now - last_tick < hop:
                    time.sleep(max(0.0, hop - (now - last_tick)))
                    continue
                last_tick = now
                window = audio_source.get_window()
                rms = float(np.sqrt(np.mean(window ** 2)))
                worker.submit(window)
                res = worker.try_collect()
                if res is None:
                    continue
                results, latency = res
                stats.passes += 1
                stats.last_latency_ms = latency
                stats.rolling_latencies.append(latency)

                silent_fired = False
                silent_q = questions_by_id.get("silent")
                if silent_q is not None and "silent" in results:
                    states["silent"].update(
                        results["silent"],
                        alpha=silent_q.smoothing,
                        enter=silent_q.enter,
                        exit=silent_q.exit,
                    )
                    silent_fired = states["silent"].fired
                for q in questions:
                    if q.id == "silent":
                        continue
                    r = results.get(q.id)
                    if r is None:
                        continue
                    st = states[q.id]
                    if isinstance(r, BoolResult):
                        st.update(r, alpha=q.smoothing, enter=q.enter, exit=q.exit)
                    else:
                        st.update(r, alpha=q.smoothing)
                    if q.id == "main_sound" and silent_fired:
                        st.smoothed_top = "silence"
                if log_fh is not None:
                    line = {
                        "ts": int(time.time() * 1000),
                        "latency_ms": round(latency, 2),
                        "rms": round(rms, 5),
                        "results": {
                            qid: _result_to_json(results[qid], states.get(qid)) for qid in results
                        },
                    }
                    log_fh.write(json.dumps(line) + "\n")
                    log_fh.flush()
                if quiet:
                    continue
                # One-line progress print (non-flickering, with \r)
                summary = " ".join(
                    f"{q.id}={_state_summary(states[q.id], q)}" for q in questions
                )
                sys.stdout.write(
                    f"\rpass {stats.passes:>4d}  {latency:5.0f}ms  rms={rms:.4f}  {summary}   "
                )
                sys.stdout.flush()
            if not quiet:
                sys.stdout.write("\n")
        finally:
            signal.signal(signal.SIGINT, _prev_sigint)

    audio_source.stop()
    if log_fh is not None:
        log_fh.close()
    return 0


def _result_to_json(res: BoolResult | ChoiceResult,
                   state: QuestionState | None = None) -> dict:
    """Serialize a typed result for the JSONL log. For bool results, also
    emit `p_smoothed` (the post-EMA smoothed P that hysteresis is applied
    to) alongside `p` (the model's raw P(Yes) for this pass) so the log is
    auditable — you can see exactly when hysteresis toggled `fired`."""
    if isinstance(res, BoolResult):
        out: dict = {"p": round(res.p, 5), "fired": res.fired, "source": res.source}
        if state is not None:
            out["p_smoothed"] = round(state.smoothed_p, 5)
        return out
    return {
        "probs": {k: round(v, 5) for k, v in res.probs.items()},
        "top": res.top,
        "source": res.source,
    }


def _state_summary(state: QuestionState, q: Question) -> str:
    if isinstance(q, BoolQuestion):
        return f"{state.smoothed_p:.2f}{'Y' if state.fired else '·'}"
    return state.smoothed_top or "?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_questions(args: argparse.Namespace) -> list[Question]:
    qs = load_default_questions()
    # --ask: ad-hoc bool model questions layered on top of the YAML.
    for i, ask_text in enumerate(args.ask or []):
        qid = f"ask_{i}"
        qs.append(BoolQuestion(
            id=qid,
            text=ask_text,
            manifest_col=None,  # not in the manifest, eval skips
            smoothing=0.5,
            enter=0.5,
            exit=0.4,
        ))
    return qs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", default="mic",
                        help='"mic" (default) or a path to a .wav file (looped)')
    parser.add_argument("--questions", default=None,
                        help="path to questions.yaml (default: repo root)")
    parser.add_argument("--hop", type=float, default=0.5,
                        help="seconds between passes (default 0.5)")
    parser.add_argument("--ask", action="append", default=[],
                        help="ad-hoc bool model question (repeatable)")
    parser.add_argument("--log", default=None,
                        help="JSONL log path (one line per pass)")
    parser.add_argument("--no-view", action="store_true",
                        help="headless: no dashboard, just the log")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds (test harness)")
    parser.add_argument("--quiet", action="store_true",
                        help="in --no-view mode, don't print per-pass summary either")
    args = parser.parse_args(argv)

    # Override the loaded YAML path if given.
    if args.questions:
        from .schema import load_questions
        # Patch the default loader used by _resolve_questions
        import earshot.prompts as prompts
        prompts.load_default_questions = lambda: load_questions(args.questions)  # type: ignore[assignment]

    qs = _resolve_questions(args)

    # Open the audio source
    try:
        audio_source = open_audio_source(args.source)
    except Exception as e:
        print(f"could not open source {args.source!r}: {e}", file=sys.stderr)
        return 1

    return run_live(
        questions=qs,
        audio_source=audio_source,
        hop=args.hop,
        log_path=args.log,
        view=not args.no_view,
        duration_seconds=args.duration,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())