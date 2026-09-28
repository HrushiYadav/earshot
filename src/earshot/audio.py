"""Stage 1 — microphone capture.

A thread-safe ring buffer that always holds the last `window_seconds` of audio at
`sr` Hz (default: 16 kHz mono float32, 3.0 s = 48 000 samples). `get_window()`
returns exactly that many samples; the front is zero-padded until the buffer
fills.

Run it as a tiny demo with `python -m earshot.audio` to see a live RMS bar and
verify your microphone is wired up.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np


DEFAULT_SR = 16_000
DEFAULT_WINDOW_SECONDS = 3.0
SILENCE_THRESHOLD = 1e-6  # absolute amplitude — anything below counts as "silent"


@dataclass
class AudioCaptureConfig:
    sr: int = DEFAULT_SR
    window_seconds: float = DEFAULT_WINDOW_SECONDS


class AudioCapture:
    """Background mic capture into a thread-safe ring buffer.

    Usage:
        with AudioCapture() as cap:
            time.sleep(3.0)
            window = cap.get_window()   # np.ndarray, shape (48000,), float32
            rms = float(np.sqrt(np.mean(window ** 2)))
    """

    def __init__(self, sr: int = DEFAULT_SR, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.sr = sr
        self.window_seconds = window_seconds
        self.window_samples: int = int(round(sr * window_seconds))
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._overflowed = False
        self._frames_captured: int = 0  # how many samples the callback has written

    # ------- sounddevice callback -------

    def _callback(self, indata, frames: int, time_info, status) -> None:
        """Runs on the PortAudio thread. Keep it short and lock-free for long."""
        if status:
            # PortAudio reported an under/overflow — record it but keep going.
            self._overflowed = True
        # sounddevice gives shape (frames, channels). We asked for channels=1
        # but stay defensive in case the host delivers stereo.
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0] if indata.ndim > 1 else indata.reshape(-1)
        mono = mono.astype(np.float32, copy=False)
        with self._lock:
            if mono.shape[0] >= self.window_samples:
                # New chunk covers the whole window — discard history.
                self._buffer = mono[-self.window_samples:]
            else:
                # Slide: drop `mono.shape[0]` oldest samples, append the new ones.
                self._buffer = np.concatenate(
                    [self._buffer[mono.shape[0]:], mono]
                )
            self._frames_captured += mono.shape[0]

    # ------- public API -------

    def start(self) -> "AudioCapture":
        # sounddevice imported lazily so this module stays import-safe in
        # environments without a working PortAudio (e.g. some CI runners).
        import sounddevice as sd  # type: ignore[import-not-found]

        if self._stream is not None:
            return self
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioCapture":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def get_window(self) -> np.ndarray:
        """Return the last `window_seconds` of audio as exactly `window_samples`
        samples. If fewer than `window_samples` have arrived, the front is
        zero-padded (and remains zero — never anything else). The returned
        array is a copy — safe to keep and mutate."""
        with self._lock:
            return self._buffer.copy()

    def frames_captured(self) -> int:
        """How many samples the callback has written since start. Useful for
        telling apart "buffer still zero-padding" (small) from "real silence"
        (large but flat)."""
        with self._lock:
            return self._frames_captured

    def is_buffer_silent(self, threshold: float = SILENCE_THRESHOLD) -> bool:
        """True when every sample in the current window is below `threshold`
        in absolute value. After several seconds of running this catches both
        'nothing has arrived yet' and 'mic is dead / muted / denied'."""
        with self._lock:
            return bool(np.max(np.abs(self._buffer)) < threshold)

    @property
    def overflowed(self) -> bool:
        return self._overflowed


# ---------------------------------------------------------------------------
# Permission-error messaging (macOS-focused; useful elsewhere too)
# ---------------------------------------------------------------------------


def mic_permission_instructions() -> str:
    """Plain-text instructions for a user blocked by OS mic permission."""
    return (
        "Microphone capture is returning silence.\n"
        "\n"
        "On macOS, the OS has blocked this process from the microphone.\n"
        "Allow it once, and the permission sticks for this app forever:\n"
        "\n"
        "  System Settings  →  Privacy & Security  →  Microphone\n"
        "  →  enable 'Terminal' (or your Terminal app / IDE)\n"
        "\n"
        "Then quit and re-run this script. If the toggle is already on,\n"
        "toggle it off and back on to retrigger the prompt."
    )


# ---------------------------------------------------------------------------
# Live RMS bar demo
# ---------------------------------------------------------------------------


def _rms_bar(rms: float, width: int = 40) -> str:
    """Map a 0..~0.3 RMS range onto a width-N bar. log-ish so quiet speech shows up."""
    # 0 dBFS = 1.0 RMS. Speech typically sits around -30 to -10 dBFS
    # (~0.03 to 0.32). Compress with sqrt so the bar isn't silent until shouting.
    level = min(1.0, np.sqrt(rms * 8.0))
    filled = int(round(level * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {rms:.4f}"


def run_demo(sr: int = DEFAULT_SR, window_seconds: float = DEFAULT_WINDOW_SECONDS,
             refresh_hz: float = 12.0) -> int:
    """Live RMS level bar. Prints one frame per `1/refresh_hz` seconds and
    detects permission problems. Returns process exit code."""
    import sounddevice as sd  # type: ignore[import-not-found]

    print(f"earshot live demo — sr={sr}, window={window_seconds}s "
          f"({int(round(sr * window_seconds))} samples)", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    cap = AudioCapture(sr=sr, window_seconds=window_seconds)
    try:
        cap.start()
    except Exception as e:  # noqa: BLE001 — surface any backend error verbatim
        print(f"could not open mic: {type(e).__name__}: {e}", flush=True)
        print(mic_permission_instructions())
        return 1

    sleep_s = 1.0 / refresh_hz
    last_permission_check = 0.0
    permission_warned = False

    try:
        while True:
            window = cap.get_window()
            rms = float(np.sqrt(np.mean(window ** 2)))
            peak = float(np.max(np.abs(window)))
            frames = cap.frames_captured()
            tail = ""
            if cap.overflowed:
                tail += "  !overflow"
            sys.stdout.write(f"\r{_rms_bar(rms)} peak={peak:.3f}  "
                             f"frames={frames}  {tail}   ")
            sys.stdout.flush()

            # After ~1.5 s, if the buffer is still silent, surface the
            # permission message once.
            now = time.monotonic()
            if (not permission_warned
                    and now - last_permission_check > 1.5
                    and cap.is_buffer_silent()
                    and frames > 0):
                print("\n", flush=True)
                print(mic_permission_instructions())
                permission_warned = True
            last_permission_check = now

            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nbye.", flush=True)
    finally:
        cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
