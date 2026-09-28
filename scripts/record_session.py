#!/usr/bin/env python3
"""Stage 1b — record a single labelled test clip per invocation.

Walks the user through ONE clip at a time. For each invocation:

  1. prints the instruction (and any sentence to read aloud),
  2. waits for the user to press Enter,
  3. runs a 3-2-1 countdown,
  4. records `seconds` (default 5 s) of mono float32 audio at 16 kHz,
  5. (optionally) plays the clip back,
  6. prints RMS / peak and warns if the level doesn't match expectations,
  7. asks keep / redo; on 'keep' upserts that clip's row into clips/manifest.csv,
  8. exits with status 0 (keep) / 0 (redo) / non-zero on bad input.

Usage:
    uv run scripts/record_session.py --list
    uv run scripts/record_session.py --clip silence_1
    uv run scripts/record_session.py --clip calm_1 --seconds 5 --debug
    uv run scripts/record_session.py --clip clap_1 --no-playback

The clips/ directory and its manifest are gitignored — only this script is
committed. Resume support is automatic via --list (already-recorded labels show ✓).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np
import sounddevice as sd
import soundfile as sf

from earshot.audio import (
    AudioCapture,
    DEFAULT_SR,
    mic_permission_instructions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO_ROOT / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.csv"
SR = DEFAULT_SR
DEFAULT_SECONDS = 5.0
COUNTDOWN_SECONDS = 3

# ---------------------------------------------------------------------------
# Manifest columns
# ---------------------------------------------------------------------------

MANIFEST_HEADER = [
    "file", "label", "description",
    "is_speaking", "multiple_speakers", "music", "angry", "said_stop",
    "clapping", "phone_or_alarm", "silent", "main_sound",
]


# ---------------------------------------------------------------------------
# Auto-play primitives
# ---------------------------------------------------------------------------

@dataclass
class AutoPlay:
    kind: str            # "afplay" (file) or "say" (TTS)
    path: str = ""       # for kind="afplay"
    text: str = ""       # for kind="say"
    voice: str = ""      # for kind="say"
    repeat: bool = False # when True (afplay only), loop until killed


VOICE_FEMALE_US = "Samantha"
VOICE_MALE_UK = "Daniel"
VOICE_FEMALE_AU = "Karen"

ALARM_SOUND = "/System/Library/Sounds/Sosumi.aiff"

SAY_FIRST_VOICE = "Hello there, I would like to talk about the weather today. It has been raining for hours."
SAY_SECOND_VOICE = "I am a completely different voice. Listen carefully, the cadence should sound slightly different."


class AutoPlayer:
    """Manage timed subprocess sounds during a recording window.

    Each spec is launched by `subprocess.Popen`. `stop()` terminates
    everything. The record of (spec, Popen, start_wall_time) is exposed
    via `records()` for --debug output.
    """

    def __init__(self, specs: list[AutoPlay], total_seconds: float, debug: bool = False):
        self.specs = list(specs)
        self.total_seconds = total_seconds
        self._debug = debug
        self._procs: list[subprocess.Popen] = []
        self._records: list[tuple[AutoPlay, subprocess.Popen, float]] = []
        self._timers: list[threading.Timer] = []
        if not specs:
            return
        any_repeat = any(s.repeat for s in specs)
        env = {**os.environ, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"}
        if any_repeat:
            for spec in specs:
                self._spawn(spec, env)
        else:
            n = max(1, len(specs))
            each = total_seconds / n
            for i, spec in enumerate(specs):
                delay = i * each
                if delay <= 0:
                    self._spawn(spec, env)
                else:
                    t = threading.Timer(delay, self._spawn, args=(spec, env))
                    t.daemon = True
                    t.start()
                    self._timers.append(t)

    def _spawn(self, spec: AutoPlay, env: dict) -> None:
        if spec.kind == "afplay":
            cmd = ["afplay", spec.path]
            if spec.repeat:
                # bash loop so it keeps playing afplay until killed.
                cmd = ["bash", "-lc", f"while true; do afplay {spec.path!r} 2>/dev/null; done"]
        elif spec.kind == "say":
            cmd = ["say", "-v", spec.voice, spec.text]
        else:
            return
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            self._procs.append(p)
            self._records.append((spec, p, time.time()))
            if self._debug:
                sys.stdout.write(f"  [debug] spawned: {cmd[0]} (pid={p.pid})\n")
        except Exception as e:
            if self._debug:
                sys.stdout.write(f"  [debug] spawn failed for {spec.kind}: {e}\n")

    def stop(self) -> None:
        for t in self._timers:
            t.cancel()
        for p in self._procs:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(p.pid), 15)
            except Exception:
                pass

    def records(self) -> list[dict]:
        """Return a snapshot of what we spawned and whether each finished naturally."""
        out = []
        for spec, p, t_start in self._records:
            rc = p.poll()
            out.append({
                "kind": spec.kind,
                "voice": spec.voice or "",
                "path": spec.path or "",
                "text": spec.text or "",
                "repeat": spec.repeat,
                "pid": p.pid,
                "start_wall": t_start,
                "finished": rc is not None,
                "returncode": rc,
            })
        return out


# ---------------------------------------------------------------------------
# Clip definitions
# ---------------------------------------------------------------------------

@dataclass
class ClipDef:
    label: str
    description: str
    sentences: list[str] = field(default_factory=list)
    auto_play: list[AutoPlay] = field(default_factory=list)
    notes: str = ""
    expected: dict[str, str] = field(default_factory=dict)


def _silence() -> dict:
    return {
        "is_speaking": "No", "multiple_speakers": "No", "music": "No",
        "angry": "No", "said_stop": "No", "clapping": "No",
        "phone_or_alarm": "No", "silent": "Yes", "main_sound": "silence",
    }


def _speech_yes(extras: dict | None = None) -> dict:
    base = {
        "is_speaking": "Yes", "multiple_speakers": "No", "music": "No",
        "angry": "No", "said_stop": "No", "clapping": "No",
        "phone_or_alarm": "No", "silent": "No", "main_sound": "speech",
    }
    if extras:
        base.update(extras)
    return base


def _noise(extras: dict | None = None) -> dict:
    base = {
        "is_speaking": "No", "multiple_speakers": "No", "music": "No",
        "angry": "No", "said_stop": "No", "clapping": "No",
        "phone_or_alarm": "No", "silent": "No", "main_sound": "noise",
    }
    if extras:
        base.update(extras)
    return base


CLIPS: list[ClipDef] = [
    # ---- silence / room noise ----
    ClipDef(
        label="silence_1",
        description="Stay completely quiet. Don't make any sound.",
        expected=_silence(),
    ),
    ClipDef(
        label="silence_2",
        description="Same — second silence take.",
        expected=_silence(),
    ),
    ClipDef(
        label="room_noise_1",
        description=(
            "Stay quiet. If you have a fan / AC, leave it on. "
            "Otherwise normal room ambience is fine."
        ),
        expected=_silence(),
        notes="Ambient (fans/AC) is still 'silent' for these questions.",
    ),

    # ---- calm speech ----
    ClipDef(
        label="calm_1",
        description="Read the line below CALMLY at normal volume:",
        sentences=["The package arrived two days before I expected it, and the contents were all intact."],
        expected=_speech_yes(),
    ),
    ClipDef(
        label="calm_2",
        description="Read the line below CALMLY at normal volume:",
        sentences=["I am recording a test clip for the Earshot project, so please ignore anything strange you hear."],
        expected=_speech_yes(),
    ),
    ClipDef(
        label="calm_3",
        description="Read the line below CALMLY at normal volume:",
        sentences=["Could you open the window a little, the room feels a bit warm this afternoon."],
        expected=_speech_yes(),
    ),

    # ---- loud speech ----
    ClipDef(
        label="loud_1",
        description="Read the line below LOUDLY, like you're genuinely annoyed:",
        sentences=["I have been waiting in this queue for forty-five minutes and nobody has explained anything!"],
        expected=_speech_yes({"angry": "Yes"}),
    ),
    ClipDef(
        label="loud_2",
        description="Read the line below LOUDLY, like you're genuinely annoyed:",
        sentences=["That truck has been idling outside my window for the past three hours and nobody will do a thing about it!"],
        expected=_speech_yes({"angry": "Yes"}),
    ),

    # ---- two voices ----
    ClipDef(
        label="two_voices_1",
        description=(
            "A 'Karen' voice will read a sentence through the speakers. "
            "Read the line below at the same time (a DIFFERENT sentence). "
            "Both voices on top of each other."
        ),
        sentences=["I am reading one sentence out loud while another voice plays from the speakers at the same time."],
        auto_play=[AutoPlay(kind="say", text=SAY_FIRST_VOICE, voice=VOICE_FEMALE_AU)],
        expected=_speech_yes({"multiple_speakers": "Yes"}),
    ),
    ClipDef(
        label="two_voices_2",
        description=(
            "Two 'say' voices take turns: first Karen (Australian), then Daniel (UK). "
            "You STAY SILENT — just let the speakers play."
        ),
        auto_play=[
            AutoPlay(kind="say", text=SAY_FIRST_VOICE, voice=VOICE_FEMALE_AU),
            AutoPlay(kind="say", text=SAY_SECOND_VOICE, voice=VOICE_MALE_UK),
        ],
        expected=_speech_yes({"multiple_speakers": "Yes"}),
        notes="Two pre-recorded voices, no human speaker.",
    ),

    # ---- music ----
    ClipDef(
        label="music_1",
        description="Play music from your phone or laptop. No talking. Just music.",
        expected={**_speech_yes(),
                  "is_speaking": "No", "silent": "No", "main_sound": "music"},
    ),
    ClipDef(
        label="music_2",
        description="Same — second music clip.",
        expected={**_speech_yes(),
                  "is_speaking": "No", "silent": "No", "main_sound": "music"},
    ),
    ClipDef(
        label="music_speech_1",
        description="Music playing from your phone or laptop. Talk OVER it (read the line below).",
        sentences=["I am recording while music is playing, so please forgive the slightly cluttered audio."],
        expected=_speech_yes({"music": "Yes"}),
    ),
    ClipDef(
        label="music_speech_2",
        description="Same setup, different sentence.",
        sentences=["The weather has been miserable lately and I am looking forward to some sunshine."],
        expected=_speech_yes({"music": "Yes"}),
    ),

    # ---- non-speech sounds ----
    ClipDef(
        label="clap_1",
        description="Clap 3-5 times during the recording.",
        expected=_noise({"clapping": "Yes"}),
    ),
    ClipDef(
        label="typing_1",
        description="Type naturally on your keyboard for the full window. Mix of keystrokes and pauses.",
        expected=_noise(),
        notes="Some mics don't pick up typing well — redo if rms < 0.005.",
    ),
    ClipDef(
        label="knock_1",
        description="Knock on the desk a few times during the recording.",
        expected=_noise(),
    ),

    # ---- stop word ----
    ClipDef(
        label="stop_1",
        description="Read the line below naturally (it contains the word 'stop'):",
        sentences=["Please stop making that noise, it has been going on all morning."],
        expected=_speech_yes({"said_stop": "Yes"}),
    ),
    ClipDef(
        label="no_stop_1",
        description="Read the line below naturally (similar meaning, NO word 'stop'):",
        sentences=["Please quiet that noise down, it has been going on all morning."],
        expected=_speech_yes({"said_stop": "No"}),
    ),

    # ---- system alarm ----
    ClipDef(
        label="alarm_1",
        description=(
            "This script will LOOP a system alert sound through the speakers. "
            "You STAY SILENT."
        ),
        auto_play=[AutoPlay(kind="afplay", path=ALARM_SOUND, repeat=True)],
        expected=_noise({"phone_or_alarm": "Yes"}),
        notes="macOS Sosumi sound, looped until capture ends.",
    ),
    ClipDef(
        label="alarm_2",
        description="Same as alarm_1 — second take.",
        auto_play=[AutoPlay(kind="afplay", path=ALARM_SOUND, repeat=True)],
        expected=_noise({"phone_or_alarm": "Yes"}),
    ),
]

# Sanity check
_EXPECTED_KEYS = set(MANIFEST_HEADER[3:])
for c in CLIPS:
    miss = _EXPECTED_KEYS - set(c.expected.keys())
    if miss:
        raise SystemExit(f"clip {c.label} missing expected keys: {miss}")

CLIPS_BY_LABEL: dict[str, ClipDef] = {c.label: c for c in CLIPS}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _countdown(seconds_left: int, label: str) -> None:
    pad = lambda s: f"{s:<22s}"  # noqa: E731
    sys.stdout.write(
        f"\r{pad(label)} ⏺ recording in {seconds_left}s...   "
    )
    sys.stdout.flush()


def _record(seconds: float, autoplay: list[AutoPlay], debug: bool
            ) -> tuple[np.ndarray, AutoPlayer | None, dict]:
    """Record `seconds` from the default mic. Returns:
        (audio, autoplay_obj_or_None, debug_meta)
    """
    sr = SR
    target_frames = int(round(seconds * sr))
    cap_seconds = seconds + 0.5  # slop so the first N samples are clean

    debug_meta: dict = {}
    if debug:
        device = sd.query_devices(kind="input")
        debug_meta["input_device_index"] = device.get("index")
        debug_meta["input_device_name"] = device.get("name", "?")
        debug_meta["input_sr"] = int(device.get("default_samplerate", 0))
        debug_meta["sample_rate"] = sr
        debug_meta["target_frames"] = target_frames
        sys.stdout.write(f"  [debug] input device: [{device.get('index')}] "
                         f"{device.get('name', '?')}\n")
        sys.stdout.write(f"  [debug] device default sr: "
                         f"{int(device.get('default_samplerate', 0))} Hz\n")
        sys.stdout.write(f"  [debug] capture sr: {sr} Hz  target frames: {target_frames}\n")

    player = AutoPlayer(autoplay, cap_seconds, debug=debug)
    cap = AudioCapture(sr=sr, window_seconds=cap_seconds)
    try:
        t0_wall = time.time()
        t0 = time.monotonic()
        if debug:
            sys.stdout.write(f"  [debug] recording start wall-clock: {t0_wall:.3f}\n")
        cap.start()
        deadline = time.monotonic() + cap_seconds + 2.0
        while cap.frames_captured() < target_frames and time.monotonic() < deadline:
            time.sleep(0.04)
        time.sleep(0.08)
        window = cap.get_window()
        t1 = time.monotonic()
        t1_wall = time.time()
    finally:
        cap.stop()
        player.stop()

    debug_meta["start_wall"] = t0_wall
    debug_meta["end_wall"] = t1_wall
    debug_meta["frames_captured"] = int(window.shape[0])
    debug_meta["autoplay"] = player.records()

    if window.shape[0] < target_frames:
        pad = np.zeros(target_frames - window.shape[0], dtype=np.float32)
        window = np.concatenate([pad, window])
    elif window.shape[0] > target_frames:
        window = window[:target_frames]
    return window.astype(np.float32, copy=True), (player if debug else None), debug_meta


# ---------------------------------------------------------------------------
# Analysis + manifest upsert
# ---------------------------------------------------------------------------

def _rms_peak(audio: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return rms, peak


def _level_warning(clip: ClipDef, rms: float) -> str | None:
    expected_silent = clip.expected.get("main_sound") == "silence"
    expects_speech = clip.expected.get("is_speaking") == "Yes"

    if expected_silent and rms > 0.015:
        return (f"expected SILENCE but got real signal "
                f"(RMS={rms:.4f}, peak too high). Re-run?")
    if expects_speech and rms < 0.005:
        return (f"expected SPEECH but got almost nothing "
                f"(RMS={rms:.4f}). Speak louder or re-run.")
    if clip.expected.get("clapping") == "Yes" and rms < 0.005:
        return (f"expected clapping but RMS={rms:.4f} is too low. "
                f"Did the claps register?")
    return None


def _completed_labels() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    done: set[str] = set()
    with open(MANIFEST_PATH, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            lab = (row.get("label") or "").strip()
            if lab:
                done.add(lab)
    return done


def _upsert_manifest(clip: ClipDef) -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("label") or "") != clip.label:
                    rows.append(row)
    new_row = {"file": f"{clip.label}.wav", "label": clip.label,
               "description": clip.description}
    for col in MANIFEST_HEADER[3:]:
        new_row[col] = clip.expected.get(col, "?")
    rows.append(new_row)
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# --list mode
# ---------------------------------------------------------------------------

def _do_list() -> int:
    done = _completed_labels()
    sys.stdout.write("\nclips/\n\n")
    sys.stdout.write(f"  {'rec':<4s}  {'#':<3s}  {'label':<18s}  description / sentence\n")
    sys.stdout.write("  " + "-" * 76 + "\n")
    for i, c in enumerate(CLIPS, 1):
        mark = "[✓]" if c.label in done else "[ ]"
        line_one = f"  {mark:<4s}  {i:<3d}  {c.label:<18s}  {c.description}"
        sys.stdout.write(line_one + "\n")
        for s in c.sentences:
            sys.stdout.write(f"           {'':>18s}  \"{s}\"\n")
        if c.auto_play:
            kinds = ",".join(sorted({s.kind for s in c.auto_play}))
            sys.stdout.write(f"           {'':>18s}  [auto: {kinds}]\n")
    sys.stdout.write("\n")
    sys.stdout.write(f"  legend: [✓] = recorded ({len(done)})   [ ] = pending ({len(CLIPS) - len(done)})\n")
    sys.stdout.write(f"  manifest: {MANIFEST_PATH}\n")
    return 0


# ---------------------------------------------------------------------------
# One-clip driver
# ---------------------------------------------------------------------------

def _run_one_clip(clip: ClipDef, *, seconds: float,
                  no_playback: bool, debug: bool) -> int:
    sys.stdout.write("\n" + "=" * 72 + "\n")
    sys.stdout.write(f"  CLIP: {clip.label}\n")
    sys.stdout.write(f"  {clip.description}\n")
    if clip.sentences:
        for s in clip.sentences:
            sys.stdout.write(f"    > {s}\n")
    if clip.notes:
        sys.stdout.write(f"  ({clip.notes})\n")
    sys.stdout.write("\n  press Enter when you're ready (or Ctrl-C to abort)...")
    sys.stdout.flush()
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n  aborted (no input).\n")
        return 130

    # Countdown
    for n in range(COUNTDOWN_SECONDS, 0, -1):
        _countdown(n, clip.label)
        time.sleep(1.0)
    sys.stdout.write(f"\r{clip.label:<22s} ⏺ RECORDING                    \n")
    sys.stdout.flush()

    # Pre-flight mic check
    try:
        sd.query_devices(kind="input")
    except Exception as e:
        sys.stdout.write(f"  ! no usable input device: {type(e).__name__}: {e}\n")
        sys.stdout.write(mic_permission_instructions() + "\n")
        return 1

    # Record
    try:
        audio, _player_dbg, dbg = _record(seconds, clip.auto_play, debug)
    except Exception as e:
        sys.stdout.write(f"  ! record failed: {type(e).__name__}: {e}\n")
        return 1

    # Debug summary first (per spec: rms / peak / samples / autoplay)
    samples_actual = dbg.get("frames_captured", audio.shape[0])
    expected_samples = dbg.get("target_frames", int(seconds * SR))
    if debug:
        sys.stdout.write(f"  [debug] recording end wall-clock: {dbg.get('end_wall', 0):.3f}\n")
        sys.stdout.write(f"  [debug] captured frames: {samples_actual} / expected: {expected_samples}\n")
        for r in dbg.get("autoplay", []):
            sys.stdout.write(f"  [debug] {r['kind']:<6s} pid={r['pid']} "
                             f"started @ {r['start_wall']:.3f}  "
                             f"{'finished naturally' if r['finished'] else 'STILL RUNNING when killed'}\n")

    rms, peak = _rms_peak(audio)
    sys.stdout.write(f"  ↳ captured {seconds:.1f}s  rms={rms:.4f}  peak={peak:.4f}\n")

    warn = _level_warning(clip, rms)
    if warn:
        sys.stdout.write(f"  ! WARNING — {warn}\n")

    # Save to temporary path; rename on keep.
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CLIPS_DIR / f".{clip.label}.tmp.wav"
    sf.write(str(tmp_path), audio, SR, subtype="FLOAT")
    file_size = tmp_path.stat().st_size
    sys.stdout.write(f"  wrote {tmp_path}  ({file_size:,} bytes)\n")

    # Playback
    if not no_playback:
        sys.stdout.write("  ♪ playback… ")
        sys.stdout.flush()
        try:
            sd.play(audio, SR)
            sd.wait()
        except Exception as e:
            sys.stdout.write(f"\n  ! playback failed: {type(e).__name__}: {e}\n")
        else:
            sys.stdout.write("done.\n")
    else:
        sys.stdout.write("  ♪ playback skipped (--no-playback).\n")

    # Keep / redo prompt
    sys.stdout.write("\n  keep / redo ?  [k] ")
    sys.stdout.flush()
    try:
        ans = input().strip().lower()[:1]
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n  aborted (no input) — tmp file kept at {}\n".format(tmp_path))
        return 130
    if not ans:
        ans = "k"

    final_path = CLIPS_DIR / f"{clip.label}.wav"

    if ans == "r":
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        sys.stdout.write(f"  redo — file removed. Re-run: uv run scripts/record_session.py --clip {clip.label}\n")
        return 0

    if ans != "k":
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        sys.stdout.write(f"  unknown answer '{ans}' — discarded.\n")
        return 0

    # Keep: rename, upsert, exit
    tmp_path.replace(final_path)
    _upsert_manifest(clip)
    sys.stdout.write(f"  saved {final_path.name}\n")
    sys.stdout.write(f"  manifest row upserted ({MANIFEST_PATH})\n")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--clip", default=None,
                        help="record just this one clip (e.g. 'calm_1')")
    parser.add_argument("--list", action="store_true",
                        help="list all clips with their instructions and recorded status")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help=f"clip length in seconds (default: {DEFAULT_SECONDS})")
    parser.add_argument("--no-playback", action="store_true",
                        help="don't play the recorded clip back")
    parser.add_argument("--debug", action="store_true",
                        help="print device / timing / autoplay debug info")
    args = parser.parse_args(argv)

    if args.list:
        return _do_list()

    if not args.clip:
        sys.stdout.write("error: --clip <label> is required (use --list to see options)\n")
        return 2

    target = CLIPS_BY_LABEL.get(args.clip)
    if target is None:
        sys.stdout.write(f"error: unknown clip '{args.clip}'. Use --list.\n")
        return 2

    return _run_one_clip(
        target,
        seconds=args.seconds,
        no_playback=args.no_playback,
        debug=args.debug,
    )


if __name__ == "__main__":
    raise SystemExit(main())
