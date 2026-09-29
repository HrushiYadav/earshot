"""Stage 5 acceptance: the live loop runs end-to-end in file mode and writes
a valid JSONL log.

Boots the model (one ~80 s MPS-kernel-compile warm-up pass), then runs
`python -m earshot.live --source clips/calm_1.wav --no-view --log <tmp>`
for `--duration 10 s` and checks every line is JSON with the expected shape
(timestamp, latency_ms, rms, results keyed by question id, every result
typed as BoolResult or ChoiceResult with the right fields).

Total wall-clock: ~95-100 s on Apple M4 MacBook Air (16 GB).

Invoke directly (no pytest required, matching the rest of tests/):

    python -m tests.test_live_smoke [--clip PATH] [--duration 10] [--timeout 180]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_CMD = [sys.executable, "-m", "earshot.live"]
DEFAULT_CLIP = REPO_ROOT / "clips" / "calm_1.wav"
DEFAULT_DURATION = 10.0
DEFAULT_TIMEOUT = 180.0


def _expected_question_ids() -> set[str]:
    """Mirror `questions.yaml` so this test fails loudly if the schema drifts."""
    # Keep in sync with questions.yaml + the schema frozen in Stage 4.
    return {
        "is_speaking", "multiple_speakers", "music", "angry", "said_stop",
        "clapping", "phone_or_alarm", "typing", "door_knock", "silent",
        "main_sound",
    }


def run(clip: Path, duration: float, timeout: float) -> int:
    """Run the live loop and validate the JSONL log. Returns 0 on pass."""
    if not clip.exists():
        print(f"FAIL: missing test clip {clip}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "live.jsonl"
        cmd = [
            *LIVE_CMD,
            "--source", str(clip),
            "--no-view", "--quiet",
            "--log", str(log_path),
            "--duration", str(duration),
            "--hop", "0.5",
        ]
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, cwd=REPO_ROOT,
            capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"FAIL: earshot.live exited {result.returncode} after {elapsed:.1f}s",
                  file=sys.stderr)
            print(f"  stdout tail: {result.stdout[-1500:]}", file=sys.stderr)
            print(f"  stderr tail: {result.stderr[-1500:]}", file=sys.stderr)
            return 1
        if not log_path.exists():
            print(f"FAIL: log file not created at {log_path}", file=sys.stderr)
            return 1

        raw = log_path.read_text()
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            print(f"FAIL: no JSONL lines written; raw={raw!r}", file=sys.stderr)
            return 1

        expected_ids = _expected_question_ids()
        bool_required = {"p", "p_smoothed", "fired", "source"}
        choice_required = {"probs", "top", "source"}
        top_choices = {"speech", "music", "noise", "silence"}

        n_fired_is_speaking = 0
        n_main_speech = 0
        for i, ln in enumerate(lines):
            rec = json.loads(ln)
            assert "ts" in rec and isinstance(rec["ts"], int), f"line {i}: bad ts"
            assert "latency_ms" in rec and isinstance(rec["latency_ms"], (int, float)), \
                f"line {i}: bad latency_ms"
            assert "rms" in rec and isinstance(rec["rms"], (int, float)), \
                f"line {i}: bad rms"
            assert "results" in rec and isinstance(rec["results"], dict), \
                f"line {i}: bad results"

            result_ids = set(rec["results"].keys())
            if result_ids != expected_ids:
                print(f"FAIL: line {i} question ids mismatch", file=sys.stderr)
                print(f"  expected: {sorted(expected_ids)}", file=sys.stderr)
                print(f"  got:      {sorted(result_ids)}", file=sys.stderr)
                return 1

            for qid, r in rec["results"].items():
                if qid == "main_sound":
                    if not choice_required.issubset(r.keys()):
                        print(f"FAIL: line {i}/{qid} missing {choice_required - set(r.keys())}",
                              file=sys.stderr)
                        return 1
                    if not isinstance(r["probs"], dict) or not r["probs"]:
                        print(f"FAIL: line {i}/{qid} bad probs {r['probs']!r}",
                              file=sys.stderr)
                        return 1
                    if r["top"] not in top_choices:
                        print(f"FAIL: line {i}/{qid} top={r['top']!r} not in {top_choices}",
                              file=sys.stderr)
                        return 1
                    if r["top"] == "speech":
                        n_main_speech += 1
                else:
                    if not bool_required.issubset(r.keys()):
                        print(f"FAIL: line {i}/{qid} missing {bool_required - set(r.keys())}",
                              file=sys.stderr)
                        return 1
                    if not (0.0 <= r["p"] <= 1.0):
                        print(f"FAIL: line {i}/{qid} p={r['p']} out of [0, 1]",
                              file=sys.stderr)
                        return 1
                    if not (0.0 <= r["p_smoothed"] <= 1.0):
                        print(f"FAIL: line {i}/{qid} p_smoothed={r['p_smoothed']} out of [0, 1]",
                              file=sys.stderr)
                        return 1
                    if qid == "is_speaking" and r["fired"]:
                        n_fired_is_speaking += 1

        print(f"PASS: {len(lines)} JSONL lines, all schema-valid ({elapsed:.1f}s total)")
        print(f"  is_speaking fired on {n_fired_is_speaking}/{len(lines)} passes")
        print(f"  main_sound.top=speech on {n_main_speech}/{len(lines)} passes")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP,
                        help="WAV file to loop (default: clips/calm_1.wav)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="seconds to run (default 10)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="subprocess timeout in seconds (default 180)")
    args = parser.parse_args(argv)
    return run(args.clip, args.duration, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())