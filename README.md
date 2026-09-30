# earshot

A shared-prefix audio decision engine for live microphone input on Apple Silicon.

![Earshot live demo — 7 s slice from a real recording](results/demo.gif)

The last 3 seconds of microphone audio is encoded once into a prefix state; many yes/no and multiple-choice questions are then answered in one batch that forks from it, with no text generation. Two tiers: signal checks computed straight from the audio (silence), and Qwen2.5-Omni-3B for everything else, at about TODO(Hz) on an Apple M4 MacBook Air (16 GB).

Read the write-up at **[hrushiyadav.com/blog/earshot](https://hrushiyadav.com/blog/earshot)** — every measurement, including the bug where the model silently ignored the audio.

## Why not just ask an audio model?

| | **Normal: ask an audio model** | **earshot** |
|---|---|---|
| Audio processing | re-encodes the same 3 s window once per question | encodes once, reuses for all questions |
| Time for 8 questions | 7.75 s | 1.64 s |
| Output | free-form text you have to parse | typed probabilities, never a string to parse |
| Format errors | yes — model can answer in any shape | none — output is `{p, fired}` or `{probs, top}` |
| Confidence | you have to guess from the text | directly calibrated, P(Yes) ∈ [0, 1] |
| Where it runs | cloud or a beefy GPU | fanless MacBook Air, fully offline |

(Times measured on `clips/calm_1.wav`, 5 s clip, 8 model questions, after 3 warm-up passes. Stage 3 bench — see [Verify](#verify) below.)

## Setup

```bash
# Python 3.11, Apple Silicon recommended
uv sync
```

The model weights download on first run (`Qwen/Qwen2.5-Omni-3B`, ~6 GB). Subsequent runs use the local cache.

## Run

```bash
# Stage 5 — live microphone loop (coming)
# uv run python -m earshot.live --questions questions.yaml --hop 0.5

# Stage 5 — replay a clip, no live UI (coming)
# uv run python -m earshot.live --source clips/calm_1.wav --no-view --log run.jsonl

# Stage 5 — ad-hoc extra question on top of the YAML set (coming)
# uv run python -m earshot.live --ask "Is someone waving?"

# Current: evaluate the YAML questions against the manifest (Stage 4)
uv run python scripts/eval_clips.py
```

## Verify

The equivalence test (Stage 3) checks that the batched fork produces the
same answers as the sequential one-question-per-forward baseline, within
0.02 in P(Yes). It also covers choice questions (Stage 4) via L1
distance across the probs dict.

```bash
# Normal run — must PASS
uv run python tests/test_equivalence.py

# Control mode — deliberately offsets suffix positions by +1; must FAIL
# (exit 0 only if at least one cell breaks, proving the test is sensitive)
uv run python tests/test_equivalence.py --wrong-pos

# Schema + bad-YAML checks (Stage 4) — all 8 must PASS
uv run python tests/test_schema.py

# Per-clip evaluation against the manifest (Stage 4)
uv run python scripts/eval_clips.py

# Stage 3 quick bench — sequential vs batched on one clip (N in 1, 4, 8)
uv run python scripts/dev/bench_batch_vs_seq.py
```

Acceptance so far:

| Check | Result |
|---|---|
| `tests/test_equivalence.py` (21 clips × 13 model questions, tolerance 0.02) | max diff 0.000000; speedup ×4.84 |
| `tests/test_equivalence.py --wrong-pos` | breaks equivalence on TODO(most) cells — control is working |
| `tests/test_schema.py` (8 tests) | TODO(pass count) |
| `scripts/eval_clips.py` overall accuracy at threshold 0.5 | TODO |
| Stage 3 bench (calm_1, after 3 warm-ups + 5 timed passes) | N=1: 0.755 s → 0.912 s; N=4: 3.101 s → 1.099 s; N=8: 7.751 s → 1.637 s |

## Layout

| Path | Role |
|---|---|
| `README.md` | this file |
| `AGENTS.md` | per-stage notes, model recipe, measured numbers (gitignored) |
| `questions.yaml` | production question set — 12 bool + 2 choice, loaded via pydantic schema |
| `pyproject.toml` | project + dependencies |
| `src/earshot/audio.py` | mic capture + ring buffer |
| `src/earshot/model.py` | load Qwen2.5-Omni-3B (thinker-only) on MPS |
| `src/earshot/scorer.py` | sequential + batched fork-and-score; bool P(Yes) + choice letter-token softmax |
| `src/earshot/schema.py` | pydantic models for `BoolQuestion`, `ChoiceQuestion`, results, YAML loader |
| `src/earshot/prompts.py` | system message, question template, signal thresholds; re-exports YAML loader |
| `scripts/eval_clips.py` | per-clip eval against `clips/manifest.csv` |
| `scripts/spike_models.py` | Stage 0 model-load probe |
| `scripts/record_clips.py`, `scripts/record_session.py` | Stage 1 — labelled clip recording |
| `scripts/dev/bench_batch_vs_seq.py` | Stage 3 quick timing |
| `tests/test_equivalence.py` | Stage 3 — sequential vs batched |
| `tests/test_schema.py` | Stage 4 — YAML schema + bad-YAML checks |
| `clips/` | user recordings (gitignored) |
| `private/` | plan, notes, references, debug scripts (gitignored) |
| `results/` | small committed reports — markdown, csv, png only |

## Results

TODO(table) — per-question accuracy, no-decode vs decode, batched
latency by N, ECE after Stage 7 calibration. See `results/` for the
charts.

## Limitations

- **Singing is counted as speech.** `is_speaking` P(Yes) is high on sung
  lyrics; we don't separate the two.
- **`angry` reacts to word meaning, not vocal stress.** The model uses
  lexical cues ("stop") as much as acoustic ones (loudness, pitch);
  Stage 7 calibration is needed to make it useful.
- **~1 Hz on a fanless Air.** With 9 model questions (Stage 4 production
  set) and the batched path, throughput is roughly one decision per
  second. Above that the model latency dominates; below that the audio
  ring buffer is the bottleneck.
- **Qwen2.5-Omni-3B is the only supported model.** The fork trick depends
  on the prefix/suffix split that the chat template renders around the
  `<|AUDIO|>` placeholder. A different audio LM would need its own
  prefix split and token-id groups for the Yes/No read-out.

## Credits

Architectural reference: [drxddy/vertix](https://github.com/drxddy/vertix) —
shared-prefix VLM decision engine on Apple Silicon (tiers, --wrong-pos
control, MLX runtime, early-exit per-layer states). Read-only
inspiration; no code copied.

Jev-style typed tool calls: [TypeSafe — Introducing System One Models
and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
