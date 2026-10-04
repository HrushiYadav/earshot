# results/

Curated, committed reports and figures for earshot. Raw artefacts (clip
recordings, model weights, per-pass logs) stay out of the repo — see
`clips/` and `bench/` (both gitignored).

## Files

| File | What it is |
| --- | --- |
| `bench.md` / `bench.csv` | Earshot latency benchmark (sequential vs batched at N ∈ [1, 4, 8, 16, 32]). |
| `bench_stage3.md` / `bench_stage3.csv` | Earlier quick bench at N ∈ [1, 4, 8] on a single clip — superseded by `bench.md`. |
| `eval_clips.md` / `eval_clips.csv` | Per-question accuracy on the 21 labelled clips in `clips/manifest.csv`. |
| `latency.png` | Chart of `bench.md` (batched vs sequential median and p90). |
| `v02_comparison.md` / `v02_comparison.png` | Comparison of earshot vs CLAP vs a trained classifier on ESC-50 (50-way) and three speech/tone tasks. |

## Demo

- `demo.gif` — short slice from a live recording, embedded in the top-level
  `README.md`. Lives here so the top-level README stays light.

## Generating them

```bash
uv run python scripts/bench.py            # bench.md + bench.csv + latency.png
uv run python scripts/eval_clips.py       # eval_clips.md + eval_clips.csv
uv run python scripts/v02/v02_earshot_save_all.py    # writes the .npz under bench/
uv run python scripts/v02/v02_clap_partc_aurocprompts.py
uv run python scripts/v02/v02_analyze.py  # consumes the .npz, prints results
uv run python scripts/v02/make_v02_chart.py          # v02_comparison.png
```