# results/

Curated, committed reports for earshot. Raw bench output and clip
artefacts stay out of the repo (see `bench/` which is gitignored and
`clips/` which is gitignored).

## Files in this directory

- `placeholder.md` — this file.
- Future: `accuracy.csv` (per-question accuracy across the manifest),
  `latency.png` (Stage 3 batched vs sequential line chart, N on x,
  ms on y), `calibration.md` (Stage 7 ECE before/after + reliability
  plot).

## TODO

- [ ] Render the Stage 3 batched-vs-sequential latency chart.
- [ ] Render per-question accuracy from `scripts/eval_clips.py`.
- [ ] After Stage 7: per-question ECE, reliability plot, raw labels.csv.
