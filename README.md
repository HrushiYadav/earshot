# Earshot

> Local voice decision engine. Listen to the last ~3 seconds of mic audio; answer 8–16 yes/no or multiple-choice questions about it every 0.5–1 second, as typed probabilities. Fully offline on Apple Silicon.

## Status

Stage 0 in progress. See [AGENTS.md](./AGENTS.md) for the full stage plan and selected model.

## Quickstart (after Stage 1+)

```bash
uv sync
uv run earshot-live --questions questions.yaml --hop 0.5
```

## How it works (one paragraph)

Audio LMs are causal. We encode a 3 s audio window once as a language-model prefix (system + audio + everything before the question text), broadcast its KV cache across a batch of question suffixes in a single forward pass, and read P(Yes) directly from the `Yes`/`No` logits at each row's last non-pad token. No text generation. Adding more questions costs almost nothing per pass — that's the whole point of the Stage 6 benchmark.

## Credits

Inspired by the [Vertix webcam demo](https://github.com/fofr/vertix) and TypeSafe's Jev.
