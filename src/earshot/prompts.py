"""Prompts — system message, question template, and the YAML question loader.

Per the AGENTS.md rule that prompt text should live in one place, this
module owns the system message, the bool question template, and the
signal threshold for `silent`. The actual question definitions live in
`questions.yaml` at the repo root; we load and re-export them from here
so the rest of the codebase has a single import surface for prompts.

Stage 2 history: this module used to define STARTER_BOOL_QUESTIONS as a
list of dicts. Stage 4 replaced those dicts with pydantic models
(`earshot.schema.BoolQuestion` / `ChoiceQuestion`) loaded from YAML.
"""
from __future__ import annotations

from pathlib import Path

from .schema import (
    BoolQuestion,
    ChoiceQuestion,
    Question,
    load_questions,
)


# Default system message sent before every question. Phrased so the model
# treats the audio as the input and the question as the prompt — and gives a
# crisp yes/no answer, which we read P(Yes) from at the last position.
SYSTEM_MESSAGE = (
    "You are an audio listener. Answer the user's yes/no question about the "
    "audio with a single word: Yes or No."
)


# Format: a fragment appended to whatever the user wants to ask, so the model
# is encouraged to answer binary rather than ramble. Used for bool questions.
QUESTION_TEMPLATE = "{question} Answer Yes or No."


# Threshold (RMS over the audio window) below which we consider the room
# silent. Picked from clips/manifest data:
#   silence_1    rms=0.001601   expected silent
#   silence_2    rms=0.004848   expected silent
#   room_noise_1 rms=0.004966   expected silent
#   alarm_1      rms=0.005111   expected non-silent
#   ... everything else is well above 0.01.
# Midpoint between the loudest "silent" and the quietest "non-silent" is
# 0.005039; we round up to 0.00505 to keep the gap on the alarm side.
# This is also written into questions.yaml's `silent` entry as
# `signal_threshold`, so the YAML and the constant agree. The YAML value
# wins at runtime; the constant here is the historical source and a sane
# fallback if the YAML is unavailable.
SIGNAL_SILENT_RMS_THRESHOLD = 0.00505


# Default YAML path (lives at the repo root, next to pyproject.toml).
DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "questions.yaml"


def load_default_questions() -> list[Question]:
    """Load the production question set from `questions.yaml`."""
    return load_questions(DEFAULT_QUESTIONS_PATH)


def question_text(question_id: str, questions: list[Question] | None = None) -> str:
    """Look up the natural-language text for a question by id.

    Raises KeyError if the id isn't in the loaded question set — that's a
    programmer error, not a runtime one, and should surface loudly.
    """
    qs = questions if questions is not None else load_default_questions()
    for q in qs:
        if q.id == question_id:
            return q.text
    raise KeyError(f"unknown question id: {question_id!r}")


__all__ = [
    "SYSTEM_MESSAGE", "QUESTION_TEMPLATE", "SIGNAL_SILENT_RMS_THRESHOLD",
    "BoolQuestion", "ChoiceQuestion", "Question",
    "DEFAULT_QUESTIONS_PATH", "load_default_questions", "load_questions",
    "question_text",
]