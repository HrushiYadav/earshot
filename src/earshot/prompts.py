"""Prompts — single source for system messages and question templates.

Per the AGENTS.md rule that prompt text should live in one place, this module is
the only place that should contain the system message, the question template,
and any per-question definitions. Other modules import from here.
"""
from __future__ import annotations


# Default system message sent before every question. Phrased so the model
# treats the audio as the input and the question as the prompt — and gives a
# crisp yes/no answer, which we read P(Yes) from at the last position.
SYSTEM_MESSAGE = (
    "You are an audio listener. Answer the user's yes/no question about the "
    "audio with a single word: Yes or No."
)


# Format: a fragment appended to whatever the user wants to ask, so the model
# is encouraged to answer binary rather than ramble.
QUESTION_TEMPLATE = "{question} Answer Yes or No."


# The 8 starter bool questions from Stage 2 of the plan. Each row:
#   id           — short stable identifier used as the manifest column name
#                  and the score_sequential return key.
#   text         — the natural-language question shown to the model.
#   manifest_col — the matching column in clips/manifest.csv used by
#                  scripts/eval_clips.py for accuracy.
STARTER_BOOL_QUESTIONS: list[dict] = [
    {
        "id": "is_speaking",
        "text": "Is someone speaking?",
        "manifest_col": "is_speaking",
    },
    {
        "id": "multiple_speakers",
        "text": "Is more than one person speaking?",
        "manifest_col": "multiple_speakers",
    },
    {
        "id": "music",
        "text": "Is music playing?",
        "manifest_col": "music",
    },
    {
        "id": "angry",
        "text": "Does the speaker sound angry or stressed?",
        "manifest_col": "angry",
    },
    {
        "id": "said_stop",
        "text": 'Did someone say the word "stop"?',
        "manifest_col": "said_stop",
    },
    {
        "id": "clapping",
        "text": "Is there clapping?",
        "manifest_col": "clapping",
    },
    {
        "id": "phone_or_alarm",
        "text": "Is a phone or alarm ringing?",
        "manifest_col": "phone_or_alarm",
    },
    {
        "id": "silent",
        "text": "Is the room silent?",
        "manifest_col": "silent",
    },
]


def question_text(question_id: str) -> str:
    """Look up the natural-language text for a question by id.

    Raises KeyError if the id isn't in STARTER_BOOL_QUESTIONS — that's a
    programmer error, not a runtime one, and should surface loudly.
    """
    for q in STARTER_BOOL_QUESTIONS:
        if q["id"] == question_id:
            return q["text"]
    raise KeyError(f"unknown question id: {question_id!r}")
