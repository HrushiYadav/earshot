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


# Production bool questions. Folded in from the Stage 2.5 tuning pass:
#   - `angry` wording was changed to the variant that scored 19/21 in eval
#     (vs the original's 18/21). Variant wording is also more concrete
#     ("shouting or speaking with a raised voice" vs "angry or stressed"),
#     which made the model more selective — that traded 2 FPs on clap/stop
#     for 2 FNs on the actual loud clips. Kept the variant on the user's
#     "keep the higher-scoring text" rule; flagged in eval output.
#   - `clapping` wording kept the original ("Is there clapping?"); v2
#     ("Is someone clapping their hands?") tied 20/21 = 20/21 — no reason
#     to switch when the original is shorter.
#   - `typing` was promoted from variant after scoring 20/21 = 0.952; only
#     miss is silence_1 (probably a start-of-recording click being read as
#     typing).
#
#   id           — short stable identifier used as the manifest column name
#                  and the score_sequential return key.
#   text         — the natural-language question shown to the model.
#   manifest_col — the matching column in clips/manifest.csv used by
#                  scripts/eval_clips.py for accuracy.
#   source       — optional, defaults to "model". Set to "signal" for
#                  questions answered from raw audio (currently just `silent`,
#                  which uses RMS).
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
        # Tuning pass winner — variant scored 19/21 vs original 18/21.
        "text": "Is the speaker shouting or speaking with a raised voice?",
        "manifest_col": "angry",
    },
    {
        "id": "said_stop",
        "text": 'Did someone say the word "stop"?',
        "manifest_col": "said_stop",
    },
    {
        "id": "clapping",
        # Original wording kept; v2 variant tied 20/21.
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
        # Answered from the audio RMS, not the model — see
        # scorer.score_signal_silent. The text above is still useful for
        # typed output (Stage 4 schema) so the user sees the question that
        # was answered.
        "text": "Is the room silent?",
        "manifest_col": "silent",
        "source": "signal",
    },
    {
        # Promoted from variant after scoring 20/21 = 0.952 in tuning pass.
        "id": "typing",
        "text": "Is someone typing on a keyboard?",
        "manifest_col": "typing",
    },
]


# Threshold (RMS over the audio window) below which we consider the room
# silent. Picked from clips/manifest data:
#   silence_1    rms=0.001601   expected silent
#   silence_2    rms=0.004848   expected silent
#   room_noise_1 rms=0.004966   expected silent
#   alarm_1      rms=0.005111   expected non-silent
#   ... everything else is well above 0.01.
# Midpoint between the loudest "silent" and the quietest "non-silent" is
# 0.005039; we round up to 0.00505 to keep the gap on the alarm side.
SIGNAL_SILENT_RMS_THRESHOLD = 0.00505


def question_text(question_id: str) -> str:
    """Look up the natural-language text for a question by id.

    Raises KeyError if the id isn't in STARTER_BOOL_QUESTIONS — that's a
    programmer error, not a runtime one, and should surface loudly.
    """
    for q in STARTER_BOOL_QUESTIONS:
        if q["id"] == question_id:
            return q["text"]
    raise KeyError(f"unknown question id: {question_id!r}")