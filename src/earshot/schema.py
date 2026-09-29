"""Stage 4 — typed schema for questions and results.

The production question set lives in `questions.yaml` (12 bool + 2 choice
in Stage 4). This module defines the pydantic models, the YAML loader,
and the typed result shapes that the rest of the codebase consumes.

Why pydantic instead of plain dataclasses: the loader is the *first*
defensive line. A typo in a YAML field, a missing required attribute,
or a bad type should fail loudly at startup with a clear field path —
not silently produce a missing question at runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Question models — discriminated union on `type`
# ---------------------------------------------------------------------------

class BaseQuestion(BaseModel):
    """Shared fields for both bool and choice questions."""
    id: str = Field(min_length=1, description="short stable identifier")
    text: str = Field(min_length=1, description="natural-language question text")
    manifest_col: str | None = Field(
        default=None,
        description="column in clips/manifest.csv used by eval_clips.py",
    )
    smoothing: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="EMA alpha used by the live loop (Stage 5); 0.0 = frozen, 1.0 = no smoothing",
    )


class BoolQuestion(BaseQuestion):
    """A yes/no question. P(Yes) is read at the last non-pad position.

    Two thresholds implement hysteresis: above `enter` the state becomes
    "yes"; below `exit` it becomes "no"; in between it stays as it was.
    Both default to 0.5/0.4 (a 0.1 deadband). `eval_clips.py` uses just
    `enter` for the simple P(Yes) >= enter decision; the live loop
    (Stage 5) uses both.
    """
    type: Literal["bool"] = "bool"
    enter: float = Field(default=0.5, ge=0.0, le=1.0)
    exit: float = Field(default=0.4, ge=0.0, le=1.0)
    source: Literal["model", "signal"] = "model"

    # signal-source questions need a signal name + threshold. Currently the
    # only supported signal is "rms" (root-mean-square amplitude of the audio
    # window). The scorer dispatches on (source, signal) and falls back to
    # raising on unknown combinations.
    signal: Literal["rms"] | None = None
    signal_threshold: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _check_signal_fields(self) -> "BoolQuestion":
        if self.exit > self.enter:
            raise ValueError(
                f"bool question {self.id!r}: exit ({self.exit}) must be <= enter ({self.enter})"
            )
        if self.source == "signal":
            if self.signal is None:
                raise ValueError(
                    f"bool question {self.id!r} has source=signal but no `signal` field"
                )
            if self.signal_threshold is None:
                raise ValueError(
                    f"bool question {self.id!r} has source=signal but no `signal_threshold`"
                )
        else:
            if self.signal is not None or self.signal_threshold is not None:
                raise ValueError(
                    f"bool question {self.id!r} has source=model but carries signal fields "
                    f"(signal={self.signal!r}, signal_threshold={self.signal_threshold!r})"
                )
        return self


class ChoiceQuestion(BaseQuestion):
    """A multiple-choice question. Options are lettered A, B, C… in order;
    P(letter) is read at the last position and softmaxed across the options."""
    type: Literal["choice"] = "choice"
    options: list[str] = Field(min_length=2, max_length=26)
    source: Literal["model"] = "model"

    @model_validator(mode="after")
    def _check_options_unique(self) -> "ChoiceQuestion":
        if len(set(self.options)) != len(self.options):
            raise ValueError(
                f"choice question {self.id!r} has duplicate options: {self.options}"
            )
        return self


Question = Annotated[
    BoolQuestion | ChoiceQuestion,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Typed results — the per-question output shape
# ---------------------------------------------------------------------------

class BoolResult(BaseModel):
    """Typed output for a bool question. `p` is P(Yes); `fired` is whether
    p crossed the question's `enter` threshold."""
    p: float = Field(ge=0.0, le=1.0)
    fired: bool


class ChoiceResult(BaseModel):
    """Typed output for a choice question. `probs` is a dict option→probability
    (sums to 1.0 within float epsilon); `top` is the highest-probability option."""
    probs: dict[str, float]
    top: str

    @model_validator(mode="after")
    def _check_top_is_in_probs(self) -> "ChoiceResult":
        if self.top not in self.probs:
            raise ValueError(
                f"ChoiceResult.top={self.top!r} is not in probs keys {list(self.probs)}"
            )
        for opt, p in self.probs.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"ChoiceResult.probs[{opt!r}]={p} is not in [0, 1]"
                )
        return self


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

class QuestionsFile(BaseModel):
    """Top-level shape of questions.yaml."""
    questions: list[Question]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "QuestionsFile":
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            seen = set()
            dups = []
            for qid in ids:
                if qid in seen:
                    dups.append(qid)
                seen.add(qid)
            raise ValueError(f"duplicate question ids in YAML: {dups}")
        return self


def load_questions(path: str | Path) -> list[Question]:
    """Load `questions.yaml` from disk and return the typed list.

    Raises pydantic.ValidationError on any schema problem (missing field,
    wrong type, duplicate id, signal source without a threshold, etc.).
    The ValidationError already includes the field path; the original
    exception's traceback carries the source line, so we propagate it
    as-is. (We previously tried to re-raise with the file path prepended,
    but ValidationError's constructor in pydantic 2 takes a non-trivial
    argument shape that doesn't compose with a plain message.)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"questions file not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "questions" not in raw:
        raise ValueError(
            f"{p}: expected top-level key `questions` (list), got {type(raw).__name__}"
        )
    parsed = QuestionsFile.model_validate(raw)
    return parsed.questions


# ---------------------------------------------------------------------------
# Lettering helper — used by both scorer (to build the suffix) and by tests
# (to verify that letter tokens are single ids in the tokenizer).
# ---------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter_for_index(i: int) -> str:
    """Return 'A' for 0, 'B' for 1, …, 'Z' for 25."""
    if not 0 <= i < len(LETTERS):
        raise IndexError(f"letter_for_index({i}): max 26 options supported")
    return LETTERS[i]
