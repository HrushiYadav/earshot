"""Stage 4 — tests for `earshot.schema` and the YAML loader.

Goals:
  1. The production `questions.yaml` loads cleanly and yields the expected
     question mix (12 bool + 2 choice).
  2. A deliberately-broken YAML file fails loudly with a pydantic
     ValidationError that mentions the offending field — so a typo at
     3 AM doesn't silently produce a missing question at runtime.

We write the bad-YAML fixtures to a temp dir, never to the repo.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from earshot.schema import (  # noqa: E402
    BoolQuestion,
    ChoiceQuestion,
    load_questions,
)


def _write_yaml(text: str) -> Path:
    """Write `text` to a temp .yaml file and return its path. The file is
    deleted by the caller (or by tmpdir cleanup) — never commit these."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=tempfile.gettempdir(),
    ) as f:
        f.write(text)
        return Path(f.name)


def test_production_yaml_loads() -> None:
    """The repo's questions.yaml must load and yield the planned mix.

    The approved question set (after the Stage 4 tuning pass):
      10 bool questions (9 model-driven + 1 signal-derived `silent`)
      1 choice question (`main_sound` with options speech/music/noise/silence)
    """
    path = REPO_ROOT / "questions.yaml"
    assert path.exists(), f"production YAML missing: {path}"
    qs = load_questions(path)
    bool_qs = [q for q in qs if isinstance(q, BoolQuestion)]
    choice_qs = [q for q in qs if isinstance(q, ChoiceQuestion)]
    assert len(bool_qs) == 10, f"expected 10 bool, got {len(bool_qs)}"
    assert len(choice_qs) == 1, f"expected 1 choice, got {len(choice_qs)}"
    # ids must be unique
    assert len({q.id for q in qs}) == len(qs)
    # the signal silent entry must carry both signal fields
    silent = next(q for q in qs if q.id == "silent")
    assert silent.source == "signal"
    assert silent.signal == "rms"
    assert silent.signal_threshold is not None and silent.signal_threshold > 0
    # main_sound's options must exactly match the manifest ground-truth vocabulary
    main_sound = next(q for q in qs if q.id == "main_sound")
    assert main_sound.options == ["speech", "music", "noise", "silence"]


def test_missing_top_level_questions_key() -> None:
    """A YAML that doesn't have a top-level `questions` key should raise."""
    path = _write_yaml("foo: bar\n")
    try:
        try:
            load_questions(path)
        except ValueError as e:
            assert "questions" in str(e), f"expected 'questions' in error, got: {e}"
            return
        raise AssertionError("expected ValueError for missing 'questions' key")
    finally:
        path.unlink(missing_ok=True)


def test_unknown_type_is_rejected() -> None:
    """A question with an unknown `type` value is invalid."""
    path = _write_yaml("""
questions:
  - id: foo
    type: rating   # not bool or choice
    text: "How good is the audio?"
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            # pydantic puts the bad field in the error string
            assert "type" in str(e).lower(), f"expected 'type' in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for unknown type")
    finally:
        path.unlink(missing_ok=True)


def test_choice_question_requires_options() -> None:
    """A choice question without `options` must fail."""
    path = _write_yaml("""
questions:
  - id: foo
    type: choice
    text: "Pick one"
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            assert "options" in str(e).lower(), f"expected 'options' in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for missing options")
    finally:
        path.unlink(missing_ok=True)


def test_signal_question_without_signal_field_fails() -> None:
    """A bool with source=signal but no `signal` field must fail."""
    path = _write_yaml("""
questions:
  - id: silent
    type: bool
    text: "Is the room silent?"
    source: signal
    threshold: 0.5
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            # Either "signal" or "signal_threshold" will be in the message.
            msg = str(e).lower()
            assert "signal" in msg, f"expected 'signal' in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for signal without signal field")
    finally:
        path.unlink(missing_ok=True)


def test_duplicate_ids_rejected() -> None:
    """Two entries with the same id must fail loudly."""
    path = _write_yaml("""
questions:
  - id: foo
    type: bool
    text: "First"
  - id: foo
    type: bool
    text: "Second (dup id)"
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            assert "foo" in str(e) or "duplicate" in str(e).lower(), \
                f"expected duplicate id in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for duplicate ids")
    finally:
        path.unlink(missing_ok=True)


def test_bool_exit_must_not_exceed_enter() -> None:
    """A bool with exit > enter has no hysteresis and should fail loudly."""
    path = _write_yaml("""
questions:
  - id: foo
    type: bool
    text: "?"
    enter: 0.3
    exit: 0.7
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            msg = str(e).lower()
            assert "exit" in msg and "enter" in msg, f"expected exit/enter in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for exit > enter")
    finally:
        path.unlink(missing_ok=True)


def test_defaults_match_documented_values() -> None:
    """Bool defaults: enter 0.5, exit 0.4, smoothing 0.5; choice smoothing 0.5."""
    path = _write_yaml("""
questions:
  - id: a
    type: bool
    text: "?"
  - id: b
    type: choice
    text: "?"
    options: [x, y]
""")
    try:
        qs = load_questions(path)
        a, b = qs
        assert a.enter == 0.5, f"default enter should be 0.5, got {a.enter}"
        assert a.exit == 0.4, f"default exit should be 0.4, got {a.exit}"
        assert a.smoothing == 0.5, f"default smoothing should be 0.5, got {a.smoothing}"
        assert b.smoothing == 0.5, f"default smoothing should be 0.5, got {b.smoothing}"
    finally:
        path.unlink(missing_ok=True)


def test_choice_options_unique() -> None:
    """Choice options must be unique."""
    path = _write_yaml("""
questions:
  - id: speaker_activity
    type: choice
    text: "Pick one"
    options: [talking, talking, silent]
""")
    try:
        try:
            load_questions(path)
        except ValidationError as e:
            assert "duplicate" in str(e).lower() or "options" in str(e).lower(), \
                f"expected duplicate options in error, got: {e}"
            return
        raise AssertionError("expected ValidationError for duplicate options")
    finally:
        path.unlink(missing_ok=True)


def test_missing_file_is_clear() -> None:
    """Loading a non-existent file should raise FileNotFoundError, not a
    generic exception."""
    bogus = Path(tempfile.gettempdir()) / "definitely_does_not_exist_xyz.yaml"
    if bogus.exists():
        bogus.unlink()
    try:
        load_questions(bogus)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing file")


def main() -> int:
    tests = [
        test_production_yaml_loads,
        test_missing_top_level_questions_key,
        test_unknown_type_is_rejected,
        test_choice_question_requires_options,
        test_signal_question_without_signal_field_fails,
        test_duplicate_ids_rejected,
        test_choice_options_unique,
        test_missing_file_is_clear,
        test_bool_exit_must_not_exceed_enter,
        test_defaults_match_documented_values,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"OK: all {len(tests)} tests passed")
        return 0
    print(f"FAILED: {failed}/{len(tests)} tests failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())