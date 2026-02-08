#!/usr/bin/env python3
"""Run unit tests without pytest (for environments without pip/pytest)."""

import sys

# Add project root
sys.path.insert(0, ".")


def test_build_messages_basic():
    from llm.context import build_messages

    out = build_messages("You are Jarvis.", "What time is it?")
    assert len(out) == 2
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    assert "What time is it?" in out[1]["content"]


def test_build_messages_with_vision():
    from llm.context import build_messages

    out = build_messages(
        "Sys", "What do you see?", vision_description="person(2)", reminders_text="Call mom"
    )
    assert "[Scene:" in out[1]["content"] and "[Reminders:" in out[1]["content"]


def test_describe_scene_empty():
    from vision.scene import describe_scene

    assert describe_scene([]) == "No notable objects"


def test_describe_scene_yolo():
    from vision.scene import describe_scene

    out = describe_scene([{"cls": 0, "conf": 0.9}, {"cls": 56, "conf": 0.7}])
    assert "person" in out and "chair" in out


def test_format_reminders():
    from utils.reminders import format_reminders_for_llm

    reminders = [{"text": "Call mom", "done": False}, {"text": "Done", "done": True}]
    out = format_reminders_for_llm(reminders)
    assert "Call mom" in out and "Done" not in out


def test_reminders_path():
    from pathlib import Path

    from utils.reminders import get_reminders_path

    p = get_reminders_path(Path("/tmp"))
    assert p.name == "reminders.json"


def main():
    tests = [
        test_build_messages_basic,
        test_build_messages_with_vision,
        test_describe_scene_empty,
        test_describe_scene_yolo,
        test_format_reminders,
        test_reminders_path,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} failed")
        sys.exit(1)
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
