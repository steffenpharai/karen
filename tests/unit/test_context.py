"""Unit tests for LLM context building."""

import pytest
from llm.context import build_messages


@pytest.mark.unit
def test_build_messages_basic():
    out = build_messages("You are Jarvis.", "What time is it?")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "You are Jarvis."
    assert out[1]["role"] == "user"
    assert "What time is it?" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_vision_and_reminders():
    out = build_messages(
        "You are Jarvis.",
        "What do you see?",
        vision_description="person(2), laptop(1)",
        reminders_text="Call mom; Review PR",
    )
    assert out[1]["role"] == "user"
    assert "[Scene:" in out[1]["content"]
    assert "person(2), laptop(1)" in out[1]["content"]
    assert "[Reminders:" in out[1]["content"]
    assert "Call mom" in out[1]["content"]


@pytest.mark.unit
def test_build_messages_with_time_and_stats():
    out = build_messages(
        "You are Jarvis.",
        "What time is it?",
        current_time="2026-02-07 12:00:00",
        system_stats="Power mode: MAXN_SUPER",
    )
    assert out[1]["role"] == "user"
    assert "[Current time:" in out[1]["content"]
    assert "2026-02-07" in out[1]["content"]
    assert "[System:" in out[1]["content"]
    assert "MAXN_SUPER" in out[1]["content"]
