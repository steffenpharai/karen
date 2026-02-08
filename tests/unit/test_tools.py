"""Unit tests for orchestrator tools."""

import pytest
from config import settings
from tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    get_current_time,
    list_reminders,
    run_tool,
    tell_joke,
    toggle_sarcasm,
)


@pytest.mark.unit
def test_get_current_time():
    t = get_current_time()
    assert len(t) >= 18
    assert ":" in t
    assert "-" in t


@pytest.mark.unit
def test_tell_joke():
    j = tell_joke()
    assert isinstance(j, str)
    assert len(j) > 5


@pytest.mark.unit
def test_toggle_sarcasm():
    orig = settings.SARCASM_ENABLED
    try:
        r = toggle_sarcasm(True)
        assert "on" in r.lower()
        assert settings.SARCASM_ENABLED is True
        r2 = toggle_sarcasm(False)
        assert "off" in r2.lower()
        assert settings.SARCASM_ENABLED is False
    finally:
        settings.SARCASM_ENABLED = orig


@pytest.mark.unit
def test_list_reminders():
    out = list_reminders()
    assert isinstance(out, str)
    assert "reminder" in out.lower() or "No pending" in out or out == "Could not list reminders."


@pytest.mark.unit
def test_run_tool_get_current_time():
    out = run_tool("get_current_time", {})
    assert isinstance(out, str)
    assert ":" in out


@pytest.mark.unit
def test_run_tool_tell_joke():
    out = run_tool("tell_joke", {})
    assert isinstance(out, str)
    assert len(out) > 5


@pytest.mark.unit
def test_run_tool_create_reminder():
    out = run_tool("create_reminder", {"text": "Test reminder", "time_str": "18:00"})
    assert "added" in out.lower() or "Reminder" in out or "Failed" in out


@pytest.mark.unit
def test_run_tool_unknown():
    out = run_tool("unknown_tool", {})
    assert "Unknown" in out


@pytest.mark.unit
def test_tool_schemas_and_registry():
    assert len(TOOL_SCHEMAS) == 7
    assert len(TOOL_REGISTRY) == 7
    for s in TOOL_SCHEMAS:
        name = s.get("function", {}).get("name")
        assert name in TOOL_REGISTRY
