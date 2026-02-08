"""Unit tests for memory (session summary, load/save)."""

from pathlib import Path

import pytest
from memory import load_session, load_summary, save_session, save_summary


@pytest.mark.unit
def test_load_save_summary(project_root):
    base = Path(project_root)
    save_summary(base, "Test summary.")
    assert load_summary(base) == "Test summary."
    save_summary(base, "")
    assert load_summary(base) == ""


@pytest.mark.unit
def test_load_session(project_root):
    base = Path(project_root)
    s = load_session(base)
    assert "summary" in s
    assert "data_dir" in s
    assert s["data_dir"] == base


@pytest.mark.unit
def test_save_session(project_root):
    base = Path(project_root)
    memory = load_session(base)
    memory["summary"] = "Brief context."
    save_session(memory)
    loaded = load_session(base)
    assert loaded["summary"] == "Brief context."
