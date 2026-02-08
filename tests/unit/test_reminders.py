"""Unit tests for reminders."""

import json

import pytest
from utils.reminders import (
    format_reminders_for_llm,
    get_reminders_path,
    load_reminders,
)


@pytest.mark.unit
def test_get_reminders_path(project_root):
    path = get_reminders_path(project_root)
    assert path.name == "reminders.json"
    assert path.parent == project_root


@pytest.mark.unit
def test_load_reminders_empty(project_root):
    assert load_reminders(project_root) == []


@pytest.mark.unit
def test_load_reminders_from_file(project_root, sample_reminders):
    path = get_reminders_path(project_root)
    path.write_text(json.dumps(sample_reminders))
    assert len(load_reminders(project_root)) == 3


@pytest.mark.unit
def test_format_reminders_for_llm(sample_reminders):
    out = format_reminders_for_llm(sample_reminders)
    assert "Call mom" in out
    assert "Review PR" in out
    assert "Buy milk" not in out  # done
    out_2 = format_reminders_for_llm(sample_reminders, max_items=1)
    assert out_2.count(";") == 0 or "Call mom" in out_2
