"""Pytest fixtures and config."""

import pytest


@pytest.fixture
def project_root(tmp_path):
    """Temporary project root for tests that need a directory."""
    return tmp_path


@pytest.fixture
def sample_reminders():
    """Sample reminders list for LLM context tests."""
    return [
        {"text": "Call mom", "done": False},
        {"text": "Buy milk", "done": True},
        {"text": "Review PR", "done": False},
    ]
