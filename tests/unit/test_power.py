"""Unit tests for utils.power."""

import pytest
from utils.power import get_power_mode, get_system_stats, get_thermal_warning


@pytest.mark.unit
def test_get_power_mode_returns_none_or_str():
    """get_power_mode returns None or a non-empty string."""
    out = get_power_mode()
    assert out is None or (isinstance(out, str) and out.strip() != "")


@pytest.mark.unit
def test_get_system_stats_returns_none_or_str():
    """get_system_stats returns None or a string for LLM context."""
    out = get_system_stats()
    assert out is None or (isinstance(out, str) and len(out) > 0)


@pytest.mark.unit
def test_get_thermal_warning_returns_none_or_str():
    """get_thermal_warning returns None or a warning string."""
    out = get_thermal_warning()
    assert out is None or (isinstance(out, str) and "thermal" in out.lower() or "throttl" in out.lower())
