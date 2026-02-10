"""Unit tests for utils.power."""

from unittest.mock import patch

import pytest
from utils.power import (
    get_battery_status,
    get_gpu_utilization,
    get_power_mode,
    get_system_stats,
    get_thermal_temperature,
    get_thermal_warning,
    should_throttle_vision,
)


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


@pytest.mark.unit
def test_get_battery_status_returns_none_or_dict():
    """get_battery_status returns None (no battery) or a dict."""
    out = get_battery_status()
    assert out is None or isinstance(out, dict)


@pytest.mark.unit
def test_get_gpu_utilization_returns_none_or_float():
    """get_gpu_utilization returns None or a float in [0, 100]."""
    out = get_gpu_utilization()
    assert out is None or (isinstance(out, float) and 0 <= out <= 100)


@pytest.mark.unit
def test_get_thermal_temperature_returns_none_or_float():
    """get_thermal_temperature returns None or a float in Celsius."""
    out = get_thermal_temperature()
    assert out is None or (isinstance(out, float) and -40 <= out <= 200)


@pytest.mark.unit
def test_should_throttle_vision_returns_bool():
    """should_throttle_vision returns bool."""
    out = should_throttle_vision()
    assert isinstance(out, bool)


@pytest.mark.unit
def test_should_throttle_high_temp():
    """should_throttle_vision returns True when temperature exceeds threshold."""
    with patch("utils.power.get_thermal_temperature", return_value=95.0):
        with patch("utils.power.get_battery_status", return_value=None):
            assert should_throttle_vision() is True


@pytest.mark.unit
def test_should_throttle_low_battery():
    """should_throttle_vision returns True when battery is critically low."""
    with patch("utils.power.get_thermal_temperature", return_value=50.0):
        with patch("utils.power.get_battery_status", return_value={"capacity_pct": 5}):
            assert should_throttle_vision() is True


@pytest.mark.unit
def test_should_not_throttle_normal():
    """should_throttle_vision returns False under normal conditions."""
    with patch("utils.power.get_thermal_temperature", return_value=50.0):
        with patch("utils.power.get_battery_status", return_value=None):
            assert should_throttle_vision() is False
