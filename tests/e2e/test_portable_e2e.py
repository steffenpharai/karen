"""E2E tests for portable mode configuration and settings."""

import pytest
from config import settings


@pytest.mark.e2e
def test_portable_settings_defaults():
    """Portable mode settings have sensible defaults."""
    assert hasattr(settings, "PORTABLE_MODE")
    assert hasattr(settings, "PORTABLE_WIDTH")
    assert hasattr(settings, "PORTABLE_HEIGHT")
    assert hasattr(settings, "PORTABLE_FPS")
    assert hasattr(settings, "PORTABLE_DEPTH_SKIP")
    assert hasattr(settings, "PORTABLE_VITALS_SKIP")

    assert settings.PORTABLE_WIDTH == 320
    assert settings.PORTABLE_HEIGHT == 320
    assert settings.PORTABLE_FPS == 10
    assert settings.PORTABLE_DEPTH_SKIP == 3
    assert settings.PORTABLE_VITALS_SKIP == 5


@pytest.mark.e2e
def test_portable_mode_can_be_enabled():
    """PORTABLE_MODE can be toggled."""
    original = settings.PORTABLE_MODE
    try:
        settings.PORTABLE_MODE = True
        assert settings.PORTABLE_MODE is True
        settings.PORTABLE_MODE = False
        assert settings.PORTABLE_MODE is False
    finally:
        settings.PORTABLE_MODE = original


@pytest.mark.e2e
def test_thermal_pause_threshold():
    """THERMAL_PAUSE_THRESHOLD has a sensible default."""
    assert hasattr(settings, "THERMAL_PAUSE_THRESHOLD")
    assert settings.THERMAL_PAUSE_THRESHOLD == 80.0


@pytest.mark.e2e
def test_depth_settings():
    """Depth settings exist and are sensible."""
    assert hasattr(settings, "DEPTH_ENGINE_PATH")
    assert hasattr(settings, "DEPTH_ENABLED")
    assert "depth_anything" in settings.DEPTH_ENGINE_PATH.lower()
