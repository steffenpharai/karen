"""E2E tests for audio (device enumeration, default sink/source). Skip when no Pulse/BT."""

import pytest
from audio import bluetooth
from audio import input as audio_input


@pytest.mark.e2e
def test_list_input_devices():
    devices = audio_input.list_input_devices()
    # May be empty in CI; on Jetson with BT we expect at least one or default
    assert isinstance(devices, list)


@pytest.mark.e2e
def test_bluetooth_default_sink_source():
    sink = bluetooth.get_default_sink_name()
    source = bluetooth.get_default_source_name()
    # One or both may be None if Pulse not running or no devices
    assert sink is None or isinstance(sink, str)
    assert source is None or isinstance(source, str)
