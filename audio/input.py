"""BT/USB mic selection and recording (sounddevice)."""

import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def record_to_file(
    path: str | Path,
    duration_sec: float = 5.0,
    sample_rate: int = 16000,
    device_index: int | None = None,
) -> bool:
    """Record from default or specified mic to a WAV file (mono 16-bit). Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sounddevice as sd

        frames = int(duration_sec * sample_rate)
        rec = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            device=device_index,
        )
        sd.wait()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(rec.tobytes())
        return path.exists()
    except Exception as e:
        logger.warning("Recording failed: %s", e)
        return False


def list_input_devices() -> list[dict]:
    """Enumerate input devices (e.g. Pixel Buds HFP, USB mic)."""
    try:
        import sounddevice as sd

        return [
            {
                "index": i,
                "name": d.get("name", ""),
                "channels": d.get("max_input_channels", 0),
                "sr": d.get("default_samplerate"),
            }
            for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) > 0
        ]
    except Exception as e:
        logger.warning("Could not list input devices: %s", e)
        return []


def get_default_input_index() -> int | None:
    """Default system input device index."""
    try:
        import sounddevice as sd

        return sd.default.device[0]
    except Exception:
        return None
