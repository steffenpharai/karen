"""Wake word: openWakeWord or Porcupine 'Jarvis' / 'Hey Jarvis'."""

import logging
import os
import threading
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

WAKE_SAMPLE_RATE = 16000
WAKE_CHUNK_SAMPLES = 1280
WAKE_THRESHOLD = 0.5

# Use TFLite and hey_jarvis only: ONNX paths (alexa_v0.1.onnx etc.) are not shipped in the package.
WAKE_MODEL_NAME = "hey_jarvis"


def _ensure_wakeword_models() -> None:
    """Ensure hey_jarvis (and feature models) exist; download if missing (no ONNX workaround)."""
    try:
        import openwakeword

        path = openwakeword.MODELS.get(WAKE_MODEL_NAME, {}).get("model_path", "")
        if path and not os.path.exists(path):
            logger.info("Downloading openWakeWord model %s (one-time)...", WAKE_MODEL_NAME)
            openwakeword.utils.download_models(model_names=["hey_jarvis_v0.1"])
    except Exception as e:
        logger.debug("openWakeWord ensure models: %s", e)


def _run_wake_loop_impl(
    callback: Callable[[], None],
    stop_event: threading.Event,
    device_index: int | None,
) -> None:
    """Inner loop: stream mic → openWakeWord → callback on detection."""
    try:
        import sounddevice as sd
        from openwakeword.model import Model
    except Exception as e:
        logger.warning("Wake word dependencies missing: %s", e)
        return
    _ensure_wakeword_models()
    try:
        oww = Model(
            inference_framework="tflite",
            wakeword_models=[WAKE_MODEL_NAME],
        )
        if not oww.prediction_buffer:
            logger.warning("No openWakeWord models loaded")
            return
        model_names = list(oww.prediction_buffer.keys())
        logger.info("Wake word models loaded: %s", model_names)
    except Exception as e:
        logger.warning("openWakeWord init failed: %s", e)
        return
    stream = None
    try:
        stream = sd.InputStream(
            device=device_index,
            channels=1,
            dtype=np.int16,
            samplerate=WAKE_SAMPLE_RATE,
            blocksize=WAKE_CHUNK_SAMPLES,
        )
        stream.start()
        while not stop_event.is_set():
            chunk, _ = stream.read(WAKE_CHUNK_SAMPLES)
            if chunk is None or chunk.size == 0:
                continue
            audio = np.frombuffer(chunk, dtype=np.int16)
            oww.predict(audio)
            for name in model_names:
                buf = oww.prediction_buffer.get(name, [])
                if buf and buf[-1] > WAKE_THRESHOLD:
                    logger.info("Wake word detected: %s", name)
                    try:
                        callback()
                    except Exception as e:
                        logger.exception("Wake callback error: %s", e)
                    break
    except Exception as e:
        logger.warning("Wake loop error: %s", e)
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def run_wake_loop(
    callback: Callable[[], None],
    device_index: int | None = None,
) -> threading.Event:
    """Start wake-word detection in a background thread. Returns a stop Event; set it to stop."""
    stop = threading.Event()
    thread = threading.Thread(
        target=_run_wake_loop_impl,
        args=(callback, stop, device_index),
        daemon=True,
    )
    thread.start()
    return stop


def create_wake_detector(
    callback: Callable[[], None],
    model_name: str = "jarvis",
) -> object | None:
    """Create wake-word detector; call callback when triggered. For compatibility; use run_wake_loop."""
    return {"callback": callback}


def is_wake_supported() -> bool:
    """Check if wake word backend is available."""
    try:
        _ensure_wakeword_models()
        from openwakeword.model import Model

        m = Model(
            inference_framework="tflite",
            wakeword_models=[WAKE_MODEL_NAME],
        )
        return len(m.prediction_buffer) > 0
    except Exception:
        return False
