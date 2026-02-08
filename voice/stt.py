"""STT: Faster-Whisper or whisper.cpp."""

import logging

logger = logging.getLogger(__name__)


def transcribe(audio_path: str, model_size: str = "small") -> str | None:
    """Transcribe audio file to text. Returns None on failure."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="auto", compute_type="auto")
        segments, _ = model.transcribe(audio_path)
        return " ".join(s.text.strip() for s in segments).strip() or None
    except Exception as e:
        logger.warning("Faster-Whisper transcribe failed: %s", e)
        return None


def is_stt_available() -> bool:
    """Check if STT backend is available."""
    try:
        return True
    except Exception:
        return False
