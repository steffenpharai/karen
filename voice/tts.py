"""TTS: Piper (British male or Jarvis-like)."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def synthesize(
    text: str, voice: str = "en_GB-alan-medium", out_dir: Path | None = None
) -> Path | None:
    """Synthesize text to WAV using Piper. Returns path to WAV or None."""
    if not text.strip():
        return None
    out_dir = out_dir or Path(tempfile.gettempdir())
    out_path = out_dir / "jarvis_tts.wav"
    # Use same interpreter as app so venv deps (e.g. pathvalidate) are available
    piper_cmd = [sys.executable, "-m", "piper", "--model", voice, "--output_file", str(out_path)]
    try:
        proc = subprocess.run(
            piper_cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and out_path.exists():
            return out_path
        logger.warning("Piper failed: %s", proc.stderr.decode() if proc.stderr else proc.returncode)
        return None
    except FileNotFoundError:
        logger.warning("Piper not available (pip install piper-tts)")
        return None
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return None


def is_tts_available() -> bool:
    """Check if Piper is available."""
    try:
        subprocess.run(
            [sys.executable, "-m", "piper", "--help"],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False
