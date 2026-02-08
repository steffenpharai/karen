"""Piper TTS → default sink (A2DP)."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def play_wav(path: str | Path) -> bool:
    """Play a WAV file to the default sink (e.g. Pixel Buds A2DP)."""
    path = Path(path)
    if not path.exists():
        logger.error("WAV not found: %s", path)
        return False
    try:
        subprocess.run(
            ["aplay", str(path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Playback failed: %s", e)
        return False
