"""Device enumeration and default sink/source hints (pactl/wpctl)."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_default_sink_name() -> str | None:
    """Query default Pulse sink (e.g. Pixel Buds A2DP)."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception as e:
        logger.debug("pactl get-default-sink failed: %s", e)
        return None


def get_default_source_name() -> str | None:
    """Query default Pulse source (e.g. HFP mic or USB mic)."""
    try:
        out = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception as e:
        logger.debug("pactl get-default-source failed: %s", e)
        return None
