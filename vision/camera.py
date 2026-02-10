"""OpenCV/V4L2 capture with configurable res/FPS and low-light detection."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Low-light detection threshold (mean brightness 0-255)
LOW_LIGHT_THRESHOLD = 30


def open_camera(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    device_path: str | None = None,
):
    """Open OpenCV VideoCapture (USB camera). Use device_path (e.g. /dev/video0) if set, else index. Returns cap or None."""
    try:
        import cv2

        source = device_path if device_path else index
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    except Exception as e:
        logger.warning("Camera open failed: %s", e)
        return None


def read_frame(cap) -> Any | None:
    """Read one frame; returns frame or None."""
    if cap is None:
        return None
    try:
        ok, frame = cap.read()
        return frame if ok else None
    except Exception:
        return None


def check_low_light(frame, threshold: int = LOW_LIGHT_THRESHOLD) -> bool:
    """Check if a frame is low-light (mean brightness below threshold).

    Returns True if the frame is too dark for reliable detection.
    Logs a warning on first detection.
    """
    if frame is None:
        return False
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < threshold:
            logger.info(
                "Low-light detected (brightness=%.1f < %d). "
                "Detection accuracy may be reduced.",
                mean_brightness, threshold,
            )
            return True
        return False
    except Exception:
        return False


def get_frame_brightness(frame) -> float | None:
    """Return mean brightness (0-255) of a BGR frame, or None."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception:
        return None
