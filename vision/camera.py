"""OpenCV/V4L2 capture with configurable res/FPS."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
