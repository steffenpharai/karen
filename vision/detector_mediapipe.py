"""MediaPipe face/pose (TensorRT or GPU delegate)."""

import logging

logger = logging.getLogger(__name__)


def create_face_detector():
    """Create MediaPipe Face Detection. Returns detector or None."""
    try:
        import mediapipe as mp

        return mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
    except Exception as e:
        logger.warning("MediaPipe face detector failed: %s", e)
        return None


def detect_faces(detector, frame) -> list:
    """Run face detection on BGR frame. Returns list of detections."""
    if detector is None or frame is None:
        return []
    try:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if not results.detections:
            return []
        return [{"bbox": d.location_data.relative_bounding_box} for d in results.detections]
    except Exception as e:
        logger.warning("Face detection failed: %s", e)
        return []
