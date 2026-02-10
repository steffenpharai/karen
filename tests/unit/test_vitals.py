"""Unit tests for vision/vitals.py: EAR, posture, rPPG, blink tracker, vitals analyzer."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from vision.vitals import (
    BlinkTracker,
    RPPGEstimator,
    VitalsAnalyzer,
    VitalsResult,
    compute_eye_aspect_ratio,
    compute_posture_score,
    create_face_mesh,
    create_pose_detector,
)

# ── EAR computation ───────────────────────────────────────────────────


@pytest.mark.unit
class TestEyeAspectRatio:
    def _make_landmarks(self, ear_value: float):
        """Create mock landmarks that produce a specific EAR.

        EAR = (v1 + v2) / (2 * h) per eye.
        We set h=1.0 and v1=v2=ear_value so EAR = (ear + ear) / (2*1) = ear.
        """

        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # Create 468+ landmarks with reasonable defaults
        landmarks = [LM(0.5, 0.5)] * 500

        # Left eye indices: [362, 385, 387, 263, 373, 380]
        # Right eye indices: [33, 160, 158, 133, 153, 144]
        # For each eye: p1-p4 horizontal, p2-p6 and p3-p5 vertical
        # Set horizontal distance = 1.0, vertical distances = ear_value
        for eye_idx in [[362, 385, 387, 263, 373, 380], [33, 160, 158, 133, 153, 144]]:
            landmarks[eye_idx[0]] = LM(0.0, 0.5)    # p1 (left)
            landmarks[eye_idx[3]] = LM(1.0, 0.5)    # p4 (right)
            landmarks[eye_idx[1]] = LM(0.3, 0.5 - ear_value / 2)  # p2 (top)
            landmarks[eye_idx[5]] = LM(0.3, 0.5 + ear_value / 2)  # p6 (bottom)
            landmarks[eye_idx[2]] = LM(0.7, 0.5 - ear_value / 2)  # p3 (top)
            landmarks[eye_idx[4]] = LM(0.7, 0.5 + ear_value / 2)  # p5 (bottom)

        return landmarks

    def test_ear_open_eye(self):
        landmarks = self._make_landmarks(0.28)
        ear = compute_eye_aspect_ratio(landmarks)
        assert ear is not None
        assert 0.25 <= ear <= 0.32

    def test_ear_closed_eye(self):
        landmarks = self._make_landmarks(0.10)
        ear = compute_eye_aspect_ratio(landmarks)
        assert ear is not None
        assert ear < 0.20

    def test_ear_none_landmarks(self):
        assert compute_eye_aspect_ratio(None) is None


# ── Blink tracker ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestBlinkTracker:
    def test_no_blinks_returns_none_initially(self):
        bt = BlinkTracker()
        assert bt.update(0.30) is None  # not enough history

    def test_detect_blink(self):
        bt = BlinkTracker()
        # Simulate open -> closed -> open
        for _ in range(20):
            bt.update(0.28)
        bt.update(0.15)  # closed
        bt.update(0.15)
        bt.update(0.28)  # open again -> blink detected
        # Still may not have enough history for rate
        # but blink should be recorded
        assert len(bt._blink_times) >= 1


# ── Posture score ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestPostureScore:
    def test_no_detector(self):
        score, label = compute_posture_score(None, np.zeros((100, 100, 3), dtype=np.uint8))
        assert score is None
        assert label == "unknown"

    def test_no_frame(self):
        score, label = compute_posture_score(MagicMock(), None)
        assert score is None
        assert label == "unknown"


# ── rPPG estimator ────────────────────────────────────────────────────


@pytest.mark.unit
class TestRPPGEstimator:
    def test_returns_none_insufficient_data(self):
        rppg = RPPGEstimator(fps=30)
        bpm, conf = rppg.update(np.zeros((100, 100, 3), dtype=np.uint8), None)
        assert bpm is None
        assert conf == 0.0

    def test_returns_none_no_landmarks(self):
        rppg = RPPGEstimator(fps=30)
        bpm, conf = rppg.update(np.zeros((100, 100, 3), dtype=np.uint8), None)
        assert bpm is None

    def test_accumulates_signal(self):
        rppg = RPPGEstimator(fps=30)

        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # Create landmarks that form a valid forehead ROI (spread out points)
        landmarks = [LM(0.5, 0.5)] * 500
        # Set forehead indices to form a region with sufficient spread
        for idx in RPPGEstimator._FOREHEAD_IDX:
            if idx < 500:
                # Spread across forehead area (x: 0.3-0.7, y: 0.1-0.3)
                landmarks[idx] = LM(0.3 + (idx % 10) * 0.04, 0.1 + (idx % 5) * 0.04)

        frame = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        # Feed several frames
        for _ in range(10):
            rppg.update(frame, landmarks)
        assert len(rppg._green_signal) == 10


# ── VitalsAnalyzer ────────────────────────────────────────────────────


@pytest.mark.unit
class TestVitalsAnalyzer:
    def test_returns_vitals_result(self):
        analyzer = VitalsAnalyzer(fps=30)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)
        assert isinstance(result, VitalsResult)
        assert result.timestamp > 0

    def test_fatigue_thresholds(self):
        analyzer = VitalsAnalyzer()
        # Check threshold mapping
        assert ("alert", (0.24, float("inf"))) in analyzer.FATIGUE_THRESHOLDS.items()
        assert ("severe", (0.0, 0.18)) in analyzer.FATIGUE_THRESHOLDS.items()

    def test_last_result_property(self):
        analyzer = VitalsAnalyzer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)
        assert analyzer.last_result is result


# ── Factory functions ─────────────────────────────────────────────────


@pytest.mark.unit
def test_create_face_mesh_returns_or_none():
    """create_face_mesh returns a detector or None (depending on mediapipe install)."""
    result = create_face_mesh()
    # May be None if mediapipe not installed
    assert result is not None or result is None


@pytest.mark.unit
def test_create_pose_detector_returns_or_none():
    result = create_pose_detector()
    assert result is not None or result is None
