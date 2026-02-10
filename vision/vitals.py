"""Vital/health monitoring: fatigue (EAR), posture, approximate heart rate (rPPG).

Uses MediaPipe Face Mesh (468 landmarks) for eye tracking and forehead ROI,
MediaPipe Pose for posture analysis.  rPPG is best-effort and returns None
when the signal is unreliable (low confidence, motion, poor lighting).

All functions accept raw BGR frames and are designed to be called from the
vision pipeline at ~10-30 FPS on Jetson Orin Nano 8 GB.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class VitalsResult:
    """Aggregated vitals snapshot from a single frame analysis."""

    fatigue_level: str = "unknown"  # "alert", "mild", "moderate", "severe", "unknown"
    eye_aspect_ratio: float | None = None
    blink_rate_per_min: float | None = None
    posture_score: float | None = None  # 0.0 (bad) to 1.0 (good)
    posture_label: str = "unknown"  # "good", "fair", "poor", "unknown"
    heart_rate_bpm: float | None = None  # None when unreliable
    heart_rate_confidence: float = 0.0  # 0-1
    alerts: list[str] = field(default_factory=list)
    timestamp: float = 0.0


# ── MediaPipe Face Mesh ───────────────────────────────────────────────

def create_face_mesh():
    """Create MediaPipe Face Mesh (468 landmarks). Returns detector or None."""
    try:
        import mediapipe as mp

        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # includes iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        logger.warning("MediaPipe Face Mesh init failed: %s", e)
        return None


def _get_face_mesh_landmarks(face_mesh, frame):
    """Run Face Mesh on a BGR frame, return landmarks list or None."""
    if face_mesh is None or frame is None:
        return None
    try:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            return results.multi_face_landmarks[0].landmark
        return None
    except Exception as e:
        logger.debug("Face mesh processing failed: %s", e)
        return None


# ── Eye Aspect Ratio (EAR) for fatigue ───────────────────────────────

# MediaPipe Face Mesh landmark indices for left and right eye
# Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def _landmark_dist(landmarks, idx_a: int, idx_b: int) -> float:
    """Euclidean distance between two normalised landmarks."""
    a = landmarks[idx_a]
    b = landmarks[idx_b]
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def compute_eye_aspect_ratio(landmarks) -> float | None:
    """Compute average EAR from Face Mesh landmarks.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) per eye.
    Typical open eye: ~0.25-0.30.  Closed: <0.20.
    """
    if landmarks is None:
        return None
    try:
        def _ear(idx):
            v1 = _landmark_dist(landmarks, idx[1], idx[5])
            v2 = _landmark_dist(landmarks, idx[2], idx[4])
            h = _landmark_dist(landmarks, idx[0], idx[3])
            if h < 1e-6:
                return 0.25  # avoid division by zero
            return (v1 + v2) / (2.0 * h)

        left_ear = _ear(_LEFT_EYE_IDX)
        right_ear = _ear(_RIGHT_EYE_IDX)
        return (left_ear + right_ear) / 2.0
    except (IndexError, AttributeError) as e:
        logger.debug("EAR computation failed: %s", e)
        return None


# ── Blink tracker ─────────────────────────────────────────────────────

class BlinkTracker:
    """Track blink rate over a rolling window from EAR values."""

    EAR_BLINK_THRESHOLD = 0.20
    WINDOW_SEC = 60.0  # 1-minute rolling window

    def __init__(self):
        self._blink_times: deque[float] = deque()
        self._prev_below = False

    def update(self, ear: float | None) -> float | None:
        """Feed an EAR value; return blinks/min or None if insufficient data."""
        if ear is None:
            return None
        now = time.monotonic()

        # Detect blink: EAR drops below threshold then returns above
        below = ear < self.EAR_BLINK_THRESHOLD
        if self._prev_below and not below:
            self._blink_times.append(now)
        self._prev_below = below

        # Prune old blinks
        cutoff = now - self.WINDOW_SEC
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        if now - (self._blink_times[0] if self._blink_times else now) < 10.0:
            return None  # not enough history
        elapsed = now - self._blink_times[0] if self._blink_times else self.WINDOW_SEC
        if elapsed < 1.0:
            return None
        return len(self._blink_times) * 60.0 / elapsed


# ── Posture analysis ─────────────────────────────────────────────────

def create_pose_detector():
    """Create MediaPipe Pose detector. Returns detector or None."""
    try:
        import mediapipe as mp

        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # lite for Jetson
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        logger.warning("MediaPipe Pose init failed: %s", e)
        return None


def compute_posture_score(pose_detector, frame) -> tuple[float | None, str]:
    """Compute posture quality from shoulder-ear alignment.

    Returns (score 0-1, label) where 1.0 = perfect upright posture.
    Uses the angle between ear-shoulder line and vertical as proxy.
    """
    if pose_detector is None or frame is None:
        return None, "unknown"
    try:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)
        if not results.pose_landmarks:
            return None, "unknown"

        lm = results.pose_landmarks.landmark
        # Landmarks: 7=left_ear, 8=right_ear, 11=left_shoulder, 12=right_shoulder
        left_ear = lm[7]
        right_ear = lm[8]
        left_shoulder = lm[11]
        right_shoulder = lm[12]

        # Average ear and shoulder positions
        ear_y = (left_ear.y + right_ear.y) / 2.0
        ear_x = (left_ear.x + right_ear.x) / 2.0
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0

        # Angle from vertical (perfect posture = 0 degrees)
        dx = ear_x - shoulder_x
        dy = ear_y - shoulder_y  # negative = ear above shoulder (normal)
        angle = abs(math.degrees(math.atan2(dx, -dy)))  # 0 = upright

        # Map angle to score: 0-10deg = good (1.0), 10-25 = fair, 25+ = poor
        if angle <= 10:
            score = 1.0
            label = "good"
        elif angle <= 25:
            score = max(0.3, 1.0 - (angle - 10) / 30.0)
            label = "fair"
        else:
            score = max(0.0, 0.3 - (angle - 25) / 50.0)
            label = "poor"

        return round(score, 2), label
    except Exception as e:
        logger.debug("Posture computation failed: %s", e)
        return None, "unknown"


# ── rPPG Heart Rate Estimation (experimental) ────────────────────────

class RPPGEstimator:
    """Remote photoplethysmography: approximate heart rate from face color changes.

    **Experimental** — requires:
    - Stable, well-lit face (forehead region visible)
    - Low motion
    - USB webcam at >= 15 FPS

    Returns None when the signal is unreliable.  Never use rPPG as the sole
    basis for medical or safety decisions.
    """

    WINDOW_SEC = 10.0  # seconds of signal to accumulate
    MIN_SAMPLES = 50   # minimum frames before estimating
    HR_LOW = 40.0      # plausible BPM range
    HR_HIGH = 180.0
    CONFIDENCE_THRESHOLD = 0.3  # below this, return None

    # Forehead ROI from Face Mesh landmarks (normalised)
    _FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356,
                     454, 323, 361, 288, 397, 365, 379, 378,
                     400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(self, fps: float = 30.0):
        self._fps = max(fps, 1.0)
        self._green_signal: deque[float] = deque(maxlen=int(self.WINDOW_SEC * fps))
        self._timestamps: deque[float] = deque(maxlen=int(self.WINDOW_SEC * fps))

    def update(self, frame, landmarks) -> tuple[float | None, float]:
        """Feed a frame + Face Mesh landmarks.

        Returns ``(bpm_or_none, confidence)``.
        """
        if frame is None or landmarks is None:
            return None, 0.0

        try:
            h, w = frame.shape[:2]
            # Extract forehead ROI mean green channel
            roi_points = []
            for idx in self._FOREHEAD_IDX:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    roi_points.append((int(lm.x * w), int(lm.y * h)))

            if len(roi_points) < 5:
                return None, 0.0

            roi_points = np.array(roi_points, dtype=np.int32)
            x_min = max(0, roi_points[:, 0].min())
            x_max = min(w, roi_points[:, 0].max())
            y_min = max(0, roi_points[:, 1].min())
            y_max = min(h, roi_points[:, 1].max())

            if x_max - x_min < 5 or y_max - y_min < 5:
                return None, 0.0

            roi = frame[y_min:y_max, x_min:x_max]
            green_mean = float(np.mean(roi[:, :, 1]))  # BGR -> green channel

            now = time.monotonic()
            self._green_signal.append(green_mean)
            self._timestamps.append(now)

            if len(self._green_signal) < self.MIN_SAMPLES:
                return None, 0.0

            return self._estimate_hr()
        except Exception as e:
            logger.debug("rPPG update failed: %s", e)
            return None, 0.0

    def _estimate_hr(self) -> tuple[float | None, float]:
        """Estimate HR from accumulated green-channel signal using FFT."""
        signal = np.array(self._green_signal, dtype=np.float64)
        timestamps = np.array(self._timestamps)

        # Compute actual FPS from timestamps
        dt = timestamps[-1] - timestamps[0]
        if dt < 3.0:  # need at least 3s of data
            return None, 0.0
        actual_fps = (len(signal) - 1) / dt

        # Detrend (remove DC + linear trend)
        signal = signal - np.mean(signal)
        if len(signal) > 1:
            x = np.arange(len(signal), dtype=np.float64)
            coeffs = np.polyfit(x, signal, 1)
            signal = signal - np.polyval(coeffs, x)

        # Simple bandpass via FFT (0.7 - 4.0 Hz = 42-240 BPM)
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / actual_fps)

        # Zero out frequencies outside heart rate range
        low_hz = self.HR_LOW / 60.0
        high_hz = self.HR_HIGH / 60.0
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        power = np.abs(fft_vals) ** 2
        power[~mask] = 0

        if power.max() < 1e-6:
            return None, 0.0

        # Peak frequency
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        bpm = peak_freq * 60.0

        # Confidence: ratio of peak power to total power in band
        band_power = power[mask]
        if band_power.sum() < 1e-6:
            return None, 0.0
        confidence = float(power[peak_idx] / band_power.sum())

        if confidence < self.CONFIDENCE_THRESHOLD:
            return None, confidence
        if bpm < self.HR_LOW or bpm > self.HR_HIGH:
            return None, 0.0

        return round(bpm, 1), round(confidence, 2)


# ── Aggregate vitals analyser ─────────────────────────────────────────

class VitalsAnalyzer:
    """Stateful analyzer: accumulates blinks, rPPG signal across frames.

    Create once and call ``analyze(frame, face_mesh, pose_detector)`` per frame.
    """

    FATIGUE_THRESHOLDS = {
        "alert": (0.24, float("inf")),
        "mild": (0.21, 0.24),
        "moderate": (0.18, 0.21),
        "severe": (0.0, 0.18),
    }

    def __init__(self, fps: float = 30.0):
        self.blink_tracker = BlinkTracker()
        self.rppg = RPPGEstimator(fps=fps)
        self._last_result = VitalsResult()

    def analyze(
        self,
        frame,
        face_mesh=None,
        pose_detector=None,
    ) -> VitalsResult:
        """Run full vitals analysis on a single frame.

        Parameters
        ----------
        frame : BGR numpy array
        face_mesh : MediaPipe Face Mesh instance (or None to skip eye/rPPG)
        pose_detector : MediaPipe Pose instance (or None to skip posture)
        """
        result = VitalsResult(timestamp=time.monotonic())
        alerts: list[str] = []

        # Face Mesh: EAR + rPPG
        landmarks = _get_face_mesh_landmarks(face_mesh, frame)
        if landmarks is not None:
            ear = compute_eye_aspect_ratio(landmarks)
            result.eye_aspect_ratio = ear

            # Fatigue level from EAR
            if ear is not None:
                for level, (lo, hi) in self.FATIGUE_THRESHOLDS.items():
                    if lo <= ear < hi:
                        result.fatigue_level = level
                        break
                if result.fatigue_level in ("moderate", "severe"):
                    alerts.append(f"Fatigue detected ({result.fatigue_level})")

            # Blink rate
            blink_rate = self.blink_tracker.update(ear)
            result.blink_rate_per_min = blink_rate
            if blink_rate is not None and blink_rate < 8:
                alerts.append("Low blink rate — possible eye strain")

            # rPPG heart rate
            bpm, conf = self.rppg.update(frame, landmarks)
            result.heart_rate_bpm = bpm
            result.heart_rate_confidence = conf
            if bpm is not None and bpm > 120:
                alerts.append(f"Elevated heart rate: {bpm:.0f} BPM")

        # Posture
        posture_score, posture_label = compute_posture_score(pose_detector, frame)
        result.posture_score = posture_score
        result.posture_label = posture_label
        if posture_label == "poor":
            alerts.append("Poor posture detected")

        result.alerts = alerts
        self._last_result = result
        return result

    @property
    def last_result(self) -> VitalsResult:
        return self._last_result
