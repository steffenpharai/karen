"""Threat & anomaly detection: heuristic scoring from tracked objects, vitals, and depth.

No extra model — pure logic using tracker velocity, detection classes, depth,
and vitals data.  Designed for the Jarvis MCU-inspired vision pipeline.

Threat levels (0-10):
  0-2: Normal / peaceful
  3-4: Minor anomaly (e.g. unfamiliar movement)
  5-6: Moderate (rapid approach, user fatigue in anomalous setting)
  7-8: Elevated (multiple fast-moving unknowns, very close approach)
  9-10: Critical (immediate danger indicators)
"""

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThreatAssessment:
    """Result of threat analysis for the current scene."""

    level: int = 0                        # 0-10
    label: str = "clear"                  # clear, low, moderate, elevated, critical
    alerts: list[str] = field(default_factory=list)
    recommendation: str = ""
    timestamp: float = 0.0


# ── Known safe / threat classes ───────────────────────────────────────

# COCO-derived class categorisation for threat heuristics
_SAFE_CLASSES = frozenset({
    "person", "cat", "dog", "bird", "laptop", "cell phone", "book",
    "cup", "bottle", "chair", "couch", "bed", "dining table", "tv",
    "remote", "keyboard", "mouse", "potted plant", "clock", "vase",
    "teddy bear", "toothbrush", "bowl", "fork", "knife", "spoon",
    "backpack", "handbag", "suitcase", "umbrella", "tie",
})

_VEHICLE_CLASSES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "boat",
    "airplane",
})

_WEAPON_LIKE_CLASSES = frozenset({
    "knife", "scissors", "baseball bat",
})


# ── Threat scorer ─────────────────────────────────────────────────────


class ThreatScorer:
    """Stateful threat scorer with temporal smoothing.

    Call ``score_scene()`` each frame with tracked objects, vitals, and depth
    to get a ThreatAssessment.  The level is smoothed over a short window
    to avoid flicker.
    """

    # Approaching velocity threshold (pixels/sec toward camera center)
    APPROACH_VELOCITY_THRESHOLD = 50.0
    # Close depth threshold (relative 0-1, lower = closer)
    CLOSE_DEPTH_THRESHOLD = 0.15
    # Sudden crowd increase
    CROWD_SURGE_THRESHOLD = 3

    def __init__(self, smoothing_frames: int = 10):
        self._smoothing_frames = smoothing_frames
        self._level_history: list[int] = []
        self._prev_person_count = 0
        self._prev_track_ids: set[int] = set()

    def score_scene(
        self,
        tracked_objects: list | None = None,
        vitals=None,
        depth_map=None,
    ) -> ThreatAssessment:
        """Evaluate threat level from current scene data.

        Parameters
        ----------
        tracked_objects : list of TrackedObject (from ByteTrackLite)
        vitals : VitalsResult (from VitalsAnalyzer), or None
        depth_map : HxW float32 depth (0=near, 1=far), or None

        Returns
        -------
        ThreatAssessment
        """
        alerts: list[str] = []
        raw_level = 0
        now = time.monotonic()

        tracks = tracked_objects or []

        # ── 1. Person analysis ────────────────────────────────────────
        person_tracks = [t for t in tracks if self._is_person(t)]
        person_count = len(person_tracks)

        # Sudden crowd surge
        new_ids = {t.track_id for t in tracks} - self._prev_track_ids
        if len(new_ids) >= self.CROWD_SURGE_THRESHOLD:
            raw_level += 3
            alerts.append(f"{len(new_ids)} new objects appeared suddenly")

        self._prev_track_ids = {t.track_id for t in tracks}

        # ── 2. Approach detection ─────────────────────────────────────
        for t in person_tracks:
            speed = math.sqrt(t.velocity[0] ** 2 + t.velocity[1] ** 2)
            # Check if approaching (positive vy = moving down in frame = approaching)
            if speed > self.APPROACH_VELOCITY_THRESHOLD:
                raw_level += 2
                alerts.append(
                    f"Person (track {t.track_id}) moving rapidly "
                    f"({speed:.0f} px/s)"
                )

            # Check depth (close proximity)
            if t.depth is not None and t.depth < self.CLOSE_DEPTH_THRESHOLD:
                raw_level += 2
                alerts.append(
                    f"Person (track {t.track_id}) very close "
                    f"(depth={t.depth:.2f})"
                )

        # ── 3. Weapon-like objects ────────────────────────────────────
        for t in tracks:
            name = getattr(t, "class_name", "").lower()
            if name in _WEAPON_LIKE_CLASSES and not self._is_person(t):
                raw_level += 2
                alerts.append(f"Potential weapon detected: {name} (track {t.track_id})")

        # ── 4. Unknown / unclassified large objects ───────────────────
        for t in tracks:
            name = getattr(t, "class_name", "").lower()
            if not name or name.startswith("class_"):
                # Large bounding box + unknown = suspicious
                xyxy = t.xyxy
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > 50000:  # large unknown object
                    raw_level += 1
                    alerts.append(f"Large unidentified object (track {t.track_id})")

        # ── 5. Vitals factor ──────────────────────────────────────────
        if vitals is not None:
            fatigue = getattr(vitals, "fatigue_level", "unknown")
            if fatigue in ("moderate", "severe") and raw_level >= 2:
                raw_level += 1
                alerts.append(
                    f"User fatigued ({fatigue}) in potentially anomalous setting"
                )

        # ── 6. Vehicle proximity ──────────────────────────────────────
        for t in tracks:
            name = getattr(t, "class_name", "").lower()
            if name in _VEHICLE_CLASSES:
                speed = math.sqrt(t.velocity[0] ** 2 + t.velocity[1] ** 2)
                if speed > self.APPROACH_VELOCITY_THRESHOLD * 1.5:
                    raw_level += 3
                    alerts.append(
                        f"Fast-moving vehicle: {name} (track {t.track_id}, "
                        f"{speed:.0f} px/s)"
                    )

        # ── Clamp and smooth ──────────────────────────────────────────
        raw_level = min(10, max(0, raw_level))
        self._level_history.append(raw_level)
        if len(self._level_history) > self._smoothing_frames:
            self._level_history = self._level_history[-self._smoothing_frames:]

        smoothed = int(round(sum(self._level_history) / len(self._level_history)))
        smoothed = min(10, max(0, smoothed))

        label = self._level_to_label(smoothed)
        recommendation = self._recommend(smoothed, alerts)

        self._prev_person_count = person_count

        return ThreatAssessment(
            level=smoothed,
            label=label,
            alerts=alerts,
            recommendation=recommendation,
            timestamp=now,
        )

    @staticmethod
    def _is_person(track) -> bool:
        """Check if a tracked object is a person."""
        name = getattr(track, "class_name", "").lower()
        cls_id = getattr(track, "cls", -1)
        return name == "person" or cls_id == 0

    @staticmethod
    def _level_to_label(level: int) -> str:
        if level <= 2:
            return "clear"
        if level <= 4:
            return "low"
        if level <= 6:
            return "moderate"
        if level <= 8:
            return "elevated"
        return "critical"

    @staticmethod
    def _recommend(level: int, alerts: list[str]) -> str:
        if level <= 2:
            return ""
        if level <= 4:
            return "Stay aware of your surroundings, sir."
        if level <= 6:
            return "I'd recommend increased vigilance, sir."
        if level <= 8:
            return "Might I suggest relocating to a safer position, sir."
        return "Immediate evasive action recommended, sir."

    def reset(self) -> None:
        """Clear history."""
        self._level_history.clear()
        self._prev_person_count = 0
        self._prev_track_ids.clear()
