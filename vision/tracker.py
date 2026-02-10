"""ByteTrack-lite: lightweight multi-object tracker using IoU matching + Kalman filter.

Pure Python — no Re-ID model, 0 MB extra GPU memory.  Designed for the Jarvis
vision pipeline on Jetson Orin Nano 8 GB.

Each detection from YOLOE is matched to existing tracks by IoU.  Unmatched
detections create new tracks; unmatched tracks are aged out after a timeout.
Velocity vectors are computed per track for the threat module.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class TrackedObject:
    """A tracked object with persistent ID across frames."""

    track_id: int
    xyxy: list[float]          # current bounding box [x1, y1, x2, y2]
    cls: int = 0               # class ID from YOLOE
    class_name: str = ""       # resolved class name
    conf: float = 0.0          # last detection confidence
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])  # pixels/sec [vx, vy]
    frames_seen: int = 0       # total frames this track has been active
    age: int = 0               # frames since last matched detection
    last_seen: float = 0.0     # monotonic timestamp of last match
    depth: float | None = None  # depth from depth module (if available)


# ── Kalman filter (simplified 2D center + velocity) ───────────────────


class SimpleKalmanBox:
    """Kalman-ish state for a bounding box: tracks center (cx, cy) + size (w, h) + velocity."""

    def __init__(self, xyxy: list[float]):
        x1, y1, x2, y2 = xyxy
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w = x2 - x1
        self.h = y2 - y1
        self.vx = 0.0
        self.vy = 0.0
        self._last_time = time.monotonic()

    def predict(self) -> list[float]:
        """Predict next position using constant-velocity model."""
        now = time.monotonic()
        dt = now - self._last_time
        if dt > 5.0:
            dt = 0.0  # don't extrapolate too far
        pred_cx = self.cx + self.vx * dt
        pred_cy = self.cy + self.vy * dt
        x1 = pred_cx - self.w / 2.0
        y1 = pred_cy - self.h / 2.0
        x2 = pred_cx + self.w / 2.0
        y2 = pred_cy + self.h / 2.0
        return [x1, y1, x2, y2]

    def update(self, xyxy: list[float]) -> None:
        """Update state with a matched detection."""
        now = time.monotonic()
        dt = now - self._last_time
        if dt < 0.001:
            dt = 0.033  # ~30 FPS default

        x1, y1, x2, y2 = xyxy
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0

        # Exponential moving average for velocity (alpha=0.4)
        alpha = 0.4
        raw_vx = (new_cx - self.cx) / dt
        raw_vy = (new_cy - self.cy) / dt
        self.vx = alpha * raw_vx + (1 - alpha) * self.vx
        self.vy = alpha * raw_vy + (1 - alpha) * self.vy

        self.cx = new_cx
        self.cy = new_cy
        self.w = x2 - x1
        self.h = y2 - y1
        self._last_time = now

    @property
    def velocity_vec(self) -> list[float]:
        return [self.vx, self.vy]


# ── IoU computation ───────────────────────────────────────────────────


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(boxes_a: list[list[float]], boxes_b: list[list[float]]) -> np.ndarray:
    """Compute IoU matrix of shape (len(a), len(b))."""
    n, m = len(boxes_a), len(boxes_b)
    mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            mat[i, j] = _iou(boxes_a[i], boxes_b[j])
    return mat


# ── Linear assignment (greedy, avoids scipy dependency) ───────────────


def _greedy_assign(
    cost_matrix: np.ndarray,
    threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy assignment: match rows to columns by highest IoU above threshold.

    Returns (matched_pairs, unmatched_rows, unmatched_cols).
    """
    n, m = cost_matrix.shape
    matched_pairs = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    # Sort all entries by descending IoU
    entries = []
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] >= threshold:
                entries.append((cost_matrix[i, j], i, j))
    entries.sort(key=lambda x: x[0], reverse=True)

    for _, i, j in entries:
        if i not in used_rows and j not in used_cols:
            matched_pairs.append((i, j))
            used_rows.add(i)
            used_cols.add(j)

    unmatched_rows = [i for i in range(n) if i not in used_rows]
    unmatched_cols = [j for j in range(m) if j not in used_cols]
    return matched_pairs, unmatched_rows, unmatched_cols


# ── ByteTrack-lite tracker ────────────────────────────────────────────


class ByteTrackLite:
    """Lightweight multi-object tracker.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without a match before removal.
    iou_threshold : float
        Minimum IoU for a detection-track match.
    min_hits : int
        Minimum matched frames before a track is considered confirmed.
    """

    def __init__(
        self,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
    ):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: list[dict] = []  # internal track state

    def update(self, detections: list[dict]) -> list[TrackedObject]:
        """Match detections to existing tracks and return updated tracked objects.

        Parameters
        ----------
        detections : list of dicts with keys: xyxy, conf, cls, (optional) class_name

        Returns
        -------
        list of TrackedObject for all confirmed tracks.
        """
        now = time.monotonic()

        # Predict positions for existing tracks
        predicted_boxes = []
        for t in self._tracks:
            kf: SimpleKalmanBox = t["kf"]
            pred = kf.predict()
            t["predicted_xyxy"] = pred
            predicted_boxes.append(pred)

        # Detection boxes
        det_boxes = [d.get("xyxy", [0, 0, 0, 0]) for d in detections]

        if predicted_boxes and det_boxes:
            iou_mat = _iou_matrix(predicted_boxes, det_boxes)
            matched, unmatched_tracks, unmatched_dets = _greedy_assign(
                iou_mat, threshold=self.iou_threshold
            )
        else:
            matched = []
            unmatched_tracks = list(range(len(self._tracks)))
            unmatched_dets = list(range(len(detections)))

        # Update matched tracks
        for t_idx, d_idx in matched:
            track = self._tracks[t_idx]
            det = detections[d_idx]
            xyxy = det.get("xyxy", [0, 0, 0, 0])
            track["kf"].update(xyxy)
            track["xyxy"] = xyxy
            track["cls"] = det.get("cls", 0)
            track["class_name"] = det.get("class_name", "")
            track["conf"] = det.get("conf", 0.0)
            track["frames_seen"] += 1
            track["age"] = 0
            track["last_seen"] = now

        # Age unmatched tracks
        for t_idx in unmatched_tracks:
            self._tracks[t_idx]["age"] += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            xyxy = det.get("xyxy", [0, 0, 0, 0])
            self._tracks.append({
                "track_id": self._next_id,
                "kf": SimpleKalmanBox(xyxy),
                "xyxy": xyxy,
                "cls": det.get("cls", 0),
                "class_name": det.get("class_name", ""),
                "conf": det.get("conf", 0.0),
                "frames_seen": 1,
                "age": 0,
                "last_seen": now,
            })
            self._next_id += 1

        # Remove dead tracks
        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]

        # Return confirmed tracks
        results = []
        for t in self._tracks:
            if t["frames_seen"] >= self.min_hits:
                kf: SimpleKalmanBox = t["kf"]
                results.append(TrackedObject(
                    track_id=t["track_id"],
                    xyxy=t["xyxy"],
                    cls=t["cls"],
                    class_name=t["class_name"],
                    conf=t["conf"],
                    velocity=kf.velocity_vec,
                    frames_seen=t["frames_seen"],
                    age=t["age"],
                    last_seen=t["last_seen"],
                ))
        return results

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def track_count(self) -> int:
        """Number of active (alive) tracks."""
        return len(self._tracks)
