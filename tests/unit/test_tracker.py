"""Unit tests for vision/tracker.py: ByteTrackLite, IoU, Kalman filter."""

import time

import numpy as np
import pytest
from vision.tracker import (
    ByteTrackLite,
    SimpleKalmanBox,
    TrackedObject,
    _greedy_assign,
    _iou,
    _iou_matrix,
)

# ── IoU computation ───────────────────────────────────────────────────


@pytest.mark.unit
class TestIoU:
    def test_perfect_overlap(self):
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.1 < iou < 0.5

    def test_zero_area(self):
        assert _iou([0, 0, 0, 0], [0, 0, 10, 10]) == pytest.approx(0.0)

    def test_iou_matrix(self):
        boxes_a = [[0, 0, 10, 10], [20, 20, 30, 30]]
        boxes_b = [[0, 0, 10, 10], [100, 100, 110, 110]]
        mat = _iou_matrix(boxes_a, boxes_b)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)
        assert mat[1, 0] == pytest.approx(0.0)


# ── Greedy assignment ─────────────────────────────────────────────────


@pytest.mark.unit
class TestGreedyAssign:
    def test_empty(self):
        matched, ur, uc = _greedy_assign(np.zeros((0, 0)))
        assert matched == []
        assert ur == []
        assert uc == []

    def test_perfect_match(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        matched, ur, uc = _greedy_assign(mat, threshold=0.3)
        assert len(matched) == 2
        assert len(ur) == 0
        assert len(uc) == 0

    def test_no_match(self):
        mat = np.array([[0.1, 0.1], [0.1, 0.1]])
        matched, ur, uc = _greedy_assign(mat, threshold=0.3)
        assert len(matched) == 0
        assert len(ur) == 2
        assert len(uc) == 2

    def test_partial_match(self):
        mat = np.array([[0.8, 0.1], [0.1, 0.1]])
        matched, ur, uc = _greedy_assign(mat, threshold=0.3)
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert 1 in ur
        assert 1 in uc


# ── SimpleKalmanBox ───────────────────────────────────────────────────


@pytest.mark.unit
class TestSimpleKalmanBox:
    def test_init(self):
        kf = SimpleKalmanBox([10, 20, 50, 60])
        assert kf.cx == pytest.approx(30.0)
        assert kf.cy == pytest.approx(40.0)
        assert kf.w == pytest.approx(40.0)
        assert kf.h == pytest.approx(40.0)

    def test_predict(self):
        kf = SimpleKalmanBox([0, 0, 10, 10])
        pred = kf.predict()
        assert len(pred) == 4
        # No velocity yet, so predict should be near original
        assert pred[0] == pytest.approx(0.0, abs=1)

    def test_update_velocity(self):
        kf = SimpleKalmanBox([0, 0, 10, 10])
        kf._last_time = time.monotonic() - 1.0  # fake 1 second ago
        kf.update([10, 10, 20, 20])
        assert kf.vx != 0  # moved right
        assert kf.vy != 0  # moved down


# ── ByteTrackLite ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestByteTrackLite:
    def test_empty_update(self):
        tracker = ByteTrackLite()
        result = tracker.update([])
        assert result == []

    def test_create_tracks(self):
        tracker = ByteTrackLite(min_hits=1)
        dets = [
            {"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0, "class_name": "person"},
            {"xyxy": [200, 200, 300, 300], "conf": 0.8, "cls": 56, "class_name": "chair"},
        ]
        result = tracker.update(dets)
        assert len(result) == 2
        assert all(isinstance(t, TrackedObject) for t in result)
        assert result[0].track_id != result[1].track_id

    def test_track_persistence(self):
        tracker = ByteTrackLite(min_hits=1)
        dets = [{"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0}]
        r1 = tracker.update(dets)
        tid = r1[0].track_id

        # Same position -> same track
        r2 = tracker.update(dets)
        assert r2[0].track_id == tid
        assert r2[0].frames_seen == 2

    def test_track_removal(self):
        tracker = ByteTrackLite(min_hits=1, max_age=2)
        dets = [{"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0}]
        tracker.update(dets)

        # No detections for max_age + 1 frames
        tracker.update([])
        tracker.update([])
        tracker.update([])
        assert tracker.track_count == 0

    def test_reset(self):
        tracker = ByteTrackLite(min_hits=1)
        tracker.update([{"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0}])
        assert tracker.track_count >= 1
        tracker.reset()
        assert tracker.track_count == 0

    def test_velocity_computation(self):
        tracker = ByteTrackLite(min_hits=1, iou_threshold=0.1)
        dets1 = [{"xyxy": [0, 0, 100, 100], "conf": 0.9, "cls": 0}]
        tracker.update(dets1)

        # Move object slightly (enough IoU overlap to match)
        dets2 = [{"xyxy": [20, 20, 120, 120], "conf": 0.9, "cls": 0}]
        result = tracker.update(dets2)
        assert len(result) == 1
        # Velocity should be non-zero (moved 20px in both axes)
        assert result[0].velocity[0] != 0 or result[0].velocity[1] != 0
